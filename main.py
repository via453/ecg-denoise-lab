"""
=============================================================================
实时 ECG 去噪服务器 (Real-time ECG Denoising Server)
=============================================================================
功能:
  1. 从串口读取下位机发送的 ECG 数据（或使用内置模拟信号）
  2. 应用 FFT 门控 / 陷波+带通 实时去噪
  3. 通过 SSE (Server-Sent Events) 推送到浏览器
  4. 实时显示去噪前后的波形

使用方式:
  # 模拟模式（无需硬件）:
  python main.py

  # 串口模式（需要硬件）:
  python main.py --port COM37 --baud 115200

依赖: flask, numpy, scipy  (pyserial 可选，默认用 ctypes 调用 Windows API)
=============================================================================
"""

import argparse
import json
import math
import os
import queue
import struct
import sys
import threading
import time
import numpy as np
from flask import Flask, Response, jsonify, render_template_string
from scipy import signal as scipy_signal
from scipy.linalg import svd
try:
    from sklearn.decomposition import PCA, FastICA
    _HAS_SKLEARN = True
except ImportError:
    print("  [警告] sklearn 未安装，PCA/ICA 方法将回退到带通滤波")
    PCA = None  # type: ignore
    FastICA = None  # type: ignore
    _HAS_SKLEARN = False
import serial

# ============================================================================
# 全局配置
# ============================================================================
FS = 250                     # 采样率 (Hz) — 与 kai4.csv 一致
WINDOW_SIZE = 128             # FFT 窗口大小（~0.5s，平衡延迟与分辨率）
HOP_SIZE = 8                  # 每次处理的步进（~32ms 更新一次）
FILTER_LOW = 0.5              # 带通下限 (Hz)
FILTER_HIGH = 40              # 带通上限 (Hz)
TRANS_BAND = 2.0              # FFT 门控过渡带宽度 (Hz)
VISIBLE_SECONDS = 5            # 网页端显示的窗口时长 (s)
BUFFER_SECONDS = 10           # 内部环形缓冲区时长 (s)
SIM_HR_BPM = 72               # 模拟信号的默认心率
SIM_NOISE_LEVEL = 0.6         # 模拟信号的噪声强度
SERVER_PORT = 8080            # Web 服务端口

# ============================================================================
# 环形缓冲区
# ============================================================================
class RingBuffer:
    """固定大小环形缓冲区，用于流式数据存储。"""

    def __init__(self, size):
        self.size = size
        self.data = np.zeros(size, dtype=np.float64)
        self.pos = 0
        self.count = 0

    def append(self, value):
        self.data[self.pos] = value
        self.pos = (self.pos + 1) % self.size
        if self.count < self.size:
            self.count += 1

    def extend(self, values):
        for v in values:
            self.append(v)

    def get_recent(self, n):
        """返回最近的 n 个采样点（时间顺序）。"""
        n = min(n, self.count)
        if n == 0:
            return np.array([])
        idx = (self.pos - n) % self.size
        if idx + n <= self.size:
            return self.data[idx:idx + n].copy()
        else:
            return np.concatenate([
                self.data[idx:],
                self.data[:idx + n - self.size],
            ])

    def get_all(self):
        return self.get_recent(self.count)


# ============================================================================
# 实时 FFT 门控处理器（重叠保留法 Overlap-Add）
# ============================================================================
class FFTGateProcessor:
    """
    基于重叠保留法的实时 FFT 门控滤波器。

    原理:
      1. 维护一个长度为 WINDOW_SIZE 的滑窗
      2. 每收集到 HOP_SIZE 个新样本，对滑窗数据做 FFT 门控
      3. 取滤波结果的后 HOP_SIZE 个样本作为输出
      4. 窗口函数消除边界伪影

    - FFT Gate 0.5-40Hz （总冠军方法移植）
    - 也可切换为 IIR 陷波+带通（延迟更低）
    """

    METHOD_FFT_GATE = "fft_gate"
    METHOD_NOTCH_BP = "notch_bp"
    METHOD_RAW = "raw"

    def __init__(self, window_size=WINDOW_SIZE, hop_size=HOP_SIZE, fs=FS):
        self.window_size = window_size
        self.hop_size = hop_size
        self.fs = fs
        self.buffer = np.zeros(window_size)
        self.olap_counter = 0
        self.method = self.METHOD_FFT_GATE

        # 预计算 IIR notch+bandpass 系数（仅计算一次）
        nyq = fs / 2.0
        self._b_band, self._a_band = scipy_signal.butter(
            4, [FILTER_LOW / nyq, FILTER_HIGH / nyq], btype='band'
        )
        self._b_notch50, self._a_notch50 = scipy_signal.iirnotch(50, 30, fs)
        self._b_notch100, self._a_notch100 = scipy_signal.iirnotch(100, 30, fs)
        # IIR 状态（用于连续滤波）
        self._zi_band = None
        self._zi_n50 = None
        self._zi_n100 = None

    def set_method(self, method):
        """切换处理方法。"""
        if method in (self.METHOD_FFT_GATE, self.METHOD_NOTCH_BP, self.METHOD_RAW):
            self.method = method
            # 重置 IIR 状态
            self._zi_band = None
            self._zi_n50 = None
            self._zi_n100 = None

    def _apply_fft_gate(self, x):
        """对窗口数据应用 FFT 门控。"""
        N = len(x)
        Xf = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(N, d=1 / self.fs)
        mask = np.ones_like(freqs, dtype=np.float64)
        # 阻带
        mask[freqs < FILTER_LOW] = 0.0
        mask[freqs > FILTER_HIGH] = 0.0
        # 平滑过渡
        il = (freqs >= FILTER_LOW) & (freqs < FILTER_LOW + TRANS_BAND)
        mask[il] = (freqs[il] - FILTER_LOW) / TRANS_BAND
        ih = (freqs > FILTER_HIGH - TRANS_BAND) & (freqs <= FILTER_HIGH)
        mask[ih] = (FILTER_HIGH - freqs[ih]) / TRANS_BAND
        return np.fft.irfft(Xf * mask, n=N)

    def _apply_notch_bp(self, x):
        """对窗口数据应用 IIR 陷波+带通（利用滤波器状态实现连续处理）。"""
        # 50Hz 陷波
        if self._zi_n50 is None:
            self._zi_n50 = scipy_signal.lfilter_zi(self._b_notch50, self._a_notch50) * x[0]
        out, self._zi_n50 = scipy_signal.lfilter(
            self._b_notch50, self._a_notch50, x, zi=self._zi_n50
        )
        # 100Hz 陷波
        if self._zi_n100 is None:
            self._zi_n100 = scipy_signal.lfilter_zi(self._b_notch100, self._a_notch100) * out[0]
        out, self._zi_n100 = scipy_signal.lfilter(
            self._b_notch100, self._a_notch100, out, zi=self._zi_n100
        )
        # 带通
        if self._zi_band is None:
            self._zi_band = scipy_signal.lfilter_zi(self._b_band, self._a_band) * out[0]
        out, self._zi_band = scipy_signal.lfilter(
            self._b_band, self._a_band, out, zi=self._zi_band
        )
        return out

    def process(self, new_samples):
        """
        处理新到达的样本。
        参数:
            new_samples: list[float] 新样本
        返回:
            list[float] 处理后的输出样本（可能为空）
        """
        output = []
        for s in new_samples:
            # 滑窗推进
            self.buffer = np.roll(self.buffer, -1)
            self.buffer[-1] = s
            self.olap_counter += 1

            if self.olap_counter >= self.hop_size:
                self.olap_counter = 0

                # 对当前窗口做处理
                if self.method == self.METHOD_FFT_GATE:
                    filtered = self._apply_fft_gate(self.buffer)
                elif self.method == self.METHOD_NOTCH_BP:
                    filtered = self._apply_notch_bp(self.buffer)
                else:  # RAW
                    filtered = self.buffer.copy()

                # 取后 HOP_SIZE 个样本输出
                output.extend(filtered[-self.hop_size:].tolist())

        return output


# ============================================================================
# 实时 R 峰检测器
# ============================================================================
class PeakDetector:
    """
    基于自适应阈值的实时 R 峰检测。
    维护一个短期基线，动态调整阈值。
    """

    def __init__(self, fs=FS, refractory_ms=200):
        self.fs = fs
        self.refractory = int(refractory_ms * fs / 1000)
        self.since_last_peak = 0
        self.peaks = []
        self.signal_buffer = []
        self.threshold = 0.0
        # 峰值追踪
        self.peak_trace = 0.5
        self.noise_trace = 0.0
        self.alpha_peak = 0.05
        self.alpha_noise = 0.05

    def detect(self, value):
        """
        对单个新样本检测 R 峰。
        返回: True=检测到 R 峰, False=无
        """
        self.signal_buffer.append(value)
        self.since_last_peak += 1

        # 需要足够的基线
        if len(self.signal_buffer) < int(0.5 * self.fs):
            self.peak_trace = max(self.peak_trace, abs(value) * 0.6)
            return False

        # 滑动窗口峰值检测
        window = np.array(self.signal_buffer[-int(0.1 * self.fs):])
        local_max = np.max(np.abs(window))
        local_idx = len(self.signal_buffer) - len(window) + np.argmax(np.abs(window))

        # 更新追踪
        if local_max > self.peak_trace:
            self.peak_trace = (1 - self.alpha_peak) * self.peak_trace + self.alpha_peak * local_max
        else:
            self.noise_trace = (1 - self.alpha_noise) * self.noise_trace + self.alpha_noise * local_max

        # 动态阈值
        self.threshold = self.noise_trace + 0.5 * (self.peak_trace - self.noise_trace)

        # 判定
        if (local_max > self.threshold
                and self.since_last_peak > self.refractory
                and local_idx == len(self.signal_buffer) - 1):
            self.peaks.append(len(self.signal_buffer) - 1)
            self.since_last_peak = 0
            return True
        return False

    def get_heart_rate(self):
        """返回当前心率估计 (bpm)，或 0。"""
        if len(self.peaks) < 2:
            return 0.0
        recent = self.peaks[-min(10, len(self.peaks)):]
        if len(recent) < 2:
            return 0.0
        intervals = np.diff(recent)
        mean_rr = np.mean(intervals) / self.fs  # 秒
        if mean_rr > 0:
            return 60.0 / mean_rr
        return 0.0


# ============================================================================
# 批处理器 — 10秒窗口全方法分析（移植自 kai4_exploration.py）
# ============================================================================
class BatchProcessor:
    """
    对 10s ECG 窗口运行多种去噪方法，自动选最优。
    
    方法列表:
    - FFT Gate 0.5-40Hz（冠军算法）
    - Notch 50/100/150Hz + 带通
    - 子带/小波去噪
    - SSA 低秩近似
    - PCA 时延嵌入
    - ICA + ZCR 选择
    - LMS 自适应滤波
    - 等权集成（前5种）
    """
    
    def __init__(self, fs=FS):
        self.fs = fs
        self.methods = {}  # name -> {'signal': np.array, 'metrics': dict}
    
    # ---- Helper: R峰检测 ----
    def detect_r_peaks(self, signal):
        """匹配 kai4 的 R 峰检测算法。"""
        from scipy.signal import find_peaks
        fs = self.fs
        nyq = fs / 2.0
        b, a = scipy_signal.butter(2, [5/nyq, 20/nyq], btype='band')
        filtered = scipy_signal.filtfilt(b, a, signal)
        abs_filt = np.abs(filtered)
        threshold = 0.5 * np.percentile(abs_filt, 95)
        min_dist = int(0.3 * fs)
        peaks, _ = find_peaks(abs_filt, height=threshold, distance=min_dist)
        if len(peaks) < 3:
            peaks, _ = find_peaks(abs_filt, height=threshold*0.4, distance=int(0.2*fs))
        margin = int(0.2*fs)
        peaks = peaks[(peaks >= margin) & (peaks < len(signal)-margin)]
        return peaks
    
    # ---- Helper: 带通 + 陷波 ----
    def _bandpass(self, signal, low=0.5, high=40, order=4):
        nyq = self.fs / 2.0
        b, a = scipy_signal.butter(order, [low/nyq, high/nyq], btype='band')
        return scipy_signal.filtfilt(b, a, signal)
    
    def _notch(self, signal, freq=50, q=30):
        b, a = scipy_signal.iirnotch(freq, q, self.fs)
        return scipy_signal.filtfilt(b, a, signal)
    
    def _notch_bp(self, signal):
        """Notch 50/100/150Hz + Bandpass 0.5-40Hz"""
        sig = self._notch(signal, 50, 30)
        sig = self._notch(sig, 100, 30)
        return self._bandpass(sig, 0.5, 40)
    
    # ---- 方法1: FFT Gate 0.5-40Hz ----
    def apply_fft_gate(self, signal):
        N = len(signal)
        Xf = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(N, d=1/self.fs)
        low, high, trans = 0.5, 40, 2.0
        mask = np.ones_like(freqs)
        mask[freqs < low] = 0
        il = (freqs >= low) & (freqs < low + trans)
        mask[il] = (freqs[il] - low) / trans
        mask[freqs > high] = 0
        ih = (freqs > high - trans) & (freqs <= high)
        mask[ih] = (high - freqs[ih]) / trans
        return np.fft.irfft(Xf * mask, n=N)
    
    # ---- 方法2: 陷波 + 带通 ----
    def apply_notch_bp(self, signal):
        return self._notch_bp(signal)
    
    # ---- 方法3: 子带/小波去噪 ----
    def apply_subband(self, signal, n_levels=4, threshold_scale=1.0):
        N = len(signal)
        nyq = self.fs / 2.0
        details = []
        current = signal.copy()
        for level in range(n_levels):
            high = self.fs / (2**(level+2))
            if high < 2:
                break
            b, a = scipy_signal.butter(4, high/nyq, btype='low')
            lp = scipy_signal.filtfilt(b, a, current)
            det = current - lp
            sigma = np.median(np.abs(det)) / 0.6745
            th = sigma * threshold_scale * np.sqrt(2 * np.log(N))
            det_th = np.sign(det) * np.maximum(np.abs(det) - th, 0)
            details.append(det_th)
            current = lp
        recon = current
        for det in reversed(details):
            recon = recon + det
        return recon
    
    # ---- 方法4: SSA 低秩 ----
    def apply_ssa(self, signal, L=120, rank=5):
        N = len(signal)
        L = min(L, N // 2)
        K = N - L + 1
        X = np.column_stack([signal[i:i+K] for i in range(L)])
        U, s, Vt = svd(X, full_matrices=False)
        rank = min(rank, len(s))
        Xd = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
        out = np.zeros(N)
        cnt = np.zeros(N)
        for p in range(K):
            for q in range(L):
                idx = p + q
                if idx < N:
                    out[idx] += Xd[p, q]
                    cnt[idx] += 1
        mask = cnt > 0
        out[mask] /= cnt[mask]
        return out
    
    # ---- 方法5: PCA 时延嵌入 ----
    def apply_pca(self, signal, n_comp=5, delay=50):
        try:
            N = len(signal)
            K = N - delay + 1
            X = np.column_stack([signal[i:i+K] for i in range(delay)])
            pca = PCA(n_components=min(n_comp, delay, K))
            scores = pca.fit_transform(X)
            Xr = pca.inverse_transform(scores)
            out = np.zeros(N)
            cnt = np.zeros(N)
            for i in range(delay):
                out[i:i+K] += Xr[:, i]
                cnt[i:i+K] += 1
            m = cnt > 0
            out[m] /= cnt[m]
            return out
        except Exception:
            return self._bandpass(signal, 0.5, 40)
    
    # ---- 方法6: ICA + ZCR选择 ----
    def apply_ica(self, signal, n_comp=5, delay=50):
        try:
            N = len(signal)
            K = N - delay + 1
            X = np.column_stack([signal[i:i+K] for i in range(delay)])
            ica = FastICA(n_components=min(n_comp, delay, K), random_state=42, max_iter=500)
            S = ica.fit_transform(X)
            zcrs = [np.sum(np.abs(np.diff(np.sign(S[:,i]))))/len(S[:,i]) for i in range(S.shape[1])]
            n_good = max(2, S.shape[1] // 2)
            good_ics = np.argsort(zcrs)[:n_good]
            Sc = np.zeros_like(S)
            for gi in good_ics:
                Sc[:, gi] = S[:, gi]
            Xr = ica.inverse_transform(Sc)
            out = np.zeros(N)
            cnt = np.zeros(N)
            for i in range(delay):
                out[i:i+K] += Xr[:, i]
                cnt[i:i+K] += 1
            m = cnt > 0
            out[m] /= cnt[m]
            return out
        except Exception:
            return self._bandpass(signal, 0.5, 40)
    
    # ---- 方法7: LMS 自适应 ----
    def apply_lms(self, signal, mu=0.01, order=30):
        N = len(signal)
        w = np.zeros(order)
        y = np.zeros(N)
        for n in range(order, N):
            xn = signal[n-order:n][::-1]
            y[n] = np.dot(w, xn)
            e = signal[n] - y[n]
            w += mu * e * xn
        return y
    
    # ---- 方法8: 等权集成 ----
    def apply_ensemble(self, signal):
        results = [
            self.apply_fft_gate(signal),
            self._notch_bp(signal),
            self.apply_subband(signal, 4, 1.0),
        ]
        return np.mean(results, axis=0)
    
    # ---- 指标计算 ----
    def compute_metrics(self, original, denoised):
        """参考 free 指标，移植自 kai4_exploration.py。"""
        peaks = self.detect_r_peaks(denoised)
        denoised = np.nan_to_num(denoised, nan=0.0, posinf=0.0, neginf=0.0)
        fs = self.fs
        
        # 平滑度
        smoothness = float(np.mean(np.diff(denoised, n=2)**2)) if len(denoised) > 2 else 0.0
        
        # 自相关 @ RR
        ac_rr = 0.0
        if len(peaks) >= 2:
            rr_median = int(np.median(np.diff(peaks)))
            den_norm = denoised - np.mean(denoised)
            if rr_median < len(denoised):
                num = np.sum(den_norm[:len(denoised)-rr_median] * den_norm[rr_median:])
                den = np.sum(den_norm**2) + 1e-10
                ac_rr = float(num / den)
        
        # Peak SNR
        peak_snr = 0.0
        if len(peaks) >= 3:
            peak_power = np.mean(np.abs(denoised[peaks])**2)
            non_peak = []
            for i in range(len(peaks)-1):
                mid_s = peaks[i] + int((peaks[i+1]-peaks[i])*0.3)
                mid_e = peaks[i] + int((peaks[i+1]-peaks[i])*0.7)
                non_peak.extend(range(mid_s, mid_e))
            non_power = np.mean(denoised[non_peak]**2) if non_peak else np.var(denoised)
            peak_snr = float(10 * np.log10(peak_power / (non_power + 1e-10)))
        
        # ECG Bandpower ratio
        freqs, psd = scipy_signal.welch(denoised, fs=fs, nperseg=min(1024, len(denoised)//4))
        ecg_band = (freqs >= 0.5) & (freqs <= 40)
        bp_ratio = float(np.sum(psd[ecg_band]) / (np.sum(psd) + 1e-10))
        
        # 峰一致性
        orig_peaks = len(self.detect_r_peaks(original))
        n_peaks = len(peaks)
        peak_consistency = float(min(n_peaks, orig_peaks) / max(n_peaks, orig_peaks, 1))
        
        return {
            'peak_snr': peak_snr,
            'autocorr_rr': ac_rr,
            'n_peaks': n_peaks,
            'smoothness': smoothness,
            'ecg_bandpower_ratio': bp_ratio,
            'peak_consistency': peak_consistency,
        }
    
    # ---- 主入口: 处理10秒窗口 ----
    def process_10s_window(self, raw_signal):
        """
        对 10s 原始 ECG 信号运行全部 8 种方法，选综合最优。
        返回: {
            'best_signal': np.array,        # 最优去噪信号
            'best_method': str,              # 方法名
            'best_metrics': dict,            # 对应指标
            'all_methods': dict,             # {方法名: {signal, metrics}}
        }
        """
        self.methods = {}
        x = np.asarray(raw_signal, dtype=np.float64)
        
        # 运行各方法
        results = [
            ('FFT Gate (0.5-40Hz)', self.apply_fft_gate(x)),
            ('Notch+BP', self.apply_notch_bp(x)),
            ('Subband (L4, mild)', self.apply_subband(x, 4, 1.0)),
            ('SSA (rank=5)', self.apply_ssa(x, 120, 5)),
            ('PCA (r=5, τ=50)', self.apply_pca(x, 5, 50)),
            ('ICA (r=5, τ=50)', self.apply_ica(x, 5, 50)),
            ('LMS (μ=0.01)', self.apply_lms(x, 0.01, 30)),
            ('Ensemble', self.apply_ensemble(x)),
        ]
        
        for name, sig in results:
            metrics = self.compute_metrics(x, sig)
            self.methods[name] = {'signal': sig, 'metrics': metrics}
        
        # 计算复合评分选最优（和 kai4 一样的 z-score 方法）
        scores = {}
        for name, data in self.methods.items():
            m = data['metrics']
            score = m['peak_snr'] * 2 + m['autocorr_rr'] * 10 + m['peak_consistency'] * 5 + m['ecg_bandpower_ratio'] * 3 - m['smoothness'] * 0.01
            scores[name] = score
        
        best_name = max(scores, key=scores.get)
        
        return {
            'best_signal': self.methods[best_name]['signal'],
            'best_method': best_name,
            'best_metrics': self.methods[best_name]['metrics'],
            'all_methods': {name: {'metrics': self.methods[name]['metrics']} for name in self.methods},
        }


# ============================================================================
# 模拟 ECG 发生器（用于无硬件测试）
# ============================================================================
class SimulatedECG:
    """
    生成带有真实感 P-QRS-T 形态的模拟 ECG 信号，叠加与 kai4.csv 相似的噪声。

    噪声成分:
    - 50Hz 工频及 100Hz 谐波
    - 低频基线漂移 (0.2Hz)
    - 宽带高斯噪声
    - 心率可变性（模拟真实 RR 间期波动）
    """

    def __init__(self, fs=FS, hr_bpm=SIM_HR_BPM, noise_level=SIM_NOISE_LEVEL):
        self.fs = fs
        self.base_hr = hr_bpm
        self.noise_level = noise_level
        self.t = 0.0  # 模拟时间
        self.next_beat_time = 0.0

        # P-QRS-T 模板: (时间偏移_s, 幅值, 宽度_s)
        self.template = [
            (-0.22, 0.12, 0.030),   # P 波
            (-0.06, -0.08, 0.012),  # Q 波
            (0.00, 1.00, 0.010),    # R 波
            (0.04, -0.12, 0.015),   # S 波
            (0.30, 0.18, 0.055),    # T 波
        ]

        # 预计算 5s 的噪声基底（用于波形连续性）
        self.noise_len = int(5 * fs)
        self._init_noise()

    def _init_noise(self):
        """初始化噪声序列。"""
        np.random.seed(int(time.time() * 1000) % 10000)
        self._noise_buf = np.random.randn(self.noise_len)
        self._noise_pos = 0

    def _get_noise_sample(self):
        """获取一个噪声样本（包含工频、基线漂移、高斯噪声）。"""
        # 宽带噪声
        white = self._noise_buf[self._noise_pos] * self.noise_level * 0.3
        self._noise_pos = (self._noise_pos + 1) % self.noise_len

        # 50Hz 工频 + 100Hz 谐波
        pli = 0.15 * math.sin(2 * math.pi * 50 * self.t)
        pli += 0.08 * math.sin(2 * math.pi * 100 * self.t)

        # 基线漂移 (0.15-0.35Hz 慢波)
        baseline = 0.15 * math.sin(2 * math.pi * 0.2 * self.t)
        baseline += 0.08 * math.sin(2 * math.pi * 0.35 * self.t)

        return white + pli + baseline

    def generate(self, n_samples):
        """
        生成 n_samples 个模拟 ECG 样本。
        返回: list[float]
        """
        output = []
        for _ in range(n_samples):
            # 当前心跳时刻的高斯脉冲叠加
            value = 0.0

            # 确定下一个心跳的 RR 间期（含轻度变异性）
            if self.t >= self.next_beat_time:
                rr = 60.0 / self.base_hr
                jitter = np.random.normal(0, 0.02)  # 心率变异性
                rr = max(0.4, rr + jitter)
                self.next_beat_time = self.t + rr

            # 计算到最近几个心拍的偏移
            for beat_offset in [-1, 0, 1]:
                beat_t = self.next_beat_time
                # 如果有上一拍
                if beat_offset < 0:
                    beat_t = beat_t - 60.0 / self.base_hr

                dt = self.t - beat_t
                for pos, amp, width in self.template:
                    t_peak = dt - pos
                    value += amp * math.exp(-0.5 * (t_peak / width) ** 2)

            # 叠加噪声
            value += self._get_noise_sample()

            output.append(value)
            self.t += 1.0 / self.fs

        return output


# ============================================================================
# 串口读取器 (pyserial)
# ============================================================================
class SerialReader:
    """
    使用 pyserial 读取串口数据。
    适配 F1 二进制 13 字节 ECG 数据包协议。
    """

    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.buffer = b''

    def open(self):
        """打开串口连接。"""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
            )
            print(f"  [串口] {self.port} 已打开，波特率 {self.baudrate}")
            return True
        except serial.SerialException as e:
            raise IOError(f"无法打开串口 {self.port}，错误: {e}")

    def read_packet(self, timeout=5.0):
        """
        读取 F1 工程的 13 字节二进制 ECG 数据包。
        协议:
          Bytes 0-1:  0xAA 0xAA (帧头)
          Bytes 2-7:  填充/未使用
          Bytes 8-11: 32位有符号整数, 大端序 (ADC 原始值)
          Byte 12:    校验和 = sum(bytes[0:12]) & 0xFF
        返回: float (mV, 经 gain=1/1000 转换) 或 None (超时/错误)
        """
        start = time.time()
        while time.time() - start < timeout:
            # 扫描 0xAA 0xAA 帧头
            while len(self.buffer) >= 2:
                if self.buffer[0] == 0xAA and self.buffer[1] == 0xAA:
                    break
                self.buffer = self.buffer[1:]

            if len(self.buffer) >= 13:
                packet = self.buffer[:13]
                # 校验和验证
                if sum(packet[:12]) & 0xFF != packet[12]:
                    self.buffer = self.buffer[1:]
                    continue
                # 解析大端 32 位有符号整数 (bytes 8-11)
                raw_val = struct.unpack('>i', packet[8:12])[0]
                ecg_mv = raw_val * (1.0 / 1000)
                self.buffer = self.buffer[13:]
                return ecg_mv

            # 读取更多数据
            if self.ser and self.ser.is_open:
                try:
                    data = self.ser.read(256)
                    if data:
                        self.buffer += data
                except Exception:
                    time.sleep(0.005)
            else:
                time.sleep(0.005)

        return None

    def close(self):
        """关闭串口。"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"  [串口] {self.port} 已关闭")

    def __del__(self):
        self.close()


# ============================================================================
# 全局共享数据
# ============================================================================
raw_buffer = RingBuffer(int(BUFFER_SECONDS * FS))
proc_buffer = RingBuffer(int(BUFFER_SECONDS * FS))
data_queue = queue.Queue(maxsize=30)  # SSE 推送队列
status_info = {
    'running': False,
    'mode': 'simulation',
    'port': 'N/A',
    'fps': 0.0,
    'heart_rate': 0.0,
    'peak_snr': 0.0,
    'method': 'fft_gate',
    'samples_received': 0,
    'signal_level': 0.0,
}
status_lock = threading.Lock()

# ============================================================================
# 数据处理线程
# ============================================================================
def data_pipeline(mode='simulation', port='COM37', baudrate=115200):
    """
    主数据处理流水线（独立线程运行）。

    流程:
       串口/模拟器 → 环形缓冲区 → FFT去噪 → 峰值检测 → SSE推送
    """
    global raw_buffer, data_queue, status_info

    # 初始化各模块
    if mode == 'serial':
        try:
            reader = SerialReader(port, baudrate)
            reader.open()
            print(f"  [流水线] 串口模式: {port} @ {baudrate}")
        except Exception as e:
            print(f"  [错误] 串口打开失败: {e}")
            print(f"  [回退] 切换到模拟模式")
            mode = 'simulation'
            reader = None
            simulator = SimulatedECG(fs=FS)
            print(f"  [流水线] 模拟模式，心率={SIM_HR_BPM} bpm")
    else:
        reader = None
        simulator = SimulatedECG(fs=FS)
        print(f"  [流水线] 模拟模式，心率={SIM_HR_BPM} bpm")

    batch_processor = BatchProcessor(fs=FS)
    batch_buffer = []          # 累积原始数据，满2500点触发批处理
    batch_results = None       # 最近一次批处理结果
    batch_trigger_time = 0     # 上次触发的时间
    batch_lock = threading.Lock()
    batch_version = 0
    last_sent_version = -1
    batch_playback = np.array([])  # 当前批次的去噪信号（供流式播放）
    batch_playback_pos = 0         # 播放位置
    batch_r_peaks = []              # 当前批次的所有R峰位置（绝对索引）
    batch_r_amps = []
    batch_start_in_procdata = 0     # 当前批次在procData中的起始位置
    total_sent_samples = 0          # 累计发送的去噪样本总数

    with status_lock:
        status_info['mode'] = mode
        status_info['port'] = port if mode == 'serial' else 'SIM'
        status_info['running'] = True

    # 主循环
    last_push_time = time.time()
    frame_count = 0
    push_interval = 1.0 / 4   # ~250ms 推送一次
    BATCH_SIZE = int(FS * 10)  # 10秒 = 2500样本
    BATCH_INTERVAL = 5.0       # 每5秒触发一次分析

    try:
        while True:
            try:
                # ---- 读取样本 ----
                if mode == 'serial' and reader:
                    try:
                        value = reader.read_packet(timeout=0.5)
                    except Exception as e:
                        print(f"  [错误] 串口读取异常: {e}")
                        time.sleep(0.01)
                        continue
                    if value is None:
                        time.sleep(0.001)
                        continue
                    new_samples = [value]
                else:
                    # 模拟模式：按批次生成（每次生成 hop_size 个样本，模拟实时采集）
                    new_samples = simulator.generate(HOP_SIZE)
                    time.sleep(HOP_SIZE / FS)  # 模拟250Hz实时速率

                # ---- 写入原始缓冲区 ----
                raw_buffer.extend(new_samples)

                with status_lock:
                    status_info['samples_received'] += len(new_samples)
                    status_info['signal_level'] = float(np.std(raw_buffer.get_recent(FS * 2)))

                # ---- 批处理累积 ----
                batch_buffer.extend(new_samples)

                # 当累积够10秒 且 距离上次触发超过5秒 -> 后台处理
                if len(batch_buffer) >= BATCH_SIZE and (time.time() - batch_trigger_time) >= BATCH_INTERVAL:
                    batch_trigger_time = time.time()
                    # 取出当前批次（双缓冲：留下空列表继续收新数据）
                    current_batch = batch_buffer[:]
                    batch_buffer = []
                    # 后台线程运行分析
                    def run_batch(batch_data):
                        nonlocal batch_results, batch_version
                        try:
                            result = batch_processor.process_10s_window(np.array(batch_data, dtype=np.float64))
                            with batch_lock:
                                batch_results = result
                                batch_version += 1
                            print(f"  [批处理] 最优: {result['best_method']}  "
                                  f"SNR={result['best_metrics']['peak_snr']:.1f}dB  "
                                  f"HR={result['best_metrics'].get('n_peaks', 0)} peaks")
                        except Exception as e:
                            print(f"  [批处理] 错误: {e}")
                    t = threading.Thread(target=run_batch, args=(current_batch,), daemon=True)
                    t.start()

                # ---- SSE 推送（按时间间隔） ----
                now = time.time()
                if now - last_push_time >= push_interval:
                    # 原始信号10秒窗口
                    vis_n = int(VISIBLE_SECONDS * 2 * FS)
                    raw_vis = raw_buffer.get_recent(vis_n).tolist()

                    # 批处理结果
                    batch_proc = []
                    r_peak_indices = []
                    r_peak_amps = []
                    method_name = 'collecting...'
                    batch_snr = 0.0
                    hr_bpm = 0.0
                    n_peaks = 0

                    with batch_lock:
                        if batch_results is not None and batch_version > 0:
                            method_name = batch_results['best_method']
                            batch_snr = round(batch_results['best_metrics'].get('peak_snr', 0), 1)
                            hr_bpm = round(hr_bpm_from_peaks(batch_results['best_signal'], FS), 1)
                            # 新批次完成：加载播放缓冲，R峰只算一次（用绝对索引）
                            if batch_version != last_sent_version:
                                batch_playback = batch_results['best_signal']
                                batch_playback_pos = 0
                                batch_start_in_procdata = total_sent_samples
                                pks, amps = batch_detect_r_peaks(batch_playback, FS)
                                # R峰存为绝对索引（batch_start偏移 + 批次内位置）
                                batch_r_peaks = [batch_start_in_procdata + int(i) for i in pks]
                                batch_r_amps = [float(a) for a in amps]
                                n_peaks = len(batch_r_peaks)
                                last_sent_version = batch_version

                    # ---- 流式播放批处理结果（每次推送发一小块，实现平滑滚动） ----
                    chunk_size = int(FS * push_interval)  # ~62点/次 = 250Hz实时速率
                    if len(batch_playback) > 0 and batch_playback_pos < len(batch_playback):
                        end = min(batch_playback_pos + chunk_size, len(batch_playback))
                        batch_proc = batch_playback[batch_playback_pos:end].tolist()
                        batch_playback_pos = end
                        total_sent_samples += len(batch_proc)
                        # 发送当前批次的所有R峰（前端可见性检查会自动过滤）
                        r_peak_indices = batch_r_peaks[:]
                        r_peak_amps = batch_r_amps[:]

                    # 检测信号是否存活
                    signal_alive = float(np.max(np.abs(raw_vis[-FS:]))) > 0.01 if len(raw_vis) >= FS else True

                    packet = {
                        'raw': raw_vis[-min(vis_n, len(raw_vis)):],
                        'batch_id': batch_version,
                        'proc': batch_proc[-min(vis_n, len(batch_proc)):] if batch_proc else [],
                        'r_peaks': {
                            'indices': r_peak_indices,
                            'amplitudes': r_peak_amps,
                        },
                        'method': method_name,
                        't_start': now - vis_n / FS,
                        't_end': now,
                        'metrics': {
                            'heart_rate': hr_bpm,
                            'peak_snr': batch_snr,
                            'signal_level': round(float(np.std(raw_vis[-FS:] if len(raw_vis) >= FS else [0])), 3),
                            'method': method_name,
                            'signal_alive': signal_alive,
                            'n_peaks': n_peaks,
                        },
                    }

                    try:
                        data_queue.put_nowait(packet)
                    except queue.Full:
                        try:
                            data_queue.get_nowait()
                            data_queue.put_nowait(packet)
                        except queue.Full:
                            pass

                    with status_lock:
                        status_info['heart_rate'] = hr_bpm
                        status_info['peak_snr'] = batch_snr
                        status_info['method'] = method_name

                    frame_count += 1
                    last_push_time = now

                # 串口模式不要跑太快
                if mode == 'serial':
                    time.sleep(0.001)

            except Exception as e:
                print(f"\n  [流水线] 错误: {e}")
                import traceback
                traceback.print_exc()

    except KeyboardInterrupt:
        print("\n  [流水线] 用户中断")
    finally:
        if reader:
            reader.close()
        with status_lock:
            status_info['running'] = False
        print("  [流水线] 已停止")


# ============================================================================
# Flask Web 服务 + SSE
# ============================================================================
app = Flask(__name__)

# 读取 HTML 模板
HERE = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(HERE, 'index.html')
with open(HTML_PATH, 'r', encoding='utf-8') as f:
    HTML_CONTENT = f.read()


@app.route('/')
def index():
    return render_template_string(HTML_CONTENT)


@app.route('/stream')
def stream():
    """SSE (Server-Sent Events) 端点，推送实时数据。"""
    def generate():
        while True:
            try:
                packet = data_queue.get(timeout=2.0)
                yield f"data: {json.dumps(packet)}\n\n"
            except queue.Empty:
                # 心跳包维持连接
                with status_lock:
                    running = status_info['running']
                yield f"data: {json.dumps({'ping': True, 'running': running})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
        },
    )


@app.route('/status')
def get_status():
    """返回当前状态（JSON）。"""
    with status_lock:
        return jsonify(status_info)


@app.route('/set_method', methods=['POST'])
def set_method():
    """切换处理方法（由前端 AJAX 调用）。"""
    from flask import request
    data = request.get_json()
    method = data.get('method', 'fft_gate')

    # 通过全局变量传递方法切换指令
    global _method_change
    if method in ('fft_gate', 'notch_bp', 'raw'):
        _method_change = method
        return jsonify({'status': 'ok', 'method': method})
    return jsonify({'status': 'error', 'message': f'未知方法: {method}'}), 400


# 全局方法切换信号
_method_change = None

def check_method_change(processor):
    """检查是否有方法切换请求（由流水线线程调用）。"""
    global _method_change
    if _method_change:
        processor.set_method(_method_change)
        print(f"  [方法切换] → {_method_change}")
        _method_change = None


# ============================================================================
# 主入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='实时 ECG 去噪服务器 — 串口读取 + FFT 门控 + Web 波形显示'
    )
    parser.add_argument('--port', type=str, default=None,
                        help='串口号，如 COM37（不指定则使用模拟模式）')
    parser.add_argument('--baud', type=int, default=115200,
                        help='串口波特率（默认 115200）')
    parser.add_argument('--http-port', type=int, default=SERVER_PORT,
                        help=f'Web 服务端口（默认 {SERVER_PORT}）')
    args = parser.parse_args()

    mode = 'simulation' if args.port is None else 'serial'
    port_str = args.port if args.port else 'COM37'
    baud = args.baud

    print("=" * 60)
    print("  实时 ECG 去噪服务器")
    print("=" * 60)
    print(f"  模式: {mode}")
    if args.port:
        print(f"  串口: {port_str} @ {baud} baud")
    print(f"  采样率: {FS} Hz")
    print(f"  去噪方法: FFT Gate ({FILTER_LOW}-{FILTER_HIGH} Hz)")
    print(f"  Web 端口: {args.http_port}")
    print("=" * 60)

    # 启动数据流水线线程
    pipeline_thread = threading.Thread(
        target=data_pipeline,
        args=(mode, port_str, baud),
        daemon=True,
    )
    pipeline_thread.start()

    # 给流水线一点时间初始化
    time.sleep(0.5)

    # 启动 Flask
    print(f"\n  >> Web UI: http://localhost:{args.http_port}")
    print(f"  >> Ctrl+C to stop\n")

    # 注册方法切换检查器（由 Flask 的 before_request 触发）
    @app.before_request
    def _check_method():
        pass  # 方法切换由流水线线程检查全局变量

    try:
        app.run(host='0.0.0.0', port=args.http_port, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n  服务停止")


# ============================================================================
# HR from peaks (matches kai4_exploration.py approach)
# ============================================================================
def hr_bpm_from_peaks(signal, fs=FS):
    """从信号中检测R峰并计算心率(bpm)。"""
    from scipy.signal import butter, filtfilt, find_peaks
    if len(signal) < int(2 * fs):
        return 0.0
    x = np.asarray(signal, dtype=float)
    nyq = fs / 2.0
    b, a = butter(2, [5/nyq, 20/nyq], btype='band')
    filtered = filtfilt(b, a, x)
    abs_filt = np.abs(filtered)
    threshold = 0.5 * np.percentile(abs_filt, 95)
    min_dist = int(0.3 * fs)
    peaks, _ = find_peaks(abs_filt, height=threshold, distance=min_dist)
    if len(peaks) < 3:
        return 0.0
    rr_ms = np.diff(peaks) / fs * 1000
    valid_rr = rr_ms[(rr_ms > 300) & (rr_ms < 1200)]
    if len(valid_rr) < 2:
        return 0.0
    return float(60000.0 / np.median(valid_rr))


# ============================================================================
# Batch R-Peak Detection (matches kai4_exploration.py approach)
# ============================================================================
def batch_detect_r_peaks(signal, fs=FS):
    from scipy.signal import butter, filtfilt, find_peaks
    if len(signal) < int(2 * fs):
        return [], []
    x = np.asarray(signal, dtype=float)
    nyq = fs / 2.0
    b, a = butter(2, [5/nyq, 20/nyq], btype='band')
    filtered = filtfilt(b, a, x)
    abs_filt = np.abs(filtered)
    threshold = 0.5 * np.percentile(abs_filt, 95)
    min_dist = int(0.3 * fs)
    peaks, _ = find_peaks(abs_filt, height=threshold, distance=min_dist)
    refined = []
    amps = []
    search_win = int(0.08 * fs)
    for p in peaks:
        start = max(0, p - search_win)
        end = min(len(x), p + search_win)
        best = start + np.argmax(np.abs(x[start:end]))
        refined.append(best)
        amps.append(float(x[best]))
    return refined, amps


if __name__ == '__main__':
    main()
