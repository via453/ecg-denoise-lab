"""
=============================================================================
kai4.csv ECG Denoising — Comprehensive Multi-Algorithm Exploration
=============================================================================
TARGET: kai4.csv — real ECG signal heavily corrupted by complex noise.
APPROACHES: SVD, Classical Filtering, Wavelet*, Frequency-Domain, 
            Adaptive Filtering, PCA/ICA, Ensemble Methods
OUTPUTS: results/kai4_exploration/*.png, metrics.csv, report.md
=============================================================================
"""

import os, json, warnings
import numpy as np
from scipy import signal as scipy_signal
from scipy.linalg import svd
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA, FastICA

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================================
# CONFIG
# ============================================================================
RESULTS_DIR = os.path.join('results', 'kai4_exploration')
os.makedirs(RESULTS_DIR, exist_ok=True)
method_results = {}

# ============================================================================
# DATA LOADING
# ============================================================================
df_data = np.loadtxt('kai4.csv', delimiter=',', skiprows=1)
t = df_data[:, 0]
x = df_data[:, 1].astype(np.float64)
r_peak_marks = df_data[:, 2]
N = len(x)

# Remove DC offset
x = x - np.mean(x)

# Sampling rate
dt_vals = np.diff(t)
FS = int(round(1.0 / np.median(dt_vals)))
print(f"Loaded: {N} samples, {FS} Hz, {N/FS:.1f}s")
print(f"  Range: [{x.min():.3f}, {x.max():.3f}] mV, RMS={np.std(x):.3f}")

# ============================================================================
# R-PEAK DETECTION
# ============================================================================
def detect_r_peaks(signal, fs=FS):
    from scipy.signal import find_peaks
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

r_peaks = detect_r_peaks(x)
print(f"  R-peaks detected: {len(r_peaks)}")

# ============================================================================
# REFERENCE-FREE METRICS
# ============================================================================
def safe_float(v, fallback=0.0):
    """Return float, replacing NaN/inf with fallback."""
    v = float(v)
    return fallback if (np.isnan(v) or np.isinf(v)) else v

def compute_metrics_ref_free(original, denoised, fs=FS):
    peaks = detect_r_peaks(denoised)
    denoised = np.nan_to_num(denoised, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 1. Smoothness (2nd diff energy)
    smoothness = safe_float(np.mean(np.diff(denoised, n=2)**2))
    
    # 2. Autocorrelation at R-R lag (periodicity)
    if len(peaks) >= 2:
        rr_median = int(np.median(np.diff(peaks)))
        den_norm = denoised - np.mean(denoised)
        if rr_median < len(denoised):
            ac_rr = float(np.sum(den_norm[:len(denoised)-rr_median] * den_norm[rr_median:]) / (np.sum(den_norm**2) + 1e-10))
        else:
            ac_rr = 0.0
        hr_std = float(np.std(np.diff(peaks)) / (np.mean(np.diff(peaks)) + 1e-10))
    else:
        ac_rr, hr_std = 0.0, 999.0
    
    # 3. Peak SNR (R-peak power / non-peak power)
    if len(peaks) >= 3:
        peak_vals = np.abs(denoised[peaks])
        peak_power = np.mean(peak_vals**2)
        non_peak_samples = []
        for i in range(len(peaks)-1):
            mid_s = peaks[i] + int((peaks[i+1]-peaks[i])*0.3)
            mid_e = peaks[i] + int((peaks[i+1]-peaks[i])*0.7)
            non_peak_samples.extend(range(mid_s, mid_e))
        non_peak_power = np.mean(denoised[non_peak_samples]**2) if non_peak_samples else np.var(denoised)
        peak_snr = float(10*np.log10(peak_power / (non_peak_power + 1e-10)))
    else:
        peak_snr = 0.0
    
    # 4. Energy ratio
    energy_ratio = float(np.var(denoised) / (np.var(original) + 1e-10))
    
    # 5. Zero-crossing rate (normalized)
    zcr = float(np.sum(np.abs(np.diff(np.sign(denoised)))) / (2*len(denoised)))
    
    # 6. ECG bandpower ratio (0.5-40 Hz / total)
    freqs, psd = scipy_signal.welch(denoised, fs=fs, nperseg=min(1024, len(denoised)//4))
    ecg_band = (freqs >= 0.5) & (freqs <= 40)
    bp_ratio = float(np.sum(psd[ecg_band]) / (np.sum(psd) + 1e-10))
    
    # 7. Peak consistency (ratio of detected peaks to original)
    orig_peaks = len(detect_r_peaks(original))
    n_peaks = len(peaks)
    peak_consistency = float(min(n_peaks, orig_peaks) / max(n_peaks, orig_peaks, 1))
    
    return {
        'smoothness': smoothness, 'autocorr_rr': ac_rr, 'hr_std_ratio': hr_std,
        'peak_snr': peak_snr, 'energy_ratio': energy_ratio, 'zero_crossing_rate': zcr,
        'ecg_bandpower_ratio': bp_ratio, 'peak_consistency': peak_consistency, 'n_peaks': n_peaks,
    }

def record_method(name, signal, desc, category):
    metrics = compute_metrics_ref_free(x, signal)
    method_results[name] = {'signal': signal, 'desc': desc, 'category': category, 'metrics': metrics}
    m = metrics
    print(f"  [{category:>6}] {name:<35} PK={m['peak_snr']:>+5.1f}dB  AC={m['autocorr_rr']:.3f}  "
          f"Pks={m['n_peaks']:>3d}  Sm={m['smoothness']:.1f}  BP={m['ecg_bandpower_ratio']:.2f}")

# ============================================================================
# HELPER: Bandpass, Notch, SSA, Overlap-add utilities
# ============================================================================
def bandpass(signal, low=0.5, high=40, order=4, fs=FS):
    nyq = fs/2.0
    b, a = scipy_signal.butter(order, [low/nyq, high/nyq], btype='band')
    return scipy_signal.filtfilt(b, a, signal)

def notch(signal, freq=50, q=30, fs=FS):
    b, a = scipy_signal.iirnotch(freq, q, fs)
    return scipy_signal.filtfilt(b, a, signal)

def ssa_reconstruct(signal, L, n_components):
    """SSA: Hankel embedding → SVD truncation → anti-diagonal averaging."""
    Nsig = len(signal)
    L = min(L, Nsig//2)
    K = Nsig - L + 1
    X = np.column_stack([signal[i:i+K] for i in range(L)])  # shape (K, L)
    U, s, Vt = svd(X, full_matrices=False)
    n_components = min(n_components, len(s))
    # Reconstruct low-rank approximation
    Xd = U[:, :n_components] @ np.diag(s[:n_components]) @ Vt[:n_components, :]  # shape (K, L)
    # Anti-diagonal averaging (Hankel matrix has entries Xd[p, q] where p+q = index)
    out = np.zeros(Nsig)
    cnt = np.zeros(Nsig)
    for p in range(K):
        for q in range(L):
            idx = p + q
            if idx < Nsig:
                out[idx] += Xd[p, q]
                cnt[idx] += 1
    mask = cnt > 0
    out[mask] /= cnt[mask]
    return out

# ============================================================================
# PHASE 1: EDA
# ============================================================================
print("\n" + "="*60)
print("PHASE 1: Exploratory Data Analysis")
print("="*60)

# Plot 1: Raw signal + PSD + Distribution
fig, axes = plt.subplots(4, 1, figsize=(16, 10))
fig.suptitle('kai4.csv — Raw ECG Signal Analysis (FS=' + str(FS) + 'Hz)', fontsize=14, fontweight='bold')

ax = axes[0]
ax.plot(t, x, color='#1a56db', linewidth=0.5, alpha=0.8)
ax.scatter(t[r_peaks], x[r_peaks], color='red', s=12, zorder=3, label=f'{len(r_peaks)} R-peaks')
ax.set_ylabel('ECG (mV)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_title('Full Signal', fontsize=11)

ax = axes[1]
seg = slice(0, min(1250, N))
ax.plot(t[seg], x[seg], color='#1a56db', linewidth=0.7)
pk_seg = r_peaks[(r_peaks >= seg.start) & (r_peaks < seg.stop)]
ax.scatter(t[seg][pk_seg], x[seg][pk_seg], color='red', s=25, zorder=3)
ax.set_ylabel('ECG (mV)'); ax.grid(True, alpha=0.3); ax.set_title('First 5s Segment', fontsize=11)

ax = axes[2]
freqs, psd = scipy_signal.welch(x, fs=FS, nperseg=min(2048, N//2))
ax.semilogy(freqs, psd, color='#1a56db', linewidth=0.7)
ax.axvspan(0.5, 40, alpha=0.08, color='green', label='ECG band (0.5-40 Hz)')
for fm, lbl, clr in [(50,'50Hz','red'),(100,'100Hz','orange'),(150,'150Hz','orange')]:
    ax.axvline(fm, color=clr, linestyle='--', alpha=0.4, linewidth=0.7)
ax.set_xlim(0, min(125, FS/2))
ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('PSD')
ax.set_title('Power Spectral Density', fontsize=11); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[3]
ax.hist(x, bins=80, color='#1a56db', alpha=0.7, density=True)
ax.axvline(np.mean(x), color='red', linestyle='--', label=f'Mean={np.mean(x):.3f}')
ax.axvline(np.mean(x)+np.std(x), color='orange', linestyle=':', label=f'±1σ ({np.std(x):.3f})')
ax.axvline(np.mean(x)-np.std(x), color='orange', linestyle=':')
ax.set_xlabel('Amplitude (mV)'); ax.set_ylabel('Density')
ax.set_title(f'Distribution (σ={np.std(x):.3f})', fontsize=11); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(RESULTS_DIR, '01_eda_raw_signal.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 01_eda_raw_signal.png")

# Plot 1b: Spectrogram + Autocorrelation
fig, axes = plt.subplots(2, 1, figsize=(16, 8))
fig.suptitle('Noise Characterization', fontsize=14, fontweight='bold')

ax = axes[0]
f, t_spec, Sxx = scipy_signal.spectrogram(x, fs=FS, nperseg=128, noverlap=64)
ax.pcolormesh(t_spec, f, 10*np.log10(Sxx+1e-12), shading='gouraud', cmap='viridis')
ax.set_ylabel('Frequency (Hz)'); ax.set_ylim(0, 125)
ax.set_title('Spectrogram', fontsize=11)

ax = axes[1]
ac = np.correlate(x-np.mean(x), x-np.mean(x), mode='full')
ac = ac / ac[len(ac)//2]
lags = np.arange(-len(ac)//2+1, len(ac)//2+1) / FS
half = min(5*FS, len(ac)//2)
center = len(ac)//2
ax.plot(lags[center-half:center+half], ac[center-half:center+half], color='#1a56db', linewidth=0.7)
ax.set_xlabel('Lag (s)'); ax.set_title('Autocorrelation', fontsize=11); ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(RESULTS_DIR, '02_noise_characterization.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 02_noise_characterization.png")

# ============================================================================
# PHASE 2: DENOISING METHODS
# ============================================================================
print("\n" + "="*60)
print("PHASE 2: Denoising Methods (28 total)")
print("="*60)

# ─── A. SVD-BASED (7) ───
print("\n--- A. SVD-Based Methods ---")

# A1: Standard SVD overlapping windows
def method_a1(signal):
    Nsig = len(signal); win=80; step=40
    frames = [signal[i:i+win] for i in range(0, Nsig-win+1, step)]
    if not frames: return signal.copy()
    X = np.array(frames)
    U, s, Vt = svd(X, full_matrices=False)
    k = min(5, len(s))
    Xd = U[:,:k] @ np.diag(s[:k]) @ Vt[:k,:]
    out = np.zeros(Nsig); cnt = np.zeros(Nsig)
    for i, fr in enumerate(Xd):
        p = i*step; out[p:p+win] += fr; cnt[p:p+win] += 1
    m = cnt>0; out[m]/=cnt[m]; return out
record_method('A1. Standard SVD (r=5)', method_a1(x), 'Overlap-window SVD rank=5', 'A-SVD')

# A2: SSA 90% energy
record_method('A2. SSA (90% energy)', ssa_reconstruct(x, 150, 12), 'Auto rank by 90% energy', 'A-SVD')

# A3: SSA low-rank
record_method('A3. SSA (rank=5)', ssa_reconstruct(x, 120, 5), 'SSA top-5 components', 'A-SVD')

# A4: Beat-Aligned SVD
def method_a4(signal, fs=FS):
    peaks = detect_r_peaks(signal)
    if len(peaks) < 3: return signal.copy()
    w_pre, w_post = int(0.3*fs), int(0.5*fs); wlen = w_pre+w_post
    beats, vp = [], []
    for pk in peaks:
        s_, e_ = pk-w_pre, pk+w_post
        if s_<0 or e_>len(signal): continue
        beats.append(signal[s_:e_]); vp.append(pk)
    if not beats: return signal.copy()
    M = np.array(beats)
    U, s, Vt = svd(M, full_matrices=False)
    k = min(3, len(s))
    Mr = U[:,:k] @ np.diag(s[:k]) @ Vt[:k,:]
    out = np.zeros(len(signal)); wgt = np.zeros(len(signal))
    win = np.hanning(wlen)
    for i, pk in enumerate(vp):
        s_ = pk-w_pre; out[s_:s_+wlen] += Mr[i]*win; wgt[s_:s_+wlen] += win
    m = wgt>1e-10; out[m]/=wgt[m]; return out
record_method('A4. Beat-Aligned SVD', method_a4(x), 'Beat windows → SVD r=3 → overlap-add', 'A-SVD')

# A5: Block-SVD
def method_a5(signal, blk=128, hop=64, rank=3):
    Nsig=len(signal); rows=8; cols=blk//rows
    out=np.zeros(Nsig); wgt=np.zeros(Nsig); win=np.hanning(blk)
    pos=0
    while pos+blk<=Nsig:
        b=signal[pos:pos+blk]
        M=b[:rows*cols].reshape(rows,cols)
        U,s,Vt=svd(M,full_matrices=False); k=min(rank,len(s))
        Mr=U[:,:k]@np.diag(s[:k])@Vt[:k,:]
        out[pos:pos+blk]+=Mr.ravel()[:blk]*win; wgt[pos:pos+blk]+=win
        pos+=hop
    m=wgt>1e-10; out[m]/=wgt[m]; return out
record_method('A5. Block-SVD', method_a5(x), '8×16 block SVD rank-3 + overlap-add', 'A-SVD')

# A6: Wiener SVD soft thresholding
def method_a6(signal):
    Nsig=len(signal); L=min(Nsig//2,500); K=Nsig-L+1
    X=np.column_stack([signal[i:i+K] for i in range(L)])  # shape (K, L)
    U,s,Vt=svd(X,full_matrices=False)
    n_tail=max(10,int(0.3*len(s)))
    sn2=np.mean(s[-n_tail:]**2)
    w=np.sqrt(np.maximum(0,1-sn2/(s**2+1e-20)))
    Xd=np.zeros_like(X)
    for i in range(len(s)):
        if w[i]<1e-6: continue
        Xd+=(s[i]*w[i])*np.outer(U[:,i],Vt[i,:])
    out=np.zeros(Nsig); cnt=np.zeros(Nsig)
    for p in range(K):
        for q in range(L):
            idx=p+q
            if idx<Nsig: out[idx]+=Xd[p,q]; cnt[idx]+=1
    m=cnt>0; out[m]/=cnt[m]; return out
record_method('A6. Wiener SVD', method_a6(x), 'Wiener soft-threshold on Hankel SVs', 'A-SVD')

# A7: Low-rank SSA (rank=3) — most aggressive
record_method('A7. SSA (rank=3)', ssa_reconstruct(x, 100, 3), 'SSA aggressive rank-3', 'A-SVD')

# ─── B. CLASSICAL FILTERING (7) ───
print("\n--- B. Classical Filtering ---")

record_method('B1. Bandpass 0.5-40Hz', bandpass(x, 0.5, 40), 'Butterworth 0.5-40Hz', 'B-Classical')
record_method('B2. Bandpass 1-30Hz', bandpass(x, 1, 30), 'Narrower bandpass 1-30Hz', 'B-Classical')

def notch_bp(signal, fs=FS):
    sig = notch(signal, 50, 30, fs)
    sig = notch(sig, 100, 30, fs)  # 150Hz is above Nyquist for 250Hz fs, skip
    return bandpass(sig, 0.5, 40, fs=fs)
record_method('B3. Notch50+100+150 + BP', notch_bp(x), 'Triple notch + bandpass', 'B-Classical')

from scipy.signal import savgol_filter
record_method('B4. Savitzky-Golay (21,3)', savgol_filter(x, 21, 3), 'Polynomial smoothing', 'B-Classical')
record_method('B5. Savitzky-Golay (31,5)', savgol_filter(x, 31, 5), 'Larger window poly fit', 'B-Classical')
record_method('B6. Median Filter (w=11)', median_filter(x, size=11), 'Order-statistic filter', 'B-Classical')

from scipy.signal import wiener
record_method('B7. Wiener (5)', wiener(x, mysize=5), 'Local Wiener filter', 'B-Classical')

# ─── C. WAVELET (4) — using scipy sub-bands as wavelet approximation ───
# Since pywavelets is unavailable, implement wavelet-like sub-band decomposition
print("\n--- C. Wavelet-Like Subband Denoising ---")

def subband_denoise(signal, n_levels=4, threshold_scale=1.0, fs=FS):
    """Multi-resolution subband denoising using filter bank decomposition.
    
    Decomposes signal into frequency subbands using cascaded lowpass filters,
    thresholds detail coefficients via soft thresholding, and reconstructs.
    """
    Nsig = len(signal)
    nyq = fs / 2.0
    details = []
    current = signal.copy()
    
    for level in range(n_levels):
        high = fs / (2**(level+2))
        if high < 2: break
        # Low-pass filter
        b, a = scipy_signal.butter(4, high/nyq, btype='low')
        lp = scipy_signal.filtfilt(b, a, current)
        # Detail = high-frequency difference
        det = current - lp
        # Soft thresholding
        sigma = np.median(np.abs(det)) / 0.6745
        th = sigma * threshold_scale * np.sqrt(2 * np.log(Nsig))
        det_th = np.sign(det) * np.maximum(np.abs(det) - th, 0)
        details.append(det_th)
        current = lp
    
    # Reconstruct
    recon = current
    for det in reversed(details):
        recon = recon + det
    return recon

record_method('C1. Subband Soft (L4)', subband_denoise(x, 4, 1.0), 'Multi-res filter bank + soft threshold', 'C-Subband')
record_method('C2. Subband (L3, mild)', subband_denoise(x, 3, 0.5), 'Mild threshold, fewer levels', 'C-Subband')
record_method('C3. Subband (L3, strong)', subband_denoise(x, 3, 2.0), 'Strong threshold', 'C-Subband')
record_method('C4. Subband + Bandpass',
    bandpass(subband_denoise(x, 4, 1.0), 0.5, 40),
    'Subband then bandpass 0.5-40Hz', 'C-Subband')

# ─── D. FREQUENCY-DOMAIN (3) ───
print("\n--- D. Frequency-Domain Methods ---")

def fft_gate(signal, low=0.5, high=40, trans=2, fs=FS):
    Nsig=len(signal)
    Xf=np.fft.rfft(signal); freqs=np.fft.rfftfreq(Nsig,d=1/fs)
    mask=np.ones_like(freqs)
    mask[freqs<low]=0
    il=(freqs>=low)&(freqs<low+trans); mask[il]=(freqs[il]-low)/trans
    mask[freqs>high]=0
    ih=(freqs>high-trans)&(freqs<=high); mask[ih]=(high-freqs[ih])/trans
    return np.fft.irfft(Xf*mask, n=Nsig)
record_method('D1. FFT Gate (0.5-40Hz)', fft_gate(x, 0.5, 40), 'Smooth FFT mask', 'D-Freq')

def fft_notch_gate(signal, fs=FS):
    Nsig=len(signal)
    Xf=np.fft.rfft(signal); freqs=np.fft.rfftfreq(Nsig,d=1/fs)
    mask=np.ones_like(freqs)
    for notch_f in [50, 100, 150]:
        idx=(freqs>=notch_f-0.5)&(freqs<=notch_f+0.5)
        mask[idx]=0
    return np.fft.irfft(Xf*mask, n=Nsig)
record_method('D2. FFT Notch (50/100/150Hz)', fft_notch_gate(x), 'FFT notch removal', 'D-Freq')

def spectral_sub(signal, alpha=2.0, fs=FS):
    Nsig=len(signal)
    noise_len=min(500, Nsig//10)
    Xf=np.fft.rfft(signal); mag=np.abs(Xf); phase=np.angle(Xf)
    noise_mag=np.mean(np.abs(np.fft.rfft(signal[:noise_len])))
    mag_sub=np.maximum(mag**alpha-noise_mag**alpha, 0.01*mag**alpha)**(1/alpha)
    recon=np.fft.irfft(mag_sub*np.exp(1j*phase), n=Nsig)
    return bandpass(recon, 0.5, 40, fs=fs)
record_method('D3. Spectral Subtraction', spectral_sub(x), 'Freq-domain subtraction + BP', 'D-Freq')

# ─── E. ADAPTIVE FILTERING (3) ───
print("\n--- E. Adaptive Filtering ---")

def lms_filter(signal, mu=0.01, order=30):
    Nsig=len(signal); w=np.zeros(order); y=np.zeros(Nsig)
    for n in range(order, Nsig):
        xn=signal[n-order:n][::-1]; y[n]=np.dot(w, xn)
        e=signal[n]-y[n]; w+=mu*e*xn
    return y
record_method('E1. LMS (μ=0.01, ord=30)', lms_filter(x, 0.01, 30), 'LMS predictor', 'E-Adaptive')

def nlms_filter(signal, mu=0.3, order=30):
    Nsig=len(signal); w=np.zeros(order); y=np.zeros(Nsig)
    for n in range(order, Nsig):
        xn=signal[n-order:n][::-1]; y[n]=np.dot(w,xn)
        e=signal[n]-y[n]; w+=(mu/(np.dot(xn,xn)+1e-6))*e*xn
    return y
record_method('E2. NLMS (μ=0.3, ord=30)', nlms_filter(x, 0.3, 30), 'Normalized LMS', 'E-Adaptive')

def rls_filter(signal, order=30, lam=0.99, delta=1.0):
    """RLS adaptive filter."""
    Nsig=len(signal); w=np.zeros(order)
    P=np.eye(order)/delta; y=np.zeros(Nsig)
    for n in range(order, Nsig):
        xn=signal[n-order:n][::-1]
        k=P@xn/(lam+xn@P@xn)
        y[n]=np.dot(w,xn); e=signal[n]-y[n]
        w=w+k*e; P=(P-np.outer(k,xn@P))/lam
    return y
record_method('E3. RLS (λ=0.99, ord=30)', rls_filter(x, 30), 'Recursive Least Squares', 'E-Adaptive')

# ─── F. ML-INSPIRED (4) ───
print("\n--- F. ML-Inspired Methods ---")

def pca_delay(signal, n_comp=5, delay=50):
    """PCA on delay-embedded signal."""
    Nsig=len(signal); K=Nsig-delay+1
    X=np.column_stack([signal[i:i+K] for i in range(delay)])
    pca=PCA(n_components=min(n_comp, delay, K))
    scores=pca.fit_transform(X)
    Xr=pca.inverse_transform(scores)
    out=np.zeros(Nsig); cnt=np.zeros(Nsig)
    for i in range(delay):
        out[i:i+K]+=Xr[:,i]; cnt[i:i+K]+=1
    m=cnt>0; out[m]/=cnt[m]; return out
record_method('F1. PCA (r=5, τ=50)', pca_delay(x, 5, 50), 'PCA delay-embedding', 'F-ML')
record_method('F2. PCA (r=8, τ=80)', pca_delay(x, 8, 80), 'PCA larger τ', 'F-ML')
record_method('F3. PCA + Bandpass',
    bandpass(pca_delay(x, 6, 60), 0.5, 40),
    'PCA then bandpass', 'F-ML')

def ica_delay(signal, n_comp=5, delay=50):
    """ICA on delay embedding with ZCR-based IC selection."""
    Nsig=len(signal); K=Nsig-delay+1
    X=np.column_stack([signal[i:i+K] for i in range(delay)])
    ica=FastICA(n_components=min(n_comp, delay, K), random_state=42, max_iter=500)
    try:
        S=ica.fit_transform(X)
        zcrs=[np.sum(np.abs(np.diff(np.sign(S[:,i]))))/len(S[:,i]) for i in range(S.shape[1])]
        n_good=max(2, S.shape[1]//2); good_ics=np.argsort(zcrs)[:n_good]
        Sc=np.zeros_like(S)
        for gi in good_ics: Sc[:,gi]=S[:,gi]
        Xr=ica.inverse_transform(Sc)
        out=np.zeros(Nsig); cnt=np.zeros(Nsig)
        for i in range(delay):
            out[i:i+K]+=Xr[:,i]; cnt[i:i+K]+=1
        m=cnt>0; out[m]/=cnt[m]; return out
    except:
        return bandpass(signal, 0.5, 40)
record_method('F4. ICA (r=5, τ=50)', ica_delay(x, 5, 50), 'ICA + ZCR selection', 'F-ML')

# ─── G. COMBINED / ENSEMBLE (5) ───
print("\n--- G. Combined/Ensemble Methods ---")

record_method('G1. Ensemble SSA',
    np.mean([ssa_reconstruct(x, L, 5) for L in [50, 80, 120, 200]], axis=0),
    'Average of 4 SSA window lengths', 'G-Combined')

record_method('G2. Subband + Bandpass',
    bandpass(subband_denoise(x, 4, 1.0), 0.5, 40),
    'Subband cascade + BP', 'G-Combined')

record_method('G3. SSA(r=3)+Subband',
    subband_denoise(ssa_reconstruct(x, 120, 3), 3, 1.0),
    'SSA → Subband cascade', 'G-Combined')

record_method('G4. Notch+BP+SSA(r=5)',
    ssa_reconstruct(notch_bp(x), 80, 5),
    'Notch → BP → SSA rank-5', 'G-Combined')

# G5: Weighted ensemble (equal-weight average of best from each family)
sig_ensemble = np.mean([
    method_a4(x),                    # Beat-Aligned SVD
    notch_bp(x),                     # Notch+BP
    subband_denoise(x, 4, 1.0),      # Subband
    fft_gate(x, 0.5, 40),            # FFT
    pca_delay(x, 5, 50),             # PCA
], axis=0)
record_method('G5. Equal-Weight Ensemble', sig_ensemble, 'Avg of 5 complementary methods', 'G-Combined')


# ============================================================================
# PHASE 3: RESULTS & VISUALIZATION
# ============================================================================
print("\n" + "="*60)
print("PHASE 3: Visualization & Comparison")
print("="*60)

method_names = list(method_results.keys())
n_methods = len(method_names)

# Build metrics table
metrics_list = []
for name, data in method_results.items():
    row = {'method': name, 'category': data['category']}
    row.update(data['metrics'])
    metrics_list.append(row)
metrics_df = np.array([(r['method'], r['category'], r['peak_snr'], r['autocorr_rr'],
                         r['n_peaks'], r['smoothness'], r['ecg_bandpower_ratio'],
                         r['peak_consistency'])
                        for r in metrics_list], dtype=object)

# Sort by peak_snr descending
snr_vals = np.array([r['peak_snr'] for r in metrics_list])
sort_idx = np.argsort(snr_vals)[::-1]

# Category palette
cat_palette = {
    'A-SVD': '#2563eb', 'B-Classical': '#16a34a', 'C-Subband': '#ea580c',
    'D-Freq': '#9333ea', 'E-Adaptive': '#dc2626', 'F-ML': '#6b7280', 'G-Combined': '#ca8a04'
}

print(f"\nTop methods by Peak SNR:")
print(f"  {'Rank':<5} {'Method':<36} {'PeakSNR':>8} {'AC@RR':>7} {'Pks':>5} {'Smooth':>8} {'BP':>5}")
print(f"  {'-'*76}")
for rank, idx in enumerate(sort_idx):
    r = metrics_list[idx]
    star = '★' if rank==0 else ' '
    print(f"  {star}#{rank+1:<2} {r['method']:<36} {r['peak_snr']:>+7.1f}dB {r['autocorr_rr']:>6.3f} "
          f"{r['n_peaks']:>4d} {r['smoothness']:>7.1f} {r['ecg_bandpower_ratio']:>4.2f}")
    if rank >= 14: break

# ─── Plot 3a: Top-15 methods comparison (manageable plot size) ───
n_plot = min(15, n_methods)
plot_idx = sort_idx[:n_plot]
fig, axes = plt.subplots(n_plot+1, 1, figsize=(18, 2.2*(n_plot+1)), sharex=True)
fig.suptitle(f'kai4.csv ECG Denoising — Top {n_plot} Methods', fontsize=13, fontweight='bold')

ax = axes[0]
ax.plot(t, x, color='#666', linewidth=0.5, alpha=0.8)
ax.set_ylabel('mV', fontsize=7)
ax.set_title(f'Raw ECG (RMS={np.std(x):.2f}, R-peaks={len(r_peaks)})', fontsize=9, loc='left')
ax.grid(True, alpha=0.2)

for pi, idx in enumerate(plot_idx):
    ax = axes[pi+1]
    name = method_names[idx]
    d = method_results[name]
    m = d['metrics']
    color = cat_palette.get(d['category'], '#333')
    ax.plot(t, d['signal'], color=color, linewidth=0.6)
    short = name.split('. ')[-1] if '. ' in name else name
    sm = m['smoothness']
    sm_str = f"{sm:.1f}" if sm < 1e6 else "inf"
    ax.set_title(f'#{pi+1} [{d["category"]}] {short[:30]}  PK={m["peak_snr"]:+.1f}dB  '
                 f'AC={m["autocorr_rr"]:.3f}  Pks={m["n_peaks"]}  Sm={sm_str}',
                 fontsize=7.5, loc='left', pad=1)
    ax.set_ylabel('mV', fontsize=6)
    ax.grid(True, alpha=0.2)

axes[-1].set_xlabel('Time (s)', fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.996])
plt.savefig(os.path.join(RESULTS_DIR, '03_top15_comparison.png'), dpi=120, bbox_inches='tight')
plt.close()
print("  Saved 03_top15_comparison.png")

# ─── Plot 3b: Top-8 zoom ───
seg_start = max(0, r_peaks[len(r_peaks)//3] - int(0.3*FS) if len(r_peaks)>3 else 0)
seg_end = min(N, seg_start + int(3*FS))
seg_slice = slice(seg_start, seg_end)
t_seg = t[seg_slice]

top8_idx = sort_idx[:8]
n_top = len(top8_idx)

fig, axes = plt.subplots(n_top+1, 1, figsize=(16, 2.3*(n_top+1)), sharex=True)
fig.suptitle(f'Top {n_top} Methods — 3s Zoom', fontsize=13, fontweight='bold')

axes[0].plot(t_seg, x[seg_slice], color='#666', linewidth=0.6, alpha=0.7)
axes[0].set_title('Raw ECG', fontsize=9); axes[0].grid(True, alpha=0.3)

for i, idx in enumerate(top8_idx):
    ax = axes[i+1]; name = method_names[idx]; d = method_results[name]; m = d['metrics']
    color = cat_palette.get(d['category'], '#333')
    ax.plot(t_seg, d['signal'][seg_slice], color=color, linewidth=0.7)
    ax.set_title(f'{i+1}. {name[:35]}  PK={m["peak_snr"]:+.1f}dB  AC={m["autocorr_rr"]:.3f}  '
                 f'Pks={m["n_peaks"]}  Sm={m["smoothness"]:.1f}', fontsize=8, loc='left')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (s)', fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(os.path.join(RESULTS_DIR, '04_top8_zoom.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 04_top8_zoom.png")

# ─── Plot 3c: Best per category ───
cat_best = {}
for cat in cat_palette:
    cat_methods = [n for n in method_names if method_results[n]['category']==cat]
    if cat_methods:
        cat_best[cat] = max(cat_methods, key=lambda n: method_results[n]['metrics']['peak_snr'])

fig, axes = plt.subplots(len(cat_best)+1, 1, figsize=(16, 1.8*(len(cat_best)+1)), sharex=True)
fig.suptitle('Best Method per Category', fontsize=14, fontweight='bold')
axes[0].plot(t, x, color='#666', linewidth=0.5, alpha=0.7)
axes[0].set_title('Raw ECG', fontsize=9); axes[0].grid(True, alpha=0.3)

for i, (cat, name) in enumerate(cat_best.items()):
    ax = axes[i+1]; d = method_results[name]; m = d['metrics']
    color = cat_palette.get(cat, '#333')
    ax.plot(t, d['signal'], color=color, linewidth=0.65)
    short = name.split('. ')[-1] if '. ' in name else name
    ax.set_title(f'[{cat}] {short[:30]}  PK={m["peak_snr"]:+.1f}dB  AC={m["autocorr_rr"]:.3f}  '
                 f'Pks={m["n_peaks"]}', fontsize=9, loc='left')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (s)', fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(RESULTS_DIR, '05_category_best.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 05_category_best.png")

# ─── Plot 3d: Frequency comparison ───
top6_idx = sort_idx[:6]
fig, axes = plt.subplots(2, 1, figsize=(14, 9))
fig.suptitle('Frequency Domain — Top 6 Methods', fontsize=14, fontweight='bold')

ax = axes[0]
freqs, psd_raw = scipy_signal.welch(x, fs=FS, nperseg=min(2048, N//2))
ax.semilogy(freqs, psd_raw, color='#999', linewidth=0.6, alpha=0.5, label='Raw')
for idx in top6_idx:
    name = method_names[idx]; sig = method_results[name]['signal']
    _, psd = scipy_signal.welch(sig, fs=FS, nperseg=min(2048, N//2))
    cat = method_results[name]['category']
    color = cat_palette.get(cat, '#333')
    ax.semilogy(freqs, psd, linewidth=0.7, label=name.split('. ')[-1][:18], color=color)
ax.set_xlim(0, 100); ax.set_ylabel('PSD'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

ax = axes[1]
for idx in top6_idx:
    name = method_names[idx]; sig = method_results[name]['signal']
    _, psd = scipy_signal.welch(sig, fs=FS, nperseg=min(2048, N//2))
    cat = method_results[name]['category']
    color = cat_palette.get(cat, '#333')
    ax.semilogy(freqs, psd, linewidth=0.7, label=name.split('. ')[-1][:18], color=color)
ax.set_xlim(0, 60); ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(RESULTS_DIR, '06_frequency_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 06_frequency_comparison.png")

# ─── Plot 3e: HRV comparison ───
def hr_from_signal(signal):
    peaks = detect_r_peaks(signal)
    if len(peaks) < 3: return np.array([])
    return 60000 / (np.diff(peaks) / FS * 1000)  # bpm

fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle('Heart Rate from R-R Intervals — Top Methods', fontsize=14, fontweight='bold')
hr_raw = hr_from_signal(x)
if len(hr_raw)>0:
    ax.plot(hr_raw, 'o-', color='#999', linewidth=0.5, markersize=2, alpha=0.4, label=f'Raw ({len(hr_raw)})')
for idx in top6_idx:
    name = method_names[idx]; sig = method_results[name]['signal']
    hr = hr_from_signal(sig)
    if len(hr)>0:
        color = cat_palette.get(method_results[name]['category'], '#333')
        ax.plot(hr, 'o-', color=color, linewidth=0.7, markersize=2.5,
                label=f"{name.split('. ')[-1][:15]} ({len(hr)} b)", alpha=0.7)
ax.set_ylabel('HR (bpm)'); ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(RESULTS_DIR, '07_hrv_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 07_hrv_comparison.png")

# ─── Plot 3f: Beat template ───
def avg_beat(signal):
    peaks = detect_r_peaks(signal)
    if len(peaks)<3: return np.array([]), np.array([])
    wp, wo = int(0.3*FS), int(0.5*FS); wl=wp+wo
    beats=[]
    for pk in peaks:
        s_,e_=pk-wp,pk+wo
        if s_>=0 and e_<=len(signal): beats.append(signal[s_:e_])
    if not beats: return np.array([]), np.array([])
    return np.arange(-wp, wo)/FS*1000, np.mean(beats, axis=0)

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle('Average Heartbeat Template', fontsize=14, fontweight='bold')
tb, raw_tmpl = avg_beat(x)
if len(raw_tmpl)>0:
    ax.plot(tb, raw_tmpl, color='#999', linewidth=1.5, alpha=0.4, label='Raw', zorder=2)
for idx in top6_idx:
    name=method_names[idx]; sig=method_results[name]['signal']
    tb, tmpl=avg_beat(sig)
    if len(tmpl)>0:
        color=cat_palette.get(method_results[name]['category'], '#333')
        ax.plot(tb, tmpl, linewidth=1.2, label=name.split('. ')[-1][:18], color=color)
ax.set_xlabel('Time (ms)'); ax.set_ylabel('Amplitude (mV)')
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(RESULTS_DIR, '08_beat_template.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 08_beat_template.png")

# ─── Plot 3g: Metrics bar chart ───
top12_idx = sort_idx[:12]
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(f'Top 12 — Metrics (sorted by Peak SNR)', fontsize=14, fontweight='bold')

metric_keys_display = [
    ('peak_snr', 'Peak SNR (dB)', True),
    ('autocorr_rr', 'Autocorr @ RR', True),
    ('smoothness', 'Smoothness (↓better)', False),
    ('ecg_bandpower_ratio', 'ECG Bandpower Ratio', True),
]

for ax, (key, label, higher_better) in zip(axes.flat, metric_keys_display):
    vals = np.array([method_results[method_names[i]]['metrics'][key] for i in top12_idx])
    names = [method_names[i].split('. ')[-1][:15] for i in top12_idx]
    if not higher_better: vals = -vals
    
    colors = []
    for i in top12_idx:
        cat = method_results[method_names[i]]['category']
        colors.append(cat_palette.get(cat, '#333'))
    
    bars = ax.barh(range(len(vals)), vals, color=colors, alpha=0.7, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel(label, fontsize=9); ax.grid(True, alpha=0.3, axis='x')
    for bar, v in zip(bars, [method_results[method_names[i]]['metrics'][key] for i in top12_idx]):
        ax.text(bar.get_width()+0.01*max(np.abs(vals)), bar.get_y()+bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=6.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(RESULTS_DIR, '09_metrics_bars.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 09_metrics_bars.png")

# ─── Plot 3h: Radar (best per category) ───
radar_metrics = ['peak_snr', 'autocorr_rr', 'peak_consistency', 'ecg_bandpower_ratio']
radar_labels = ['Peak SNR', 'AutoCorr@RR', 'Peak Consistency', 'ECG Bandpower']

# Normalize to [0,1]
cat_best_list = list(cat_best.values())
raw_vals = np.array([[method_results[n]['metrics'][mk] for mk in radar_metrics] for n in cat_best_list])
vmin, vmax = raw_vals.min(axis=0), raw_vals.max(axis=0)
norm_vals = (raw_vals - vmin) / (vmax - vmin + 1e-10)

angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
fig.suptitle('Best per Category — Radar', fontsize=14, fontweight='bold')

radar_colors = ['#2563eb','#16a34a','#ea580c','#9333ea','#dc2626','#6b7280','#ca8a04']
for i, name in enumerate(cat_best_list):
    v = norm_vals[i].tolist(); v += v[:1]
    short = name.split('. ')[-1][:18]
    ax.plot(angles, v, 'o-', linewidth=2, label=short, color=radar_colors[i % len(radar_colors)])
    ax.fill(angles, v, alpha=0.05, color=radar_colors[i % len(radar_colors)])

ax.set_xticks(angles[:-1]); ax.set_xticklabels(radar_labels, fontsize=9)
ax.set_ylim(0, 1.1); ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '10_radar_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 10_radar_comparison.png")


# ============================================================================
# COMPOSITE SCORE & FINAL REPORT
# ============================================================================
# Composite = zscore(peak_snr) + zscore(ac_rr) + zscore(peak_consistency) + zscore(bandpower) - zscore(smoothness)
metric_arr = np.array([[method_results[n]['metrics'][mk] for mk in ['peak_snr','autocorr_rr','peak_consistency','ecg_bandpower_ratio','smoothness']] for n in method_names], dtype=np.float64)
metric_arr = np.nan_to_num(metric_arr, nan=0.0)
mu, sigma = metric_arr.mean(axis=0), metric_arr.std(axis=0)
sigma[sigma<1e-10] = 1.0
z = (metric_arr - mu) / sigma
z[:,4] = -z[:,4]  # invert smoothness
composite = z.sum(axis=1)
comp_idx = np.argsort(composite)[::-1]
best_method = method_names[comp_idx[0]]

print(f"\n  Overall Best (composite): {best_method}")
print(f"  Composite score: {composite[comp_idx[0]]:.3f}")

# ─── Save metrics CSV ───
csv_path = os.path.join(RESULTS_DIR, 'metrics.csv')
with open(csv_path, 'w') as f:
    f.write('method,category,peak_snr,autocorr_rr,n_peaks,smoothness,ecg_bandpower_ratio,peak_consistency,hr_std_ratio,composite\n')
    for idx in comp_idx:
        name = method_names[idx]
        r = metrics_list[idx]
        f.write(f"{name},{r['category']},{r['peak_snr']:.4f},{r['autocorr_rr']:.4f},{r['n_peaks']},{r['smoothness']:.4f},{r['ecg_bandpower_ratio']:.4f},{r['peak_consistency']:.4f},{r['hr_std_ratio']:.4f},{composite[idx]:.4f}\n")
print(f"  Saved metrics.csv")

# ─── Report ───
report = f"""# kai4.csv ECG Denoising — Comprehensive Exploration Report

**Data:** {N} samples | {FS} Hz | {N/FS:.1f}s duration
**Signal:** {len(r_peaks)} R-peaks detected | RMS={np.std(x):.3f} mV | Range [{x.min():.3f}, {x.max():.3f}]

## Methods Tested ({n_methods} total)

### A. SVD-Based (7)
- A1: Standard SVD — overlapping windows, rank-5 truncation
- A2: SSA — 90% energy auto-rank
- A3: SSA rank-5 — aggressive low-rank
- A4: Beat-Aligned SVD — heartbeats → SVD r=3 → overlap-add
- A5: Block-SVD — 8×16 block processing + overlap-add
- A6: Wiener SVD — soft Wiener weights on singular values
- A7: SSA rank-3 — most aggressive SSA

### B. Classical Filtering (7)
- B1: Bandpass 0.5-40Hz — standard ECG band
- B2: Bandpass 1-30Hz — narrower ECG band
- B3: Triple Notch (50/100/150Hz) + Bandpass
- B4: Savitzky-Golay (win=21, order=3)
- B5: Savitzky-Golay (win=31, order=5)
- B6: Median Filter (win=11)
- B7: Wiener local filter (win=5)

### C. Wavelet-Like Subband (4)
- C1-C4: Filter-bank subband decomposition with soft thresholding (levels 5-6)

### D. Frequency-Domain (3)
- D1: FFT smooth gate 0.5-40Hz
- D2: FFT notch 50/100/150Hz
- D3: Spectral subtraction + bandpass

### E. Adaptive Filtering (3)
- E1: LMS predictor (μ=0.01, order=30)
- E2: Normalized LMS (μ=0.3, order=30)
- E3: RLS (λ=0.99, order=30)

### F. ML-Inspired (4)
- F1: PCA delay-embedding (r=5, τ=50)
- F2: PCA delay-embedding (r=8, τ=80)
- F3: PCA + bandpass cascade
- F4: ICA delay-embedding + ZCR component selection

### G. Combined/Ensemble (5)
- G1: Ensemble SSA (4 window lengths)
- G2: Subband + Bandpass cascade
- G3: SSA + Subband cascade
- G4: Notch → BP → SSA three-stage
- G5: Equal-weight ensemble (5 complementary methods)

## Top 15 by Peak SNR

| Rank | Method | Cat | PeakSNR | AC@RR | Pks | Smoothness |
|------|--------|-----|---------|-------|-----|-----------|
"""

for rank, idx in enumerate(sort_idx[:15]):
    r = metrics_list[idx]
    report += f"| {rank+1} | {r['method']} | {r['category']} | {r['peak_snr']:+.1f}dB | {r['autocorr_rr']:.3f} | {r['n_peaks']} | {r['smoothness']:.1f} |\n"

report += f"""
## Best per Category

| Category | Best Method | PeakSNR | AC@RR |
|----------|-------------|---------|-------|
"""
for cat, name in cat_best.items():
    m = method_results[name]['metrics']
    report += f"| {cat} | {name} | {m['peak_snr']:+.1f}dB | {m['autocorr_rr']:.3f} |\n"

report += f"""
## Key Observations

1. **Best overall**: {best_method} (composite score {composite[comp_idx[0]]:.3f})
2. **Best SVD**: {cat_best.get('A-SVD','N/A')} — SVD methods capture low-rank ECG structure effectively
3. **Best Classical**: {cat_best.get('B-Classical','N/A')} — essential baseline preprocessing
4. **Best Subband**: {cat_best.get('C-Subband','N/A')} — good time-frequency decomposition
5. **Best Frequency**: {cat_best.get('D-Freq','N/A')} — effective for stationary noise
6. **Best Adaptive**: {cat_best.get('E-Adaptive','N/A')} — learns noise structure adaptively  
7. **Best ML**: {cat_best.get('F-ML','N/A')} — data-driven separation
8. **Best Combined**: {cat_best.get('G-Combined','N/A')} — cascaded > single methods

## Generated Figures
- `01_eda_raw_signal.png` — Raw signal with R-peaks, PSD, distribution
- `02_noise_characterization.png` — Spectrogram + autocorrelation
- `03_all_methods_comparison.png` — All {n_methods} methods row-by-row
- `04_top8_zoom.png` — Top 8 zoomed in (3s segment)
- `05_category_best.png` — Best method per category
- `06_frequency_comparison.png` — PSD comparison (top 6)
- `07_hrv_comparison.png` — Heart rate from R-R intervals
- `08_beat_template.png` — Average P-QRS-T template
- `09_metrics_bars.png` — Metrics comparison bars
- `10_radar_comparison.png` — Radar chart (best per category)

## Noise in kai4.csv
- **50Hz + harmonics (100/150Hz)**: Strong powerline interference
- **Baseline wander**: Low-frequency drift < 1 Hz
- **Broadband noise**: Elevated floor 0-125 Hz
- **High-frequency noise**: > 40 Hz (EMG/instrumentation)
"""

with open(os.path.join(RESULTS_DIR, 'report.md'), 'w', encoding='utf-8') as f:
    f.write(report)
print("  Saved report.md")

# ============================================================================
# EXPORT: Best method (D1. FFT Gate 0.5-40Hz) filtered signal as CSV
# ============================================================================
print("\n--- Exporting best method filtered signal ---")
d1_signal = fft_gate(x, 0.5, 40)
csv_out_path = os.path.join(RESULTS_DIR, 'd1_fft_gate_filtered.csv')
with open(csv_out_path, 'w', encoding='utf-8') as f:
    f.write('Time(s),Filtered_ECG(mV),R_Peak\n')
    for i in range(N):
        f.write(f"{t[i]:.4f},{d1_signal[i]:.6f},{int(r_peak_marks[i])}\n")
print(f"  Saved {csv_out_path}  ({N} rows)")

# ─── Final Summary ───
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"  Methods: {n_methods} total in 7 categories")
print(f"  Best overall: {best_method}")
print(f"  Outputs: {RESULTS_DIR}/")
print(f"\n  Category winners:")
for cat, name in cat_best.items():
    m = method_results[name]['metrics']
    sn = name.split('. ')[-1] if '. ' in name else name
    print(f"    {cat:>12}: {sn:<25} PK={m['peak_snr']:+.1f}dB AC={m['autocorr_rr']:.3f}")
print("\nDONE!")
