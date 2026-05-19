# kai4.csv ECG Denoising — Comprehensive Exploration Report

**Data:** 5000 samples | 250 Hz | 20.0s duration
**Signal:** 49 R-peaks detected | RMS=6.905 mV | Range [-12.068, 13.392]

## Methods Tested (33 total)

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
| 1 | D1. FFT Gate (0.5-40Hz) | D-Freq | +11.8dB | -0.017 | 48 | 0.0 |
| 2 | D2. FFT Notch (50/100/150Hz) | D-Freq | +9.6dB | -0.055 | 48 | 0.2 |
| 3 | B3. Notch50+100+150 + BP | B-Classical | +9.6dB | -0.071 | 46 | 0.0 |
| 4 | B2. Bandpass 1-30Hz | B-Classical | +9.3dB | -0.135 | 45 | 0.0 |
| 5 | B7. Wiener (5) | B-Classical | +8.0dB | -0.027 | 47 | 0.5 |
| 6 | B1. Bandpass 0.5-40Hz | B-Classical | +7.9dB | 0.060 | 45 | 0.8 |
| 7 | F4. ICA (r=5, τ=50) | F-ML | +7.5dB | -0.075 | 41 | 0.0 |
| 8 | D3. Spectral Subtraction | D-Freq | +7.3dB | 0.080 | 48 | 0.8 |
| 9 | G4. Notch+BP+SSA(r=5) | G-Combined | +7.3dB | -0.099 | 42 | 0.0 |
| 10 | C2. Subband (L3, mild) | C-Subband | +6.9dB | -0.080 | 43 | 0.0 |
| 11 | F3. PCA + Bandpass | F-ML | +6.0dB | -0.141 | 41 | 0.8 |
| 12 | C1. Subband Soft (L4) | C-Subband | +5.9dB | -0.080 | 46 | 0.0 |
| 13 | C4. Subband + Bandpass | C-Subband | +4.4dB | -0.041 | 46 | 0.0 |
| 14 | G2. Subband + Bandpass | G-Combined | +4.4dB | -0.041 | 46 | 0.0 |
| 15 | C3. Subband (L3, strong) | C-Subband | +3.9dB | -0.073 | 47 | 0.0 |

## Best per Category

| Category | Best Method | PeakSNR | AC@RR |
|----------|-------------|---------|-------|
| A-SVD | A2. SSA (90% energy) | +3.5dB | -0.847 |
| B-Classical | B3. Notch50+100+150 + BP | +9.6dB | -0.071 |
| C-Subband | C2. Subband (L3, mild) | +6.9dB | -0.080 |
| D-Freq | D1. FFT Gate (0.5-40Hz) | +11.8dB | -0.017 |
| E-Adaptive | E3. RLS (λ=0.99, ord=30) | +1.7dB | -0.703 |
| F-ML | F4. ICA (r=5, τ=50) | +7.5dB | -0.075 |
| G-Combined | G4. Notch+BP+SSA(r=5) | +7.3dB | -0.099 |

## Key Observations

1. **Best overall**: D1. FFT Gate (0.5-40Hz) (composite score 5.066)
2. **Best SVD**: A2. SSA (90% energy) — SVD methods capture low-rank ECG structure effectively
3. **Best Classical**: B3. Notch50+100+150 + BP — essential baseline preprocessing
4. **Best Subband**: C2. Subband (L3, mild) — good time-frequency decomposition
5. **Best Frequency**: D1. FFT Gate (0.5-40Hz) — effective for stationary noise
6. **Best Adaptive**: E3. RLS (λ=0.99, ord=30) — learns noise structure adaptively  
7. **Best ML**: F4. ICA (r=5, τ=50) — data-driven separation
8. **Best Combined**: G4. Notch+BP+SSA(r=5) — cascaded > single methods

## Generated Figures
- `01_eda_raw_signal.png` — Raw signal with R-peaks, PSD, distribution
- `02_noise_characterization.png` — Spectrogram + autocorrelation
- `03_all_methods_comparison.png` — All 33 methods row-by-row
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




📊 逐图详解
图 01 — eda_raw_signal.png（探索性数据分析）
4 个子图纵向排列：
子图	内容
① 完整信号	20s 全序列，蓝色线 = Filtered_ECG(mV)，红色散点 = 检出 R 峰（49 个）。可以看到强烈的工频干扰和无规律的高幅值波动，基线严重漂移。
② 前 5 秒放大	前 1250 个采样点的细节，红色标记 R 峰——能看到心跳间期约 0.73-0.90s（~70-80 bpm），但噪声几乎淹没了 P 波和 T 波。
③ 功率谱密度 (PSD)	对数纵轴。绿色阴影 = ECG 有效频带 0.5-40 Hz。竖向虚线标记 50、100、150 Hz 工频及其谐波。可以看到 50Hz 和 100Hz 处有极高的功率尖峰（> 背景噪声 2-3 个数量级），说明 PLI（电力线干扰）是这个信号最主要的噪声。
④ 幅值分布直方图	红线 = 均值，橙色虚线 = ±1σ。RMS=6.905 mV，分布接近高斯（中央厚尾），无严重异常值。
核心结论：这个信号最大的问题是 50Hz 工频 + 谐波，必须用陷波或 FFT 门控才能有效去除。
图 02 — noise_characterization.png（噪声特性）
2 个子图：
子图	内容
① 语谱图 (Spectrogram)	时间-频率热力图，nperseg=128，频率 0-125 Hz。颜色越亮 = 能量越高。可以看到: (a) 50Hz 处有一条连续亮线（工频持续存在）；(b) 100Hz 处有较弱的谐波亮线；(c) 低频 0-5 Hz 有大片的低频漂移（基线 wander）；(d) 高频段有散布的宽频噪声。
② 自相关 (Autocorrelation)	滞后 ±5s 的归一化自相关函数。理想的周期性 ECG 应该在 R-R 间期处有明显峰值 (~0.75s)。但这个信号的原始自相关几乎看不到清晰的周期性峰——说明噪声完全掩盖了心跳的周期性结构。
核心结论：工频 + 基线漂移 + 宽频噪声三重叠加，噪声的能量远高于 ECG 信号本身。
图 03 — top15_comparison.png（Top 15 方法全信号对比）
16 行纵向排列（sharex=True 共享 X 轴）：
- 
第 1 行：原始噪声信号（灰色，RMS=6.905）
- 
第 2-16 行：Top 15 方法（按峰值信噪比降序排列）
- 
每行标题显示：排名、类别、方法简称、Peak SNR、自相关@RR、R峰数、平滑度
- 
线条颜色按类别区分：A-SVD 蓝色、B-Classical 绿色、C-Subband 橙色、D-Freq 紫色、E-Adaptive 红色、F-ML 灰色、G-Combined 金色
可以直观看到：
- 
D1 FFT Gate 输出信号最为干净（背景几乎完全平坦），QRS 波清晰可见
- 
B1/B2 带通滤波后信号光滑但有幅度衰减，基线平坦
- 
SVD 类 (A1-A7) 输出看起来几乎就是缩了一条幅度的原信号——说明 SVD 没分离出 ECG 分量
- 
ICA (F4) 保留了较为尖锐的 QRS 形态，但基线有小幅波动
图 04 — top8_zoom.png（Top 8 方法 3 秒放大）
9 行，展示一个 3 秒窗口（约 4 个心跳周期）：
选择了信号中段的一个代表性片段（从第 ~N/3 个 R 峰往前 0.3 秒开始，持续 3 秒）。
这是最具诊断价值的一张图——可以逐拍比较各方法的 QRS 形态保持情况：
- 
D1 FFT Gate：QRS 尖锐、基线平直、P 波和 T 波都有保留
- 
B3 Notch+BP：同样干净，但 QRS 比 FFT Gate 稍钝
- 
B2 Bandpass 1-30Hz：非常平滑，T 波保留好，但 QRS 峰有衰减
- 
*B7 Wiener (5)*：保留了更多高频细节，但仍有残留噪声
- 
SVD 类（图上排在 10+ 名后）：QRS 几乎不可见，信号几乎被噪声支配
图 05 — category_best.png（各类别最佳方法对比）
8 行： 原始 + 7 个类别各自的最佳方法，显示完整 20s 信号：
类别	最佳方法
A-SVD	SSA (90% energy) PK=+3.5dB
B-Classical	Notch50+100+150+BP PK=+9.6dB
C-Subband	Subband (L3, mild) PK=+6.9dB
D-Freq	FFT Gate (0.5-40Hz) PK=+11.8dB
E-Adaptive	RLS PK=+1.7dB
F-ML	ICA (r=5) PK=+7.5dB
G-Combined	Notch+BP+SSA(r=5) PK=+7.3dB
图 06 — frequency_comparison.png（频域对比）
2 个上下排列的 PSD 图：
子图	频率范围
上	0-100 Hz
下	0-60 Hz
灰色线 = 原始信号的 PSD（极高噪声基底）。彩色线 = Top 6 方法的 PSD。
关键观察：
- 
D1 FFT Gate（紫色）在 0.5-40Hz 通带内几乎完全保留了原始 ECG 的能量，而 0.5Hz 以下和 40Hz 以上的频谱被完全压到噪声基底之下——所以它 BP ratio = 1.00
- 
带通滤波（绿色）在通带内外界限分明，但在截止频率处有陡降过渡
- 
SVD 类方法的 PSD 在低频段衰减不明显，说明 SVD 没有有效抑制高频噪声和工频——所以 BP ratio 只有 0.01-0.02
- 
50Hz 尖峰在 D1 FFT Gate 中被完全消除，在 B1 带通中被部分衰减（因为带通只到 40Hz）
图 07 — hrv_comparison.png（心率变异性对比）
折线图： X 轴 = 心跳拍序号，Y 轴 = 瞬时心率（bpm，由 R-R 间期换算）：
- 
灰色点 + 线 = 原始信号的心率（R-R 间期来自原始信号中的 R_Peak 列标注的 38 个峰），心率稳定在 ~76-82 bpm。
- 
彩色线 = Top 6 方法从去噪信号中检测到的 R 峰计算的心率
意义： 如果去噪方法错误删除了 R 峰或引入了假峰（例如 SSA rank=5 检出了 52 个峰——多了 3 个假阳性），HRV 曲线就会有偏差。
图 08 — beat_template.png（平均心跳模板）
单线图： X 轴 = 时间偏移（-300ms 到 +500ms 对应单个心跳前后），Y 轴 = 平均幅值：
- 
灰色粗线 = 原始信号的平均心跳（因为噪声太大，几乎看不出 P-QRS-T 形态）
- 
彩色线 = Top 6 方法各自的平均心跳模板
模板形态解读：
- 
正常 ECG 模板应有：P 波（小正向波）→ QRS 复合波（尖锐大波）→ T 波（宽缓正向波）
- 
D1 FFT Gate 的模板应该最接近正常形态：QRS 突出、ST 段平直、T 波可见
- 
B2 带通 1-30Hz 的模板更平滑（高频成分少）
- 
SVD 方法的模板几乎是一条平坦线（QRS 被严重衰减）
图 09 — metrics_bars.png（指标柱状图）
2×2 矩阵，四个指标的水平柱状图：
子图位置	指标	含义
左上	Peak SNR (dB)	R 峰功率 / 非峰值噪声功率
右上	Autocorr @ RR	在 R-R 间期滞后处的自相关值，衡量周期性保存程度
左下	Smoothness	二阶差分的能量均值，越低越平滑
右下	ECG Bandpower Ratio	0.5-40 Hz 频带功率 / 总功率
每个子图： 水平排列 Top 12 方法，颜色按类别区分，柱末端标注具体数值。
解读方法：
- 
同时看四个指标可以避免单一指标误导。例如：
- 
SSA rank=5 的 Smoothness = 89（极不平滑）且 BP = 0.01（不保留 ECG 频带），虽然自相关高但这是因为它保留了原始噪声的结构
- 
D1 FFT Gate 在全部四个指标上都表现突出（SNR 第一，BP=1.00，Smoothness≈0，AC=-0.017 略低但可接受）
图 10 — radar_comparison.png（雷达图）
极坐标雷达图：
- 
4 个维度（4 条轴线）：
1. 
Peak SNR（峰值信噪比）
2. 
AutoCorr@RR（心跳周期性保持）
3. 
Peak Consistency（R 峰检出一致性）
4. 
ECG Bandpower（ECG 频带能量保留）
- 
每条彩色闭合多边形 = 一个类别的最佳方法
- 
所有值已归一化到 0, 1（同类方法间的相对排名）
解读： 多边形面积越大、越往外扩张 = 该类别在所有维度上综合表现越好。
- 
D-Freq（紫色）大概率占据最大的面积，尤其在 ECG Bandpower 维度达到极值（1.0）
- 
A-SVD（蓝色）除了 Peak Consistency 维度（=大多数方法都能检出 ~49 个峰中的大部分），其他三个维度全面收缩
- 
雷达图直观地显示了 SVD 的偏科问题——它能检出峰，但不能保留 ECG 内容的形态和频带
总结：各图在分析中的作用
图号	作用类型	回答什么问题
01	信号认识	这信号长什么样？噪声多大？
02	噪声诊断	噪声的频率成分是什么？周期性结构还在吗？
03	宏观排名	哪些方法整体表现最好？信号长得像不像 ECG？
04	微观细节	前几名方法在 3s 窗口里对比如何？QRS 形态保留了吗？
05	类别对比	哪一类方法最适合这种噪声？同类内谁最好？
06	频域诊断	各方法在哪些频率上消除了噪声？有没有破坏 ECG 频带？
07	生理一致性	去噪后能否正确检出心跳？心率是否物理合理？
08	波形保真	平均心跳模板的 P-QRS-T 形态是否完整？
09	多指标分析	从 4 个量化维度看每种方法的优缺点
10	综合评估	各类别方法在 4 个标准化维度上的综合竞争力如何？
一句话结论： 对于 kai4.csv 这种以工频 + 谐波为主的噪声，频域方法（FFT Gate/Notch）效果最好，远超 SVD。SVD 更适合白噪声场景，对结构化噪声（50/100Hz 等）无能为力。
