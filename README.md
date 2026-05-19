# ECG Denoising Lab

实时心电(ECG)信号去噪与可视化系统。从串口采集 STM32 心电数据，通过多种算法去噪并在浏览器中实时显示波形。

## 功能

- **串口采集** — 读取 STM32F1 的 13 字节二进制 ECG 数据包
- **8 种去噪算法** — FFT 门控、陷波+带通、子带/小波、SSA、PCA、ICA、LMS、等权集成
- **自动选优** — 每 10 秒窗口运行全部方法，自动选择综合评分最优结果
- **实时可视化** — 通过 SSE + Canvas 在浏览器波形显示
- **鼠标交互** — 滚轮垂直缩放、Shift+滚轮水平缩放、双击重置
- **R 峰检测** — 使用 scipy.signal.find_peaks + 自适应阈值

## 快速开始

### 安装依赖

```bash
pip install flask numpy scipy scikit-learn pyserial
```

### 模拟模式（无需硬件）

```bash
python main.py
# 浏览器打开 http://localhost:8080
```

### 串口模式（接 STM32 硬件）

```bash
python main.py --port COM37 --baud 115200
```

## 架构

```
串口 → read_packet()
        ↓
  环形缓冲区 (10s)
        ↓ (每 5 秒触发)
  BatchProcessor (后台线程)
  ├─ FFT Gate (0.5-40Hz)
  ├─ Notch + Bandpass
  ├─ Subband Denoising
  ├─ SSA Low-Rank
  ├─ PCA Delay-Embedding
  ├─ ICA + ZCR Selection
  ├─ LMS Adaptive
  └─ Ensemble → 选最优
        ↓
  SSE → 浏览器 Canvas
```

## 操作

| 操作 | 效果 |
|------|------|
| 滚轮 | 垂直缩放（幅值） |
| Shift + 滚轮 | 水平缩放（时基） |
| 双击画布 | 重置缩放 |

## 版本

- `main.py` — 当前 V2（批处理 + 流式播放）
- `.versions/main_v1.py` — V1 备份（实时 FFT Gate 滑窗）

## 技术栈

- Python + NumPy/SciPy + Scikit-learn（信号处理）
- Flask + Server-Sent Events（后端推送）
- HTML5 Canvas（前端渲染）
- pyserial（串口通信）
