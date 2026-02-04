import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
import pywt
from PyEMD import EMD  # !pip install EMD-signal
import vmdpy  # 对于 VMD
import ewtpy  # 对于 EWT
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # 过滤pywt边界警告

# Shocklet Transform 自定义实现 (基于 GitHub compstorylab/discrete-shocklet-transform 的简化版)
def shocklet_transform(ts, tau=1, alpha=1.0):
    # 简单扩散方程模拟 shocklet 分解
    diff = np.diff(ts)
    shocklets = np.cumsum(np.exp(-alpha * np.arange(len(diff))) * diff)
    if len(shocklets) < len(ts):
        shocklets = np.pad(shocklets, (0, 1), mode='constant')
    return shocklets  # 返回 shocklet 序列作为模态

# Hurst Exponent 计算函数 (R/S 方法)
def hurst_exponent(ts):
    if len(ts) < 10:  # 避免小窗口计算不准
        return np.nan
    lags = range(2, min(100, len(ts)//2))
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    if any(t == 0 for t in tau):  # 避免 log(0)
        return np.nan
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

# 解析命令行参数
parser = argparse.ArgumentParser(description='Extract rolling Fourier and Wavelet features from financial time series CSV.')
parser.add_argument('--filename', type=str, required=True, help='Path to the CSV file with columns: date, open, close, high, low, volume, marketcap')
args = parser.parse_args()

# 读取 CSV
df = pd.read_csv(args.filename)
df['Date'] = pd.to_datetime(df['Date'])  # 确保 Date 是 datetime 类型
df = df.sort_values('Date').reset_index(drop=True)  # 按日期排序

# 定义窗口大小
window_size = 14

# 初始化特征列
df['fourier_amp_1'] = np.nan
df['fourier_amp_2'] = np.nan
df['fourier_amp_3'] = np.nan  # 前3个主导幅度
df['wavelet_energy_approx'] = np.nan  # 近似系数能量
df['wavelet_energy_detail1'] = np.nan  # 细节系数能量 (level 1)
df['wavelet_energy_detail2'] = np.nan  # level 2
df['emd_energy_imf1'] = np.nan  # EMD 第一 IMF 能量
df['emd_energy_imf2'] = np.nan  # 第二 IMF
df['vmd_energy_mode1'] = np.nan  # VMD 第一模态能量
df['vmd_energy_mode2'] = np.nan  # 第二模态
df['ewt_energy_mode1'] = np.nan  # EWT 第一模态能量
df['ewt_energy_mode2'] = np.nan  # 第二模态
df['shocklet_energy_mode'] = np.nan  # Shocklet 序列能量 (简化单模态)
df['hurst_exponent'] = np.nan  # Hurst Exponent

# 滚动计算特征
for i in range(window_size - 1, len(df)):
    # window_prices = df['Close'].iloc[i - window_size + 1 : i + 1].values
    window_prices = df['Close'].iloc[i - window_size + 1 : i + 1].values.copy()
    
    # 傅里叶变换
    N = len(window_prices)
    fft_values = fft(window_prices)
    amplitudes = np.abs(fft_values) / N
    positive_freq_mask = fftfreq(N) > 0
    sorted_indices = np.argsort(amplitudes[positive_freq_mask])[::-1][:3]  # 前3主导
    dominant_amps = amplitudes[positive_freq_mask][sorted_indices]
    
    df.at[i, 'fourier_amp_1'] = dominant_amps[0] if len(dominant_amps) > 0 else np.nan
    df.at[i, 'fourier_amp_2'] = dominant_amps[1] if len(dominant_amps) > 1 else np.nan
    df.at[i, 'fourier_amp_3'] = dominant_amps[2] if len(dominant_amps) > 2 else np.nan
    
    # 小波变换 (db4, level=2) -> 动态level避免警告
    max_level = int(np.log2(len(window_prices) - 1))  # 计算最大可能level
    level = min(2, max_level) if max_level > 0 else 1  # 限制到1或2
    coeffs = pywt.wavedec(window_prices, 'db4', level=level)
    energies = [np.sum(np.abs(c)**2) for c in coeffs]  # 统一用模平方能量（实数）
    
    df.at[i, 'wavelet_energy_approx'] = energies[0]
    df.at[i, 'wavelet_energy_detail1'] = energies[1] if len(energies) > 1 else np.nan
    df.at[i, 'wavelet_energy_detail2'] = energies[2] if len(energies) > 2 else np.nan
    
    # EMD (使用 PyEMD, 分解成 IMFs, 取前2能量) -> 优化边界处理
    emd = EMD(spline_kind='akima')  # 改用'akima'边界插值，提高分解稳定性
    IMFs = emd(window_prices)
    if len(IMFs) >= 2:
        df.at[i, 'emd_energy_imf1'] = np.sum(np.abs(IMFs[0])**2)
        df.at[i, 'emd_energy_imf2'] = np.sum(np.abs(IMFs[1])**2)
    else:
        # 可选填充：用第一个IMF重复或0填充，视需求
        df.at[i, 'emd_energy_imf1'] = np.sum(np.abs(IMFs[0])**2) if len(IMFs) > 0 else np.nan
        df.at[i, 'emd_energy_imf2'] = 0  # 或np.nan，继续空值

    # VMD (使用 vmdpy, 分解成2模态, 取能量) -> 用模平方避免复数
    num_modes = 2
    modes = vmdpy.VMD(window_prices, alpha=2000, tau=0, K=num_modes, DC=0, init=1, tol=1e-7)
    if modes is not None and len(modes) >= 2:
        df.at[i, 'vmd_energy_mode1'] = np.sum(np.abs(modes[0])**2)
        df.at[i, 'vmd_energy_mode2'] = np.sum(np.abs(modes[1])**2)
    
    # EWT (使用 ewtpy, 分解成2模态, 取能量)
    ewt, _, _ = ewtpy.EWT1D(window_prices, N=2)
    if ewt.shape[1] >= 2:
        df.at[i, 'ewt_energy_mode1'] = np.sum(np.abs(ewt[:,0])**2)
        df.at[i, 'ewt_energy_mode2'] = np.sum(np.abs(ewt[:,1])**2)
    
    # Shocklet Transform (自定义简化, 计算 shocklet 序列能量)
    shocklets = shocklet_transform(window_prices)
    df.at[i, 'shocklet_energy_mode'] = np.sum(np.abs(shocklets)**2)  # 统一模平方
    
    # Hurst Exponent
    df.at[i, 'hurst_exponent'] = hurst_exponent(window_prices)

# 保存新 CSV
output_filename = args.filename
df.to_csv(output_filename, index=False)
print(f"特征数据已保存到 {output_filename}")