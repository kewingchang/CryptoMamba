import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
import pywt
import argparse
import warnings
import sys

# 尝试导入高级信号处理库，如果缺失则报错或提示
try:
    from PyEMD import EMD  # pip install EMD-signal
except ImportError:
    print("Error: PyEMD not found. Please install via 'pip install EMD-signal'")
    sys.exit(1)

try:
    import vmdpy  # pip install vmdpy
except ImportError:
    print("Error: vmdpy not found. Please install via 'pip install vmdpy'")
    sys.exit(1)

try:
    import ewtpy  # pip install ewtpy
except ImportError:
    print("Error: ewtpy not found. Please install via 'pip install ewtpy'")
    sys.exit(1)

warnings.filterwarnings("ignore")

# Shocklet Transform 自定义实现
def shocklet_transform(ts, tau=1, alpha=1.0):
    diff = np.diff(ts)
    shocklets = np.cumsum(np.exp(-alpha * np.arange(len(diff))) * diff)
    if len(shocklets) < len(ts):
        shocklets = np.pad(shocklets, (0, 1), mode='constant')
    return shocklets

# Hurst Exponent 计算函数 (R/S 方法)
def hurst_exponent(ts):
    n = len(ts)
    if n < 20:  # 窗口太小，统计学意义不大，返回 NaN
        return np.nan
    
    # 检查是否为常数序列（方差为0），防止除以0
    if np.std(ts) == 0:
        return 0.5 # 随机游走状态
        
    lags = range(2, min(n // 2, 20)) # 限制 lag 范围
    tau = []
    for lag in lags:
        # 简单 R/S 估算
        std_val = np.std(np.subtract(ts[lag:], ts[:-lag]))
        if std_val == 0:
            tau.append(1e-10) # 避免 log(0)
        else:
            tau.append(std_val)
            
    if len(tau) < 2: 
        return np.nan
        
    try:
        # 拟合双对数坐标
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except:
        return np.nan

def main():
    parser = argparse.ArgumentParser(description='Extract rolling Frequency & Wavelet features.')
    parser.add_argument('--filename', type=str, required=True, help='Path to the CSV file')
    # 将窗口大小参数化，默认 30 以保证 EMD/VMD/Hurst 的稳定性
    parser.add_argument('--window', type=int, default=30, help='Rolling window size (default: 30)')
    args = parser.parse_args()

    # 读取 CSV
    try:
        df = pd.read_csv(args.filename)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # 必要的预处理
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    
    # 确保 Close 列存在且为数值
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close']).reset_index(drop=True)

    window_size = args.window
    print(f"Processing data with Window Size = {window_size}...")
    print("Features: FFT, Wavelet, EMD, VMD, EWT, Shocklet, Hurst")

    # 初始化特征列 (使用 float 类型以支持 NaN)
    new_features = [
        'fourier_amp_1', 'fourier_amp_2', 'fourier_amp_3',
        'wavelet_energy_approx', 'wavelet_energy_detail1', 'wavelet_energy_detail2',
        'emd_energy_imf1', 'emd_energy_imf2',
        'vmd_energy_mode1', 'vmd_energy_mode2',
        'ewt_energy_mode1', 'ewt_energy_mode2',
        'shocklet_energy_mode', 'hurst_exponent'
    ]
    
    for col in new_features:
        df[col] = np.nan

    # 预初始化对象以提升性能
    # PyWavelets 对象
    wavelet_name = 'db4'
    
    # EMD 对象
    emd = EMD(spline_kind='akima')

    # 遍历计算
    total_rows = len(df)
    for i in range(window_size - 1, total_rows):
        # 简单的进度打印
        if i % 100 == 0:
            print(f"Processing row {i}/{total_rows}...", end='\r')

        # 获取窗口数据 (复制以避免切片视图警告)
        window_prices = df['Close'].iloc[i - window_size + 1 : i + 1].values.copy()
        
        # 0. 基础检查：如果窗口内含有 NaN，跳过
        if np.isnan(window_prices).any():
            continue

        # --- 1. Fourier Transform ---
        try:
            N = len(window_prices)
            fft_values = fft(window_prices)
            amplitudes = np.abs(fft_values) / N
            # 排除直流分量 (0Hz)
            pos_mask = fftfreq(N) > 0
            valid_amps = amplitudes[pos_mask]
            if len(valid_amps) > 0:
                sorted_amps = np.sort(valid_amps)[::-1] # 降序
                df.at[i, 'fourier_amp_1'] = sorted_amps[0] if len(sorted_amps) > 0 else 0
                df.at[i, 'fourier_amp_2'] = sorted_amps[1] if len(sorted_amps) > 1 else 0
                df.at[i, 'fourier_amp_3'] = sorted_amps[2] if len(sorted_amps) > 2 else 0
        except Exception: pass

        # --- 2. Wavelet Transform ---
        try:
            # 动态计算最大层级，防止窗口过小报错
            max_level = pywt.dwt_max_level(len(window_prices), pywt.Wavelet(wavelet_name).dec_len)
            level = min(2, max_level)
            if level > 0:
                coeffs = pywt.wavedec(window_prices, wavelet_name, level=level)
                # coeffs[0] 是近似系数(Approx), coeffs[1:] 是细节系数(Details)
                df.at[i, 'wavelet_energy_approx'] = np.sum(np.abs(coeffs[0])**2)
                if len(coeffs) > 1:
                    df.at[i, 'wavelet_energy_detail1'] = np.sum(np.abs(coeffs[1])**2)
                if len(coeffs) > 2:
                    df.at[i, 'wavelet_energy_detail2'] = np.sum(np.abs(coeffs[2])**2)
        except Exception: pass

        # --- 3. EMD (Empirical Mode Decomposition) ---
        try:
            # max_imf=2 限制分解层数，提高性能
            IMFs = emd(window_prices, max_imf=3)
            if IMFs is not None and len(IMFs) > 0:
                df.at[i, 'emd_energy_imf1'] = np.sum(np.abs(IMFs[0])**2)
                if len(IMFs) > 1:
                    df.at[i, 'emd_energy_imf2'] = np.sum(np.abs(IMFs[1])**2)
                else:
                    df.at[i, 'emd_energy_imf2'] = 0
        except Exception: pass

        # --- 4. VMD (Variational Mode Decomposition) ---
        try:
            # alpha: 带宽限制 (2000是常用值), tau: 噪声容忍度
            # vmdpy.VMD 返回 (u, u_hat, omega)
            u, _, _ = vmdpy.VMD(window_prices, alpha=2000, tau=0, K=2, DC=0, init=1, tol=1e-7)
            if u is not None and len(u) >= 2:
                df.at[i, 'vmd_energy_mode1'] = np.sum(np.abs(u[0])**2)
                df.at[i, 'vmd_energy_mode2'] = np.sum(np.abs(u[1])**2)
        except Exception: pass

        # --- 5. EWT (Empirical Wavelet Transform) ---
        try:
            # EWT1D 返回 (ewt, mra, boundaries)
            ewt, _, _ = ewtpy.EWT1D(window_prices, N=2)
            # ewt shape 是 (T, N)，需要转置或切片
            if ewt is not None and ewt.shape[1] >= 2:
                df.at[i, 'ewt_energy_mode1'] = np.sum(np.abs(ewt[:, 0])**2)
                df.at[i, 'ewt_energy_mode2'] = np.sum(np.abs(ewt[:, 1])**2)
        except Exception: pass

        # --- 6. Shocklet & Hurst ---
        try:
            shocklets = shocklet_transform(window_prices)
            df.at[i, 'shocklet_energy_mode'] = np.sum(np.abs(shocklets)**2)
        except Exception: pass

        df.at[i, 'hurst_exponent'] = hurst_exponent(window_prices)

    print("\nCalculation complete.")
    
    # 保存文件
    df.to_csv(args.filename, index=False)
    print(f"Features saved to {args.filename}")

if __name__ == "__main__":
    main()