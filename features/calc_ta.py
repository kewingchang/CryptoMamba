import argparse
import pandas as pd
import numpy as np
import talib 
import pandas_ta as pta 
from ta import add_all_ta_features 


def calculate_fib_retracement(high, low):
    """Calculate Fibonacci retracement levels for a given high-low range."""
    diff = high - low
    levels = {
        'fib_0': low,
        'fib_236': low + 0.236 * diff,
        'fib_382': low + 0.382 * diff,
        'fib_500': low + 0.5 * diff,
        'fib_618': low + 0.618 * diff,
        'fib_786': low + 0.786 * diff,
        'fib_100': high
    }
    return levels

def main():
    parser = argparse.ArgumentParser(description="Calculate Comprehensive Technical Indicators (Merged Version).")
    parser.add_argument('--filename', type=str, required=True, help="The CSV filename to process.")
    args = parser.parse_args()
    filename = args.filename

    # 1. 读取与清洗
    print(f"Reading {filename}...")
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error: {e}")
        return

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
    
    # 必需列检查
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = df[col].astype(np.float64)

    # 提取 numpy 数组加速 talib
    open_p = df['Open'].values
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    volume = df['Volume'].values
    
    # 防除零微小值
    epsilon = 1e-9

    print("Step 1: Restoring Specific Features from Original Script...")
    
    # --- 1. Momentum / Oscillators (Specific Parameters) ---
    # Stochastic (Slow & Fast) - 原始参数 restoration
    # Slow: 14, 3, 3
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['Stoch_K'] = slowk
    df['Stoch_D'] = slowd

    # SLOWK and SLOWD
    slowk_5, slowd_5 = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['SlowK_5'] = slowk_5
    df['SlowD_5'] = slowd_5
    
    # Fast: 5, 3
    fastk, fastd = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['FastK_5'] = fastk
    df['FastD_5'] = fastd

    # Stoch RSI
    fastk_rsi, fastd_rsi = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['FastK_RSI_14'] = fastk_rsi
    df['FastD_RSI_14'] = fastd_rsi
    
    # Ultosc
    df['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    
    # Williams R
    df['WILLR_14'] = talib.WILLR(high, low, close, timeperiod=14)
    
    # APO / PPO
    df['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    df['PPO'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
    
    # MOM
    df['MOM_10'] = talib.MOM(close, timeperiod=10)

    df = df.copy()
    
    # BOP (Balance Of Power)
    df['BOP'] = talib.BOP(open_p, high, low, close)
    
    # CCI Specific
    df['CCI_14'] = talib.CCI(high, low, close, timeperiod=14)
    df['CCI_20'] = talib.CCI(high, low, close, timeperiod=20)
    
    # CMO
    df['CMO_14'] = talib.CMO(close, timeperiod=14)
    
    # ROC Family (Rate of Change) - Specific 10 period
    df['ROC_10'] = talib.ROC(close, timeperiod=10)
    df['ROCP_10'] = talib.ROCP(close, timeperiod=10)
    df['ROCR_10'] = talib.ROCR(close, timeperiod=10)
    df['ROCR100_10'] = talib.ROCR100(close, timeperiod=10)

    df = df.copy()
    
    # TRIX
    df['TRIX_30'] = talib.TRIX(close, timeperiod=30)
    
    # --- 2. Directional Indicators (ADX, DI, DM) ---
    df['ADX_14'] = talib.ADX(high, low, close, timeperiod=14)
    df['ADXR_14'] = talib.ADXR(high, low, close, timeperiod=14)
    df['DX_14'] = talib.DX(high, low, close, timeperiod=14)
    df['MINUS_DI_14'] = talib.MINUS_DI(high, low, close, timeperiod=14)
    df['PLUS_DI_14'] = talib.PLUS_DI(high, low, close, timeperiod=14)
    df['MINUS_DM_14'] = talib.MINUS_DM(high, low, timeperiod=14)
    df['PLUS_DM_14'] = talib.PLUS_DM(high, low, timeperiod=14)
    
    # --- 3. MACD Variations ---
    # Standard MACD is covered in multi-scale, but let's keep specific named ones
    # MACD
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal
    df['MACD_Hist'] = macdhist

    # MACDEXT
    macd_ext, macdsignal_ext, macdhist_ext = talib.MACDEXT(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    df['MACDEXT'] = macd_ext
    df['MACDEXT_Signal'] = macdsignal_ext
    df['MACDEXT_Hist'] = macdhist_ext

    # MACDFIX
    macd_fix, macdsignal_fix, macdhist_fix = talib.MACDFIX(close, signalperiod=9)
    df['MACDFIX'] = macd_fix
    df['MACDFIX_Signal'] = macdsignal_fix
    df['MACDFIX_Hist'] = macdhist_fix

    df = df.copy()

    # --- 4. Moving Averages (Advanced Types) ---
    # DEMA, TEMA, TRIMA, KAMA, MAMA, T3
    df['DEMA_30'] = talib.DEMA(close, timeperiod=30)
    df['TEMA_30'] = talib.TEMA(close, timeperiod=30)
    df['TRIMA_30'] = talib.TRIMA(close, timeperiod=30)
    df['WMA_30'] = talib.WMA(close, timeperiod=30)
    df['KAMA_30'] = talib.KAMA(close, timeperiod=30)
    df['T3_5'] = talib.T3(close, timeperiod=5, vfactor=0)
    
    mama, fama = talib.MAMA(close, fastlimit=0.5, slowlimit=0.05)
    df['MAMA'] = mama
    df['FAMA'] = fama

    # MIDPOINT - MidPoint over period, timeperiod=14
    df['MIDPOINT_14'] = talib.MIDPOINT(close, timeperiod=14)
    # MIDPRICE - Midpoint Price over period, timeperiod=14
    df['MIDPRICE_14'] = talib.MIDPRICE(high, low, timeperiod=14)

    # SAREXT - Parabolic SAR - Extended
    df['SAREXT'] = talib.SAREXT(high, low, startvalue=0, offsetonreverse=0, accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2, accelerationinitshort=0.015, accelerationshort=0.015, accelerationmaxshort=0.15)

    df = df.copy()
    
    df['HT_TRENDLINE'] = talib.HT_TRENDLINE(close)
    
    # Specific SMAs/EMAs from original script
    df['SMA_21'] = talib.SMA(close, timeperiod=21)
    df['SMA_100'] = talib.SMA(close, timeperiod=100)
    df['EMA_21'] = talib.EMA(close, timeperiod=21)
    df['EMA_12'] = talib.EMA(close, timeperiod=12)
    df['EMA_100'] = talib.EMA(close, timeperiod=100)
    df['MA_21'] = talib.MA(close, timeperiod=21)
    df['MA_100'] = talib.MA(close, timeperiod=100)

    # Bollinger Bands(20,2)
    upper_20, middle_20, lower_20 = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['BB_Upper'] = upper_20
    df['BB_Middle'] = middle_20
    df['BB_Lower'] = lower_20

    # BBANDS - Bollinger Bands
    upper_5, middle_5, lower_5 = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df['BB_Upper_5'] = upper_5
    df['BB_Middle_5'] = middle_5
    df['BB_Lower_5'] = lower_5
    
    # --- 5. Volatility (Specific) ---
    df['NATR_14'] = talib.NATR(high, low, close, timeperiod=14)
    df['TRANGE'] = talib.TRANGE(high, low, close)
    df['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)

    df = df.copy()
    
    # --- 6. Cycle Indicators ---
    df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
    df['HT_DCPHASE'] = talib.HT_DCPHASE(close)
    inphase, quadrature = talib.HT_PHASOR(close)
    df['HT_PHASOR_inphase'] = inphase
    df['HT_PHASOR_quadrature'] = quadrature
    sine, leadsine = talib.HT_SINE(close)
    df['HT_SINE_sine'] = sine
    df['HT_SINE_leadsine'] = leadsine
    df['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)
    
    # --- 7. Price Transform / Stats ---
    df['AVGPRICE'] = talib.AVGPRICE(open_p, high, low, close)
    df['MEDPRICE'] = talib.MEDPRICE(high, low)
    df['TYPPRICE'] = talib.TYPPRICE(high, low, close)
    df['WCLPRICE'] = talib.WCLPRICE(high, low, close)
    
    # Aroon Osc specific
    df['AROON_osc_14'] = talib.AROONOSC(high, low, timeperiod=14)
    # Aroon Indicator (25 periods): Measures trend strength with Aroon Up and Down
    aroonup, aroondown = talib.AROON(high, low, timeperiod=25)
    df['Aroon_Up'] = aroonup
    df['Aroon_Down'] = aroondown

    df['CMF_14'] = df.ta.cmf(length=14)
    df['CMF_20'] = df.ta.cmf(length=20)

    # Parabolic SAR (acceleration=0.02, maximum=0.2): Dynamic trend reversal indicator
    df['PSAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
    
    # MAVP
    periods = np.nan_to_num(df['HT_DCPERIOD'], nan=14).clip(2, 30).astype(np.float64)
    df['MAVP'] = talib.MAVP(close, periods, minperiod=2, maxperiod=30, matype=0)

    df = df.copy()

    print("Step 2: Generating Multi-scale Features (Enhanced)...")
    # ==========================================
    # 多尺度窗口策略
    # ==========================================
    windows = [7, 14, 30, 90]

    for w in windows:
        suffix = f"_{w}"
        
        # 避免重复覆盖已经计算过的特定指标 (如 RSI_14)
        # 如果列已存在，我们只需计算"平稳化"版本
        
        # --- A. 基础指标 ---
        if f'RSI{suffix}' not in df.columns:
            df[f'RSI{suffix}'] = talib.RSI(close, timeperiod=w)
            
        if f'CCI{suffix}' not in df.columns:
            df[f'CCI{suffix}'] = talib.CCI(high, low, close, timeperiod=w)
            
        df[f'ROC{suffix}'] = talib.ROC(close, timeperiod=w)
        
        # --- B. 均线与乖离率 (保留原始 + 新增Dist) ---
        sma = talib.SMA(close, timeperiod=w)
        df[f'SMA{suffix}'] = sma
        df[f'Dist_SMA{suffix}'] = (close - sma) / (sma + epsilon)
        
        ema = talib.EMA(close, timeperiod=w)
        df[f'EMA{suffix}'] = ema
        df[f'Dist_EMA{suffix}'] = (close - ema) / (ema + epsilon)
        
        # --- C. 布林带 (保留原始 + 新增%B) ---
        upper, middle, lower = talib.BBANDS(close, timeperiod=w, nbdevup=2, nbdevdn=2)
        df[f'BB_Upper{suffix}'] = upper
        df[f'BB_Middle{suffix}'] = middle
        df[f'BB_Lower{suffix}'] = lower
        df[f'BB_PctB{suffix}'] = (close - lower) / (upper - lower + epsilon)
        df[f'BB_Width{suffix}'] = (upper - lower) / (middle + epsilon)

        # --- D. Donchian / Fib ---
        roll_high = df['High'].rolling(w).max()
        roll_low = df['Low'].rolling(w).min()
        df[f'Donchian_High{suffix}'] = roll_high
        df[f'Donchian_Low{suffix}'] = roll_low
        df[f'Price_in_Range{suffix}'] = (df['Close'] - roll_low) / (roll_high - roll_low + epsilon)
    
    df = df.copy()

    print("Step 3: Calculating Safe Slope Features (Replacing Cumulative)...")
    # 替代原始的 OBV/AD，计算其斜率 (Stationary)    
    # AD Slope
    ad = talib.AD(high, low, close, volume)
    df['AD_Slope_14'] = talib.LINEARREG_SLOPE(ad, timeperiod=14) / (df['Volume'].rolling(14).mean() + epsilon)

    print("Step 4: Adding 'ta' library features...")
    # 添加 ta 库所有特征
    df = add_all_ta_features(df, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=False)
    
    df = df.copy()

    print("Step 5: Adding Pandas TA features...")
    # SuperTrend
    st = pta.supertrend(df['High'], df['Low'], df['Close'], length=14, multiplier=3.0)
    if st is not None:
        df = pd.concat([df, st], axis=1)

    # pandas_ta indicators (select useful ones)
    # Ichimoku Cloud
    ichimoku, _ = pta.ichimoku(high=df['High'], low=df['Low'], close=df['Close'])
    df = pd.concat([df, ichimoku], axis=1)  # Adds ISA_9, ISB_26, IKS_26, ICS_26, etc.
    
    # Choppiness Index
    df['CHOP_14'] = pta.chop(df['High'], df['Low'], df['Close'], length=14)
    
    # Fib Retracement (Rolling 14)
    rolling_high = df['High'].rolling(window=14).max()
    rolling_low = df['Low'].rolling(window=14).min()
    fib_levels = pd.DataFrame([calculate_fib_retracement(h, l) for h, l in zip(rolling_high, rolling_low)], index=df.index)
    df = pd.concat([df, fib_levels.add_prefix('Rolling_14_')], axis=1)
    
    # Custom Price Ratios
    df['Close_to_Open_Ratio'] = close / open_p
    df['High_to_Low_Ratio'] = high / low
    df['Close_to_High_Ratio'] = close / high
    df['Close_to_Low_Ratio'] = close / low
    df['Open_to_High_Ratio'] = open_p / high
    df['Open_to_Low_Ratio'] = open_p / low
    # High-Low Spread
    df['High_Low_Spread'] = high - low
    # Open-Close Spread
    df['Open_Close_Spread'] = open_p - close

    # EWMA (Exponential Weighted Moving Average, span=14)
    df['EWMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()
    df['bar_strength'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'] + epsilon)
    
    df = df.copy()
    
    print("Step 6: Cleaning ONLY Cumulative Indicators...")
    
    # 定义必须删除的累积型指标关键词 (只删这几个)
    # 这些指标的值随时间无限增长，必须清洗
    cumulative_cols_to_drop = []
    
    # 精确列表
    cumulative_list = [
        'volume_obv', 'volume_nvi', 'volume_pvi', 'volume_adi', 
        'trend_ad' # ta库生成的累积AD线
    ]

    for col in df.columns:
        if col in cumulative_list:
            cumulative_cols_to_drop.append(col)
    
    if cumulative_cols_to_drop:
        print(f"Dropping {len(cumulative_cols_to_drop)} cumulative columns: {cumulative_cols_to_drop}")
        df.drop(columns=cumulative_cols_to_drop, inplace=True, errors='ignore')

    # 收尾
    df = df.reset_index()
    print(f"Final column count: {len(df.columns)}")
    print("Saving to CSV...")
    df.to_csv(filename, index=False)
    print("Done.")

if __name__ == "__main__":
    main()