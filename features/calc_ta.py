import argparse
import pandas as pd
import talib  # pip install TA-Lib
import numpy as np
import pandas_ta as pta  # pip install pandas_ta
from ta import add_all_ta_features  # pip install ta
from statsmodels.tsa.api import ExponentialSmoothing  # For EWMA alternative if needed


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

def apply_rolling_df(df, window_size, func):
    """Custom rolling apply to pass sub-DataFrame to func."""
    result = pd.Series(np.nan, index=df.index)
    for i in range(window_size - 1, len(df)):
        window = df.iloc[i - window_size + 1 : i + 1]
        result.iloc[i] = func(window)
    return result

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate technical indicators and update CSV file.")
    parser.add_argument('--filename', type=str, required=True, help="The CSV filename to process.")
    args = parser.parse_args()

    filename = args.filename

    # Read the CSV file
    df = pd.read_csv(filename, parse_dates=['Date'])

    # Set index to Date and sort
    df = df.set_index('Date')
    df = df.sort_index()

    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert relevant columns to float64 to ensure compatibility with TA-Lib
    df['Open'] = df['Open'].astype(np.float64)
    df['High'] = df['High'].astype(np.float64)
    df['Low'] = df['Low'].astype(np.float64)
    df['Close'] = df['Close'].astype(np.float64)
    df['Volume'] = df['Volume'].astype(np.float64)

    # Convert to numpy arrays for talib
    open_price = df['Open'].values
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    volume = df['Volume'].values

    # 1. RSI(14)
    df['RSI_14'] = talib.RSI(close, timeperiod=14)

    # 2. MACD(12/26/9)
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal
    df['MACD_Hist'] = macdhist

    # 3. EMA(12)
    df['EMA_12'] = talib.EMA(close, timeperiod=12)

    # 4. Bollinger Bands(20,2)
    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower

    # 5. Stochastic Oscillator, Stoch-14 (%K=14, %D=3)
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['Stoch_K'] = slowk
    df['Stoch_D'] = slowd

    # 6. ATR(14)
    df['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)

    df = df.copy()  # 碎片化修复：添加copy

    # Additional complex derived indicators from OHLCV

    # Aroon Indicator (25 periods): Measures trend strength with Aroon Up and Down
    aroonup, aroondown = talib.AROON(high, low, timeperiod=25)
    df['Aroon_Up'] = aroonup
    df['Aroon_Down'] = aroondown

    # Chaikin Money Flow approximation using Chaikin A/D Oscillator (fast=3, slow=10) -> 使用apply_rolling_df
    def calc_adosc(window):
        return talib.ADOSC(window['High'].values, window['Low'].values, window['Close'].values, window['Volume'].values, fastperiod=3, slowperiod=10)[-1]
    df['CMF'] = apply_rolling_df(df, 14, calc_adosc)

    # Parabolic SAR (acceleration=0.02, maximum=0.2): Dynamic trend reversal indicator
    df['PSAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)

    df = df.copy()  # 碎片化修复

    # Additional recommended indicators from OHLCV

    # ADX (14 periods): Average Directional Index for trend strength
    df['ADX_14'] = talib.ADX(high, low, close, timeperiod=14)

    # CCI (20 periods): Commodity Channel Index for momentum
    df['CCI_20'] = talib.CCI(high, low, close, timeperiod=20)

    # OBV: On-Balance Volume -> 使用apply_rolling_df
    def calc_obv(window):
        return talib.OBV(window['Close'].values, window['Volume'].values)[-1]
    df['OBV'] = apply_rolling_df(df, 14, calc_obv)

    df = df.copy()  # 碎片化修复

    # Momentum Indicators (new additions)

    # ADXR - Average Directional Movement Index Rating, timeperiod=14
    df['ADXR_14'] = talib.ADXR(high, low, close, timeperiod=14)

    # APO - Absolute Price Oscillator, fastperiod=12, slowperiod=26, matype=0
    df['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)

    # AROON_osc - Aroon Oscillator, timeperiod=14
    df['AROON_osc_14'] = talib.AROONOSC(high, low, timeperiod=14)

    # BOP - Balance Of Power
    df['BOP'] = talib.BOP(open_price, high, low, close)

    # CCI - Commodity Channel Index, timeperiod=14 (different from existing CCI_20)
    df['CCI_14'] = talib.CCI(high, low, close, timeperiod=14)

    # CMO - Chande Momentum Oscillator, timeperiod=14
    df['CMO_14'] = talib.CMO(close, timeperiod=14)

    # DX - Directional Movement Index, timeperiod=14
    df['DX_14'] = talib.DX(high, low, close, timeperiod=14)

    # MACDEXT - MACD with controllable MA type, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0
    macd_ext, macdsignal_ext, macdhist_ext = talib.MACDEXT(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    df['MACDEXT'] = macd_ext
    df['MACDEXT_Signal'] = macdsignal_ext
    df['MACDEXT_Hist'] = macdhist_ext

    # MACDFIX - Moving Average Convergence Divergence Fix 12/26, signalperiod=9
    macd_fix, macdsignal_fix, macdhist_fix = talib.MACDFIX(close, signalperiod=9)
    df['MACDFIX'] = macd_fix
    df['MACDFIX_Signal'] = macdsignal_fix
    df['MACDFIX_Hist'] = macdhist_fix

    # MFI - Money Flow Index, timeperiod=14
    df['MFI_14'] = talib.MFI(high, low, close, volume, timeperiod=14)

    # MINUS_DI - Minus Directional Indicator, timeperiod=14
    df['MINUS_DI_14'] = talib.MINUS_DI(high, low, close, timeperiod=14)

    # MINUS_DM - Minus Directional Movement, timeperiod=14
    df['MINUS_DM_14'] = talib.MINUS_DM(high, low, timeperiod=14)

    # MOM - Momentum, timeperiod=10
    df['MOM_10'] = talib.MOM(close, timeperiod=10)

    # PLUS_DI - Plus Directional Indicator, timeperiod=14
    df['PLUS_DI_14'] = talib.PLUS_DI(high, low, close, timeperiod=14)

    # PLUS_DM - Plus Directional Movement, timeperiod=14
    df['PLUS_DM_14'] = talib.PLUS_DM(high, low, timeperiod=14)

    # PPO - Percentage Price Oscillator, fastperiod=12, slowperiod=26, matype=0
    df['PPO'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)

    # ROC - Rate of change : ((price/prevPrice)-1)*100, timeperiod=10
    df['ROC_10'] = talib.ROC(close, timeperiod=10)

    # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice, timeperiod=10
    df['ROCP_10'] = talib.ROCP(close, timeperiod=10)

    # ROCR - Rate of change ratio: (price/prevPrice), timeperiod=10
    df['ROCR_10'] = talib.ROCR(close, timeperiod=10)

    # ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100, timeperiod=10
    df['ROCR100_10'] = talib.ROCR100(close, timeperiod=10)

    # SLOWK and SLOWD - Stochastic, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
    slowk_5, slowd_5 = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['SlowK_5'] = slowk_5
    df['SlowD_5'] = slowd_5

    # FASTK and FASTD - Stochastic Fast, fastk_period=5, fastd_period=3, fastd_matype=0
    fastk, fastd = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['FastK_5'] = fastk
    df['FastD_5'] = fastd

    # FASTK_rsi and FASTD_rsi - Stochastic Relative Strength Index, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0
    fastk_rsi, fastd_rsi = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['FastK_RSI_14'] = fastk_rsi
    df['FastD_RSI_14'] = fastd_rsi

    # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA, timeperiod=30
    df['TRIX_30'] = talib.TRIX(close, timeperiod=30)

    # ULTOSC - Ultimate Oscillator, timeperiod1=7, timeperiod2=14, timeperiod3=28
    df['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

    # WILLR - Williams' %R, timeperiod=14
    df['WILLR_14'] = talib.WILLR(high, low, close, timeperiod=14)

    df = df.copy()  # 碎片化修复

    # Overlap Indicators (new additions)

    # BBANDS - Bollinger Bands, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0 (different from existing timeperiod=20)
    upper_5, middle_5, lower_5 = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df['BB_Upper_5'] = upper_5
    df['BB_Middle_5'] = middle_5
    df['BB_Lower_5'] = lower_5

    # DEMA - Double Exponential Moving Average, timeperiod=30
    df['DEMA_30'] = talib.DEMA(close, timeperiod=30)

    # EMA - Exponential Moving Average, several timeperiods used. ex. EMA_21, EMA_100
    df['EMA_21'] = talib.EMA(close, timeperiod=21)
    df['EMA_100'] = talib.EMA(close, timeperiod=100)

    # HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
    df['HT_TRENDLINE'] = talib.HT_TRENDLINE(close)

    # KAMA - Kaufman Adaptive Moving Average, timeperiod=30
    df['KAMA_30'] = talib.KAMA(close, timeperiod=30)

    # MA - Moving average, several timeperiods used. ex. MA_21, MA_100
    df['MA_21'] = talib.MA(close, timeperiod=21)
    df['MA_100'] = talib.MA(close, timeperiod=100)

    # MAMA and FAMA - MESA Adaptive Moving Average, fastlimit=0.5, slowlimit=0.05
    mama, fama = talib.MAMA(close, fastlimit=0.5, slowlimit=0.05)
    df['MAMA'] = mama
    df['FAMA'] = fama

    # MIDPOINT - MidPoint over period, timeperiod=14
    df['MIDPOINT_14'] = talib.MIDPOINT(close, timeperiod=14)

    # MIDPRICE - Midpoint Price over period, timeperiod=14
    df['MIDPRICE_14'] = talib.MIDPRICE(high, low, timeperiod=14)

    # SAREXT - Parabolic SAR - Extended
    df['SAREXT'] = talib.SAREXT(high, low, startvalue=0, offsetonreverse=0, accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2, accelerationinitshort=0.015, accelerationshort=0.015, accelerationmaxshort=0.15)

    # SMA - Simple Moving Average, several timeperiods used. ex. SMA_21, SMA_100
    df['SMA_21'] = talib.SMA(close, timeperiod=21)
    df['SMA_100'] = talib.SMA(close, timeperiod=100)

    # T3 - Triple Exponential Moving Average (T3), timeperiod=5, vfactor=0
    df['T3_5'] = talib.T3(close, timeperiod=5, vfactor=0)

    # TEMA - Triple Exponential Moving Average, timeperiod=30
    df['TEMA_30'] = talib.TEMA(close, timeperiod=30)

    # TRIMA - Triangular Moving Average, timeperiod=30
    df['TRIMA_30'] = talib.TRIMA(close, timeperiod=30)

    # WMA - Weighted Moving Average, timeperiod=30
    df['WMA_30'] = talib.WMA(close, timeperiod=30)

    df = df.copy()  # 碎片化修复

    # Volatility Indicators (new additions)

    # NATR - Normalized Average True Range, timeperiod=14
    df['NATR_14'] = talib.NATR(high, low, close, timeperiod=14)

    # TRANGE - True Range
    df['TRANGE'] = talib.TRANGE(high, low, close)

    df = df.copy()  # 碎片化修复

    # Volume Indicators (new additions)

    # AD - Chaikin A/D Line -> 使用apply_rolling_df
    def calc_ad(window):
        return talib.AD(window['High'].values, window['Low'].values, window['Close'].values, window['Volume'].values)[-1]
    df['AD'] = apply_rolling_df(df, 14, calc_ad)

    # High-Low Spread
    df['High_Low_Spread'] = df['High'] - df['Low']

    # Open-Close Spread
    df['Open_Close_Spread'] = df['Open'] - df['Close']

    # Log Volume (add 1 to avoid log(0) if Volume can be zero, though typically >0)
    df['Log_Volume'] = np.log(df['Volume'] + 1)

    # Current Volatility (vol_current): 30-day rolling realized volatility based on log returns
    # Calculate log returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    # Rolling standard deviation (unbiased, ddof=1) over 14 days -> 改为ddof=0避免RuntimeWarning
    df['vol_current'] = df['Log_Return'].rolling(window=14).std(ddof=0)
    df['vol_shock'] = df['vol_current'] / (df['vol_current'].rolling(30).mean())

    df = df.copy()  # 碎片化修复

    # New additions from TA-Lib

    # 1. Cycle Indicators
    df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
    df['HT_DCPHASE'] = talib.HT_DCPHASE(close)
    inphase, quadrature = talib.HT_PHASOR(close)
    df['HT_PHASOR_inphase'] = inphase
    df['HT_PHASOR_quadrature'] = quadrature
    sine, leadsine = talib.HT_SINE(close)
    df['HT_SINE_sine'] = sine
    df['HT_SINE_leadsine'] = leadsine
    df['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)

    # 2. Price Transform
    df['AVGPRICE'] = talib.AVGPRICE(open_price, high, low, close)
    df['MEDPRICE'] = talib.MEDPRICE(high, low)
    df['TYPPRICE'] = talib.TYPPRICE(high, low, close)
    df['WCLPRICE'] = talib.WCLPRICE(high, low, close)

    # 为了避免 PerformanceWarning，收集 Pattern Recognition 和 Price Ratios 到一个临时 dict
    new_cols = {}

    # 3. Pattern Recognition (top 5 useful: Doji, Hammer, Engulfing, Morning Star, Evening Star)
    new_cols['CDLDOJI'] = talib.CDLDOJI(open_price, high, low, close)
    new_cols['CDLHAMMER'] = talib.CDLHAMMER(open_price, high, low, close)
    new_cols['CDLENGULFING'] = talib.CDLENGULFING(open_price, high, low, close)
    new_cols['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(open_price, high, low, close)
    new_cols['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(open_price, high, low, close)

    # 4. Price Ratios (custom)
    new_cols['Close_to_Open_Ratio'] = close / open_price
    new_cols['High_to_Low_Ratio'] = high / low
    new_cols['Close_to_High_Ratio'] = close / high
    new_cols['Close_to_Low_Ratio'] = close / low
    new_cols['Open_to_High_Ratio'] = open_price / high
    new_cols['Open_to_Low_Ratio'] = open_price / low

    # 一次性添加这些列
    temp_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, temp_df], axis=1)

    # 5. MAVP (use HT_DCPERIOD as variable periods, clipped to 2-30)
    periods = np.nan_to_num(df['HT_DCPERIOD'], nan=14).clip(2, 30).astype(np.float64)
    df['MAVP'] = talib.MAVP(close, periods, minperiod=2, maxperiod=30, matype=0)

    df = df.copy()  # 去碎片化整个 DataFrame

    # pandas_ta indicators (select useful ones)
    # Ichimoku Cloud
    ichimoku, _ = pta.ichimoku(high=df['High'], low=df['Low'], close=df['Close'])
    df = pd.concat([df, ichimoku], axis=1)  # Adds ISA_9, ISB_26, IKS_26, ICS_26, etc.

    # VWAP -> 使用apply_rolling_df
    def calc_vwap(window):
        return pta.vwap(window['High'], window['Low'], window['Close'], window['Volume']).iloc[-1]
    df['VWAP'] = apply_rolling_df(df, 14, calc_vwap)

    # SuperTrend
    df['SuperTrend'] = pta.supertrend(high=df['High'], low=df['Low'], close=df['Close'], period=7, multiplier=3.0)['SUPERT_7_3.0']

    # Choppiness Index
    df['CHOP_14'] = pta.chop(high=df['High'], low=df['Low'], close=df['Close'], length=14)

    # Accumulation/Distribution Index -> 使用apply_rolling_df
    def calc_adi(window):
        return pta.ad(window['High'], window['Low'], window['Close'], window['Volume']).iloc[-1]
    df['ADI'] = apply_rolling_df(df, 14, calc_adi)

    df = df.copy()  # 再次去碎片化

    # ta library indicators (add all features)
    df = add_all_ta_features(df, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)
    # This adds many like volume_adi, volatility_bbw, trend_macd, etc., but may have overlaps; keep for initial screening

    df = df.copy()  # 碎片化修复

    # Fibonacci Retracement (rolling 14-day max high and min low)
    rolling_high = df['High'].rolling(window=14).max()
    rolling_low = df['Low'].rolling(window=14).min()
    fib_levels = pd.DataFrame([calculate_fib_retracement(h, l) for h, l in zip(rolling_high, rolling_low)], index=df.index)
    df = pd.concat([df, fib_levels.add_prefix('Rolling_14_')], axis=1)

    # EWMA (Exponential Weighted Moving Average, span=14)
    df['EWMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()

    df = df.copy()  # 碎片化修复

    # 增补251202
    # df['rvol'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['bar_strength'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-8)
    log_hl = np.log(df['High'] / df['Low'])
    log_co = np.log(df['Close'] / df['Open'])
    df['gk_vol'] = np.sqrt(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2)
    # 

    # Save the updated DataFrame back to the original CSV file
    df.to_csv(filename, index=True)
    print("Updated CSV file")

if __name__ == "__main__":
    main()