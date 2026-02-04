# encode_time_more.py
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta

def add_cyclic_features(df, col_name, period, prefix):
    """
    辅助函数：为指定列添加 sin/cos 编码
    """
    df[f'{prefix}_sin'] = np.sin(2 * np.pi * df[col_name] / period)
    df[f'{prefix}_cos'] = np.cos(2 * np.pi * df[col_name] / period)
    return df

# Parse command line arguments
parser = argparse.ArgumentParser(description='Add cyclic encoding and next-day features to CSV file.')
parser.add_argument('--filename', type=str, required=True, help='The CSV file to process')
args = parser.parse_args()

# Read the CSV file
df = pd.read_csv(args.filename)

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# ==========================================
# 1. 基础时间特征 (Current Day)
# ==========================================
print("Generating Current Day features...")
df['weekday'] = df['Date'].dt.weekday  # 0=Monday, 6=Sunday
df['month'] = df['Date'].dt.month
df['day_of_year'] = df['Date'].dt.dayofyear
df['day_of_month'] = df['Date'].dt.day

# 标记是否周末 (当前)
df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

# 基础 Sin/Cos 编码 (当前)
# 检查结论：你的编码公式是正确的。
# Month: 1-12, Period 12. Dec(12) -> 2pi -> 0, Jan(1) -> 2pi/12. 完美闭环。
# Weekday: 0-6, Period 7. Sun(6) -> 12pi/7, Mon(0) -> 0. 完美闭环。
df = add_cyclic_features(df, 'weekday', 7, 'wd')
df = add_cyclic_features(df, 'month', 12, 'month')
df = add_cyclic_features(df, 'day_of_year', 365.25, 'day')

# 获取当月总天数
df['days_in_month'] = df['Date'].dt.days_in_month
# 手动计算循环特征
df['day_of_month_sin'] = np.sin(2 * np.pi * (df['day_of_month'] - 1) / df['days_in_month'])
df['day_of_month_cos'] = np.cos(2 * np.pi * (df['day_of_month'] - 1) / df['days_in_month'])


# ==========================================
# 2. 下一日预测特征 (Next Day / Target Context)
# 核心逻辑：在 T 时刻，我们已知 T+1 的日历属性
# ==========================================
print("Generating Next Day (Target) features...")

# 计算下一天的日期对象 (这样做比 shift(-1) 更好，因为保留了最后一行用于实盘预测)
next_dates = df['Date'] + pd.Timedelta(days=1)

# 下一天的基础属性
df['next_weekday'] = next_dates.dt.weekday
df['next_month'] = next_dates.dt.month
df['next_day_of_year'] = next_dates.dt.dayofyear
# 获取明天是几号，以及明天所在月份的总天数
df['next_day_of_month'] = next_dates.dt.day
df['next_days_in_month'] = next_dates.dt.days_in_month  # 动态获取当月总天数


# [重要] 平移后的特征
# 1. 明天是不是周末？ (这比今天是周末更能决定明天的波动)
df['next_is_weekend'] = df['next_weekday'].isin([5, 6]).astype(int)

# 2. 明天是不是月初/月末？ (资金流向关键点)
df['next_is_month_start'] = next_dates.dt.is_month_start.astype(int)
df['next_is_month_end'] = next_dates.dt.is_month_end.astype(int)
df['next_is_quarter_end'] = next_dates.dt.is_quarter_end.astype(int) # 季度交割日

# 3. 下一天的 Sin/Cos 编码
# 模型应该知道它正处于周期的哪个位置进入明天
df['next_wd_sin'] = np.sin(2 * np.pi * df['next_weekday'] / 7)
df['next_wd_cos'] = np.cos(2 * np.pi * df['next_weekday'] / 7)

df['next_month_sin'] = np.sin(2 * np.pi * df['next_month'] / 12)
df['next_month_cos'] = np.cos(2 * np.pi * df['next_month'] / 12)

df['next_day_sin'] = np.sin(2 * np.pi * df['next_day_of_year'] / 365.25)
df['next_day_cos'] = np.cos(2 * np.pi * df['next_day_of_year'] / 365.25)

df['day_sin'] = np.sin(2 * np.pi * df['Date'].dt.dayofyear / 365.25)
df['day_cos'] = np.cos(2 * np.pi * df['Date'].dt.dayofyear / 365.25)

df['next_day_of_month_sin'] = np.sin(2 * np.pi * (df['next_day_of_month'] - 1) / df['next_days_in_month'])
df['next_day_of_month_cos'] = np.cos(2 * np.pi * (df['next_day_of_month'] - 1) / df['next_days_in_month'])

def get_weekend_phase(dt):
    wd = dt.weekday()
    if wd in [0, 1, 2, 3]: return 0.0   # Mon-Thu
    if wd == 4: return 0.5              # Fri (Entering Weekend)
    if wd == 5: return 1.0              # Sat (Deep Weekend)
    if wd == 6: return -0.5             # Sun (Exiting Weekend)
    return 0.0

df['weekend_phase'] = df['Date'].apply(get_weekend_phase)


# ==========================================
# 3. 减半周期特征 (Bitcoin Halving)
# ==========================================
print("Generating Halving Cycle features...")
halving_dates = pd.to_datetime([
    '2012-11-28', '2016-07-09', '2020-05-11', '2024-04-20', '2028-03-31'
])

# Current Halving Status
df['last_halving'] = df['Date'].apply(lambda x: halving_dates[halving_dates <= x].max())
df['next_halving'] = df['Date'].apply(lambda x: halving_dates[halving_dates >= x].min())

df['days_since_last_halving'] = (df['Date'] - df['last_halving']).dt.days
df['days_before_next_halving'] = (df['next_halving'] - df['Date']).dt.days

# [平移] 明天的减半状态
# 逻辑：明天比今天距离下一次减半少1天，距离上一次多1天
df['next_days_since_last'] = df['days_since_last_halving'] + 1
df['next_days_before_next'] = df['days_before_next_halving'] - 1

# 减半周期编码 (Current)
period_halving = 1460.0
df['halving_pos_sin'] = np.sin(2 * np.pi * df['days_since_last_halving'] / period_halving)
df['halving_pos_cos'] = np.cos(2 * np.pi * df['days_since_last_halving'] / period_halving)

# [平移] 减半周期编码 (Next Day)
# 告诉模型：明天我们在四年周期中的位置
df['next_halving_pos_sin'] = np.sin(2 * np.pi * df['next_days_since_last'] / period_halving)
df['next_halving_pos_cos'] = np.cos(2 * np.pi * df['next_days_since_last'] / period_halving)

# 倒计时特征
df['halving_countdown_30'] = (df['days_before_next_halving'] <= 30).astype(int)


# 多尺度月份编码 (保留你的逻辑)
month_scales = [3, 6]
for p in month_scales:
    # Current
    df[f'month_sin_{p}'], df[f'month_cos_{p}'] = zip(*df['month'].apply(lambda x: (np.sin(2*np.pi*(x-1)/p), np.cos(2*np.pi*(x-1)/p))))
    # Next (平移)
    df[f'next_month_sin_{p}'], df[f'next_month_cos_{p}'] = zip(*df['next_month'].apply(lambda x: (np.sin(2*np.pi*(x-1)/p), np.cos(2*np.pi*(x-1)/p))))

# ==========================================
# 4. 高级/实验性特征
# ==========================================
# Atan2 角度特征 (保留你的原始逻辑，这些用于捕获相位)
df['wd_angle'] = np.arctan2(df['wd_sin'], df['wd_cos'])
df['yr_angle'] = np.arctan2(df['day_sin'], df['day_cos'])
df['mn_angle'] = np.arctan2(df['month_sin'], df['month_cos'])
df['daymn_angle'] = np.arctan2(df['day_of_month_sin'], df['day_of_month_cos'])
df['next_daymn_angle'] = np.arctan2(df['next_day_of_month_sin'], df['next_day_of_month_cos'])
# Next Day Angles
df['next_wd_angle'] = np.arctan2(df['next_wd_sin'], df['next_wd_cos'])
df['next_yr_angle'] = np.arctan2(df['next_day_sin'], df['next_day_cos'])
df['next_mn_angle'] = np.arctan2(df['next_month_sin'], df['next_month_cos'])
# others
df['halving_pos_angle'] = np.arctan2(df['halving_pos_sin'], df['halving_pos_cos'])
df['next_halving_pos_angle'] = np.arctan2(df['next_halving_pos_sin'], df['next_halving_pos_cos'])
df['month_3_angle'] = np.arctan2(df['month_sin_3'], df['month_cos_3'])
df['next_month_3_angle'] = np.arctan2(df['next_month_sin_3'], df['next_month_cos_3'])
df['month_6_angle'] = np.arctan2(df['month_sin_6'], df['month_cos_6'])
df['next_month_6_angle'] = np.arctan2(df['next_month_sin_6'], df['next_month_cos_6'])
# 实验：归一化
df['next_yr_angle_norm'] = np.arctan2(df['next_day_sin'], df['next_day_cos']) / np.pi
df['yr_angle_norm'] = np.arctan2(df['day_sin'], df['day_cos']) / np.pi


# ==========================================
# Clean up
# ==========================================
# Drop helper columns if needed, but usually keeping them is fine for debug.
# Dropping the date object columns created for calculation to keep CSV clean-ish
cols_to_drop = ['last_halving', 'next_halving'] 
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Save
df.to_csv(args.filename, index=False)

print(f"File {args.filename} updated successfully.")
print("Added features include: next_is_weekend, next_wd_sin/cos, next_day_sin/cos, next_is_month_start, etc.")