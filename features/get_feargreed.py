import argparse
import pandas as pd
import numpy as np
import fear_and_greed # pip install fear-and-greed-crypto
from fear_and_greed import FearAndGreedIndex
from datetime import datetime, timedelta

# pip install fear-and-greed-crypto

def get_fng_data(start_date, end_date):
    """
    获取 Fear and Greed 数据并整理为 DataFrame
    """
    print(f"正在获取 F&G 数据: {start_date} 至 {end_date} ...")
    
    # 转换为 fear_and_greed 库需要的 datetime 格式
    # 注意：API获取通常需要一点余量，防止时区差异导致缺数据
    fetch_start = pd.to_datetime(start_date) - timedelta(days=5)
    fetch_end = pd.to_datetime(end_date)
    
    # 获取数据
    # historical_data = fear_and_greed.get().get_all()
    # 初始化客户端
    fng = FearAndGreedIndex()
    
    # 获取历史数据 (返回 list of dicts)
    try:
        historical_data = fng.get_historical_data(fetch_start, fetch_end)
    except Exception as e:
        raise Exception(f"Failed to fetch historical data: {e}")
    
    if not historical_data:
        raise ValueError("No historical data returned for the specified range. Note: Data starts from Feb 2018.")
    
    # 构建 DataFrame
    fng_df = pd.DataFrame(historical_data)
    
    # 格式化
    # F&G API 返回的 timestamp 是 string 格式的 unix time
    fng_df['timestamp'] = pd.to_datetime(fng_df['timestamp'].astype(float), unit='s')
    fng_df['fng_value'] = pd.to_numeric(fng_df['value'], errors='coerce')
    
    # 只保留日期部分用于合并 (UTC 0点)
    fng_df['Date'] = fng_df['timestamp'].dt.date
    fng_df = fng_df[['Date', 'fng_value']].sort_values('Date')
    
    # 去重，防止同一天有多条数据
    fng_df = fng_df.drop_duplicates(subset=['Date'], keep='last')
    
    return fng_df

def calculate_advanced_features(df):
    """
    计算高级情绪特征：滞后、背离、融合
    """
    print("正在计算高级特征...")
    
    # --- 0. 预处理 ---
    # 确保价格数据是数值型
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 计算价格的日收益率 (Log Return)，用于后续计算相关性
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 归一化 F&G (0-1)，神经网络更喜欢小数值
    df['fng_norm'] = df['fng_value'] / 100.0

    # --- 1. 滞后特征 (Lag Features) ---
    # Mamba 需要历史上下文。
    # "昨天的恐慌"对"今天的价格"影响可能比"今天的恐慌"（还没发生完）更大
    for lag in [1, 2, 3]:
        df[f'fng_lag_{lag}'] = df['fng_norm'].shift(lag)

    # --- 2. 动量与波动 (Momentum & Volatility) ---
    # 情绪的变化方向
    df['fng_diff'] = df['fng_norm'].diff()
    # 情绪变化的加速度 (二阶差分)
    df['fng_accel'] = df['fng_diff'].diff()
    # 情绪波动率 (7天标准差): 市场分歧越大，变盘概率越大
    df['fng_volatility_7d'] = df['fng_norm'].rolling(window=7).std()

    # --- 3. 极端状态标记 (Regime Flags) ---
    df['is_extreme_fear'] = (df['fng_value'] <= 20).astype(int)
    df['is_extreme_greed'] = (df['fng_value'] >= 80).astype(int)

    # --- 4. 价格与情绪背离 (Price-Sentiment Divergence) ---
    # 这是一个强力特征。
    # 逻辑：价格在涨 (Return > 0) 但情绪在跌 (Fng_Diff < 0) -> 顶背离风险
    # 逻辑：价格在跌 (Return < 0) 但情绪在涨 (Fng_Diff > 0) -> 底背离机会
    
    # 计算过去 5 天的价格趋势和情绪趋势
    price_trend_5d = df['Close'].diff(5)
    fng_trend_5d = df['fng_norm'].diff(5)
    
    # 背离分数：如果符号相反，乘积为负。数值越大代表背离越严重
    # 为了让特征对模型友好，我们做一个简单的交互
    # 1 = 同向, -1 = 反向 (背离)
    df['divergence_flag'] = np.sign(price_trend_5d) * np.sign(fng_trend_5d)
    
    # 具体的背离强度特征
    # 如果价格创新高(14天最高)但恐慌指数没有创新高 -> 这是一个具体的数值差
    price_rank = df['Close'].rolling(14).rank()
    fng_rank = df['fng_norm'].rolling(14).rank()
    # Rank 差值：如果价格 Rank很高(接近14天高点) 但 Fng Rank很低 -> 巨大的背离
    df['divergence_rank_diff'] = price_rank - fng_rank

    # --- 5. 融合特征 (Fusion Methods) ---
    # 将 F&G 与 OHLC 数据结合
    
    # A. 恐慌加权成交量 (Sentiment-Weighted Volume)
    # 逻辑：在极度恐慌时的巨大成交量通常意味着"恐慌抛售"（Capitulation），往往是底部
    # 我们用 (1 - fng_norm) 代表恐慌度。恐慌度越高，Volume权重越大
    if 'Volume' in df.columns:
        # 为了防止数值过大，先对 Volume 做 Log 处理
        log_vol = np.log1p(df['Volume'])
        df['vol_fear_interaction'] = log_vol * (1 - df['fng_norm'])
    
    # B. 情绪调整后的价格波动 (Sentiment-Adjusted Range)
    # (High - Low) / Close 代表当天的波动率
    # 如果波动率很大且是极度贪婪，往往是顶部派发
    df['daily_range_pct'] = (df['High'] - df['Low']) / df['Close']
    df['fng_range_interaction'] = df['daily_range_pct'] * df['fng_norm']

    return df

def main():
    parser = argparse.ArgumentParser(description="为加密货币数据添加 Fear & Greed 高级特征")
    parser.add_argument("--filename", type=str, required=True, help="输入的CSV文件路径 (例如: ETH_Dataset_daily.csv)")
    args = parser.parse_args()

    # 1. 读取本地数据
    try:
        print(f"正在读取文件: {args.filename}")
        df = pd.read_csv(args.filename)
        
        # 自动识别日期列
        date_col = None
        for col in ['Date', 'date', 'timestamp', 'time']:
            if col in df.columns:
                date_col = col
                break
        
        if not date_col:
            raise ValueError("在CSV中未找到日期列 (Date, date, timestamp, time)")
            
        # 统一转换为 date 对象
        df[date_col] = pd.to_datetime(df[date_col]).dt.date
        df = df.sort_values(date_col)
        
        start_date = df[date_col].min()
        end_date = df[date_col].max()
        print(f"数据范围: {start_date} 到 {end_date}")

    except Exception as e:
        print(f"读取CSV失败: {e}")
        return

    # 2. 获取 F&G 数据
    try:
        fng_df = get_fng_data(start_date, end_date)
    except Exception as e:
        print(f"获取 F&G 数据失败: {e}")
        return

    # 3. 合并数据 (Left Join 保留原始数据的所有行)
    print("正在合并数据...")
    merged_df = pd.merge(df, fng_df, left_on=date_col, right_on='Date', how='left')
    
    # 如果有两个日期列，删掉多余的
    if 'Date' in merged_df.columns and date_col != 'Date':
        merged_df = merged_df.drop(columns=['Date'])

    # 4. 填充缺失值
    # F&G 数据最早开始于 2018年。如果你的数据更早，会有 NaN。
    # 策略：对于早期的 NaN，用 50 (中性) 填充，或者用当年的均值。这里用 50 填充防止报错。
    # 对于中间的缺失（周末/节假日），用 ffill (前向填充)
    # merged_df['fng_value'] = merged_df['fng_value'].fillna(method='ffill').fillna(50)

    # 5. 计算高级特征
    final_df = calculate_advanced_features(merged_df)

    # 6. 清理 NaN (由于 diff 和 rolling 产生的)
    # 对于时序模型，前几行数据因为没有足够的历史做 rolling，通常需要丢弃
    original_len = len(final_df)
    final_df = final_df.dropna()
    print(f"删除了 {original_len - len(final_df)} 行由于计算窗口产生的空数据")

    # 7. 保存文件
    # output_filename = args.filename.replace('.csv', '_with_fng_enhanced.csv')
    output_filename = args.filename
    final_df.to_csv(output_filename, index=False)
    print(f"成功! 处理后的文件已保存为: {output_filename}")
    print("包含的新特征列示例:")
    print([c for c in final_df.columns if 'fng' in c or 'divergence' in c or 'interaction' in c])

if __name__ == "__main__":
    main()