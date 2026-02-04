import pandas as pd
import numpy as np
import argparse
import sys
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import warnings

# 忽略计算中的一些警告
warnings.filterwarnings('ignore')

def process_auto_cross(input_file, output_file):
    print(f"Reading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        sys.exit(1)

    # ==========================================
    # 1. 配置与固定特征保护
    # ==========================================
    # 这些特征必须原样保留在输出中
    fixed_features = [
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    # 检查固定特征是否存在
    missing_fixed = [c for c in fixed_features if c not in df.columns]
    if missing_fixed:
        print(f"Warning: Some fixed features are missing: {missing_fixed}")

    # ==========================================
    # 2. 构建代理目标 (Proxy Target)
    # ==========================================
    # 为了评估哪些交叉特征“有意义”，我们需要一个预测目标。
    # 对于时序预测，最通用的目标是：下一日的对数收益率 (Next Day Log Return)
    print("Constructing proxy target for feature evaluation...")
    
    # 使用 Close 计算收益率
    if 'Close' not in df.columns:
        raise ValueError("Critical Error: 'Close' column missing, cannot determine feature importance.")
        
    # 计算 Target: T+1 的收益率
    # 逻辑：如果一个特征能跟明天的涨跌有高互信息，它就是好特征
    target_series = np.log(df['Close']).diff().shift(-1)
    
    # ==========================================
    # 3. 数据清洗与准备
    # ==========================================
    # 移除无法用于计算的列（非数值列）
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    
    # 移除由 target 产生的 NaN 行 (最后一行) 以及数据本身的 NaN
    # 我们只用这个清洗后的数据来“评估重要性”，不影响最终输出的行数
    valid_indices = target_series.notna() & numeric_df.notna().all(axis=1)
    X_eval = numeric_df.loc[valid_indices]
    y_eval = target_series.loc[valid_indices]

    # 标准化数据用于评估 (互信息计算对尺度不敏感，但相关性敏感)
    scaler = StandardScaler()
    X_eval_scaled = pd.DataFrame(scaler.fit_transform(X_eval), columns=X_eval.columns, index=X_eval.index)

    # ==========================================
    # 4. 第一轮筛选：寻找最强的“种子”特征
    # ==========================================
    print("Phase 1: Selecting top 'Seed' features based on Information Correlation...")
    
    # 使用简单的皮尔逊相关性的绝对值作为速度最快的筛选器
    # (也可以用 mutual_info_regression，但对于几百列数据会很慢，这里用 correlation 加速)
    correlations = X_eval_scaled.corrwith(y_eval).abs()
    
    # 排除一些明显会导致数据泄露或无意义的列 (如 Date 转换来的数值)
    exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close'] 
    # Open/High/Low/Close 绝对值通常不直接用于交叉，用它们的衍生指标（如 vol, rsi）交叉更有意义
    
    candidates = correlations.drop([c for c in exclude_cols if c in correlations.index], errors='ignore')
    
    # 选取 Top N 个种子特征 (例如前 30 个)
    TOP_N_SEEDS = 50
    top_seeds = candidates.nlargest(TOP_N_SEEDS).index.tolist()
    
    print(f"Top {len(top_seeds)} seed features selected: {top_seeds[:5]} ...")

    # ==========================================
    # 5. 生成交叉特征 (Cross Generation)
    # ==========================================
    print("Phase 2: Generating candidate cross features...")
    
    # 显式生成两两乘积: FeatureA_x_FeatureB
    # 这模拟了 DCN 中 x_0 * x_l 的操作
    cross_candidates = pd.DataFrame(index=X_eval.index)
    
    interaction_names = []
    
    # 遍历所有组合
    for feat_a, feat_b in combinations(top_seeds, 2):
        col_name = f"CROSS_{feat_a}_x_{feat_b}"
        # 计算交叉项
        cross_candidates[col_name] = X_eval_scaled[feat_a] * X_eval_scaled[feat_b]
        interaction_names.append(col_name)

    print(f"Generated {len(interaction_names)} candidate interactions.")

    # ==========================================
    # 6. 第二轮筛选：寻找最有意义的交叉
    # ==========================================
    print("Phase 3: Filtering for the most predictive interactions...")
    
    # 再次计算新生成的交叉特征与目标的相关性
    cross_correlations = cross_candidates.corrwith(y_eval).abs()
    
    # 选取 Top K 个最佳交叉特征 (例如前 50 个)
    TOP_K_CROSS = 120
    best_cross_features = cross_correlations.nlargest(TOP_K_CROSS).index.tolist()
    
    print(f"Selected top {len(best_cross_features)} cross features.")
    
    # ==========================================
    # 7. 生成最终数据
    # ==========================================
    print("Constructing final dataset...")
    
    # 复制原始数据
    df_final = df.copy()
    
    # 将选中的交叉特征计算并添加到最终 DataFrame
    # 注意：这里我们使用原始数据(非标准化)进行相乘，或者根据你的需求使用标准化数据
    # 为了保持物理意义（如 vol * rsi），通常直接相乘即可，
    # 但为了数值稳定性，建议先对原始列做简单的中心化再相乘，这里为了脚本通用性，直接相乘。
    # 
    # *重要优化*：如果直接相乘数值会很大，建议对这两个列先做 robust scale。
    # 但考虑到 Mamba 有 RevIN，我们直接相乘，依靠 RevIN 处理幅度。
    
    for col_name in best_cross_features:
        # 解析列名获取原始特征名 "CROSS_A_x_B" -> A, B
        parts = col_name.replace("CROSS_", "").split("_x_")
        feat_a = parts[0]
        feat_b = parts[1]
        
        # 计算并添加 (处理 NaN)
        # 使用 ffill 填充原始数据中的空洞，防止交叉结果为 NaN
        series_a = df[feat_a].ffill().fillna(0)
        series_b = df[feat_b].ffill().fillna(0)
        
        # 核心逻辑：交叉 = A * B
        df_final[col_name] = series_a * series_b

    # 再次确保固定特征都在
    # (如果features.csv里本来就没有这些列，这里不会报错，只是确保如果在里面就不被丢弃)
    # 当前逻辑是 append 模式，所以原始列都在。

    # 处理可能的 Inf / NaN
    df_final.replace([np.inf, -np.inf], 0, inplace=True)
    df_final.fillna(0, inplace=True)

    if output_file is None:
        output_path = input_file
        # output_file = input_file.replace('.csv', '_with_auto_cross.csv')

    print(f"Total columns: {len(df_final.columns)}")
    print(f"Saving to {output_file}...")
    df_final.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automatically generate meaningful cross features based on predictive power.')
    parser.add_argument('--filename', type=str, required=True, help='Input CSV file containing raw features')
    args = parser.parse_args()

    process_auto_cross(args.filename, None)