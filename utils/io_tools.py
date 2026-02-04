import pandas as pd
import yaml
import sys
import os

def load_yaml(filepath):
    """
    通用：读取YAML配置文件
    Args:
        filepath (str): yaml文件路径
    Returns:
        dict: 配置字典
    """
    if not os.path.exists(filepath):
        print(f"[Fatal] Config file not found: {filepath}")
        sys.exit(1)
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
            if content is None:
                return {}
            return content
    except Exception as e:
        print(f"[Fatal] Error loading config {filepath}: {e}")
        sys.exit(1)


def save_yaml(data, filepath):
    """保存最优参数回 params.yaml"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"[Output] Best params saved to {filepath}")
    except Exception as e:
        print(f"[Error] Failed to save params: {e}")


def load_data(path, date_col='Date', date_format=None):
    """
    通用：加载CSV数据并处理日期索引
    Args:
        path (str): csv文件路径
        date_col (str): 日期列名，默认为 'Date'
        date_format (str): 日期格式字符串 (例如 '%Y-%m-%d')。如果不传，pandas自动推断。
    Returns:
        pd.DataFrame
    """
    print(f"[Data] Loading CSV from {path}...")
    
    if not os.path.exists(path):
        print(f"[Fatal] CSV file not found: {path}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[Fatal] Failed to read CSV: {e}")
        sys.exit(1)
        
    # 日期处理
    if date_col in df.columns:
        try:
            if date_format:
                df[date_col] = pd.to_datetime(df[date_col], format=date_format)
            else:
                df[date_col] = pd.to_datetime(df[date_col])
            
            # 按日期排序是时序模型的关键
            df = df.sort_values(date_col).reset_index(drop=True)
        except Exception as e:
            print(f"[Fatal] Date parsing failed for column '{date_col}'. Error: {e}")
            sys.exit(1)
    else:
        # 如果是必须有时序的训练，这里应该报错；但作为通用工具，只是警告
        print(f"[Warning] Column '{date_col}' not found. Data not sorted by date.")
        
    return df