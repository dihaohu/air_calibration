"""
数据加载模块：读取CSV/Excel，统一列名，转datetime
"""

import pandas as pd
import numpy as np
from .config import (
    REFERENCE_FILE, TOBECALIBRATED_FILE,
    REFERENCE_COLS, SELFBUILD_COLS
)


def load_reference_data():
    """
    加载国控点数据，统一列名，转换为datetime
    """
    print("加载国控点数据...")
    df = pd.read_excel(REFERENCE_FILE)
    
    # 重命名列
    df = df.rename(columns=REFERENCE_COLS)
    
    # 转换时间列为datetime
    df["time"] = pd.to_datetime(df["time"])
    
    # 按时间升序排序
    df = df.sort_values("time").reset_index(drop=True)
    
    print(f"  原始数据量: {len(df)} 行")
    print(f"  时间范围: {df['time'].min()} 至 {df['time'].max()}")
    print(f"  列: {df.columns.tolist()}")
    
    return df


def load_selfbuild_data():
    """
    加载自建点数据，统一列名，转换为datetime
    """
    print("加载自建点数据...")
    df = pd.read_excel(TOBECALIBRATED_FILE)
    
    # 重命名列
    df = df.rename(columns=SELFBUILD_COLS)
    
    # 转换时间列为datetime
    df["time"] = pd.to_datetime(df["time"])
    
    # 按时间升序排序
    df = df.sort_values("time").reset_index(drop=True)
    
    print(f"  原始数据量: {len(df)} 行")
    print(f"  时间范围: {df['time'].min()} 至 {df['time'].max()}")
    print(f"  列: {df.columns.tolist()}")
    
    return df


def get_basic_stats(df, name):
    """打印DataFrame的基本统计信息"""
    print(f"\n=== {name} 基本统计 ===")
    print(f"数据量: {len(df)} 行")
    print(f"时间范围: {df['time'].min()} 至 {df['time'].max()}")
    print(f"\n缺失值统计:")
    print(df.isnull().sum())
    print(f"\n数值列统计:")
    print(df.describe())
