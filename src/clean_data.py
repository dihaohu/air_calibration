"""
数据清洗模块：
1. 两层去重
2. 类型转换与异常字符处理
3. 物理异常过滤（硬阈值）
4. 缺失值处理（分数据集）
"""

import pandas as pd
import numpy as np
from .config import (
    REFERENCE_POLLUTANT_COLS,
    SELFBUILD_NUMERIC_COLS,
    PHYSICAL_THRESHOLDS,
)


def deduplicate_records(df, name="数据集"):
    """
    两层去重：
    1. 全字段完全重复的行删除
    2. 同一时间戳多条记录，取数值列均值
    """
    print(f"\n=== [{name}] 两层去重 ===")
    original_count = len(df)
    
    # 第一层：完全重复的行（全字段完全相同）
    df_no_dup = df.drop_duplicates()
    dup_count_1 = original_count - len(df_no_dup)
    print(f"  第一层（完全重复）: 删除 {dup_count_1} 行, 剩余 {len(df_no_dup)} 行")
    
    # 第二层：同一时间戳多条记录，取均值
    numeric_cols = [c for c in df.columns if c != "time"]
    df_grouped = df_no_dup.groupby("time", as_index=False)[numeric_cols].mean()
    
    # 重新附加时间列（groupby后时间列会被保留）
    dup_count_2 = len(df_no_dup) - len(df_grouped)
    print(f"  第二层（同时间戳合并）: 合并 {dup_count_2} 条重复记录, 剩余 {len(df_grouped)} 行")
    
    return df_grouped


def convert_and_clean_types(df, numeric_cols, name="数据集"):
    """
    类型转换：将数值列强制转为浮点型，无法转换的置为缺失值
    """
    print(f"\n=== [{name}] 类型转换 ===")
    
    for col in numeric_cols:
        if col in df.columns:
            before_count = df[col].notna().sum()
            
            # 尝试转换为数值，无法转换的变为NaN
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
            after_count = df[col].notna().sum()
            if after_count < before_count:
                lost = before_count - after_count
                print(f"  {col}: 转换后损失 {lost} 条非空记录")
    
    return df


def apply_physical_filters(df, thresholds, name="数据集"):
    """
    物理异常过滤：只删明显不可能的值（< 0 或超出合理范围）
    保留真实污染峰值，不过度清洗
    """
    print(f"\n=== [{name}] 物理异常过滤 ===")
    
    # 需要过滤的列（气象列）
    weather_cols = ["wind", "rain", "rh", "temp", "pressure"]
    filter_cols = [c for c in weather_cols if c in df.columns]
    
    # 污染物列也检查负值
    pollutant_cols = ["pm25", "pm10", "co", "no2", "so2", "o3"]
    for col in pollutant_cols:
        if col in df.columns:
            filter_cols.append(col)
    
    total_removed = 0
    for col in filter_cols:
        if col not in thresholds:
            continue
            
        thresh = thresholds[col]
        
        # 下界过滤（< min）
        below_mask = df[col] < thresh["min"]
        below_count = below_mask.sum()
        
        # 上界过滤（> max）
        above_mask = df[col] > thresh["max"]
        above_count = above_mask.sum()
        
        # 删除超出范围的记录
        invalid_mask = below_mask | above_mask
        if invalid_mask.sum() > 0:
            df = df[~invalid_mask]
            total_removed += invalid_mask.sum()
            if below_count > 0:
                print(f"  {col}: 删除 {below_count} 条 < {thresh['min']} 的记录")
            if above_count > 0:
                print(f"  {col}: 删除 {above_count} 条 > {thresh['max']} 的记录")
    
    print(f"  共删除 {total_removed} 条异常记录, 剩余 {len(df)} 行")
    
    return df


def clean_reference_data(df):
    """
    国控点数据清洗流程：
    1. 两层去重
    2. 类型转换
    3. 物理异常过滤
    4. 缺失值处理：六项污染物任一缺失则整点丢弃
    """
    print("\n" + "="*60)
    print("国控点数据清洗流程")
    print("="*60)
    
    # 1. 去重
    df = deduplicate_records(df, "国控点")
    
    # 2. 类型转换（国控点只有污染物列）
    df = convert_and_clean_types(df, REFERENCE_POLLUTANT_COLS, "国控点")
    
    # 3. 物理异常过滤
    df = apply_physical_filters(df, PHYSICAL_THRESHOLDS, "国控点")
    
    # 4. 缺失值处理：六项污染物任一缺失则整点丢弃
    print(f"\n=== [国控点] 缺失值处理 ===")
    before_count = len(df)
    df = df.dropna(subset=REFERENCE_POLLUTANT_COLS)
    removed = before_count - len(df)
    print(f"  因污染物缺失丢弃: {removed} 行, 剩余 {len(df)} 行")
    
    print(f"\n[国控点] 清洗完成: {len(df)} 行")
    
    return df


def clean_selfbuild_data(df):
    """
    自建点数据清洗流程：
    1. 两层去重
    2. 类型转换
    3. 物理异常过滤（不过度激进）
    4. 缺失值：先保留，后续在小时窗口内处理
    """
    print("\n" + "="*60)
    print("自建点数据清洗流程")
    print("="*60)
    
    # 1. 去重
    df = deduplicate_records(df, "自建点")
    
    # 2. 类型转换
    df = convert_and_clean_types(df, SELFBUILD_NUMERIC_COLS, "自建点")
    
    # 3. 物理异常过滤
    df = apply_physical_filters(df, PHYSICAL_THRESHOLDS, "自建点")
    
    # 4. 缺失值暂不处理，后续在小时窗口内统计时处理
    print(f"\n=== [自建点] 缺失值处理 ===")
    missing_counts = df[SELFBUILD_NUMERIC_COLS].isnull().sum()
    print("  各列缺失值数量（暂不处理，将在小时窗口统计时处理）:")
    print(missing_counts[missing_counts > 0])
    
    print(f"\n[自建点] 清洗完成: {len(df)} 行")
    
    return df


def save_cleaned_data(reference_df, selfbuild_df, 
                     reference_path, selfbuild_path):
    """保存清洗后的数据"""
    print("\n" + "="*60)
    print("保存清洗后数据")
    print("="*60)
    
    reference_df.to_parquet(reference_path, index=False)
    print(f"  国控点数据已保存: {reference_path}")
    
    selfbuild_df.to_parquet(selfbuild_path, index=False)
    print(f"  自建点数据已保存: {selfbuild_path}")
