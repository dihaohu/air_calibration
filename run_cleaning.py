"""
数据清洗主程序
按顺序执行：加载 -> 清洗 -> 对齐 -> 构建标准特征表 -> 划分数据集
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.load_data import load_reference_data, load_selfbuild_data, get_basic_stats
from src.clean_data import clean_reference_data, clean_selfbuild_data, save_cleaned_data
from src.align_hourly import (
    align_and_build_hourly_samples,
    reorder_columns,
    handle_missing_features,
    save_hourly_data,
    print_feature_table_summary,
)
from src.config import (
    PROCESSED_DATA_DIR,
    REFERENCE_CLEAN_FILE,
    SELFBUILD_CLEAN_FILE,
    HOURLY_MERGED_FILE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)


def split_dataset(hourly_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    按时间顺序划分训练集、验证集、测试集
    """
    print("\n" + "="*60)
    print("数据集划分（按时间顺序）")
    print("="*60)
    
    n = len(hourly_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = hourly_df.iloc[:train_end].copy()
    val_df = hourly_df.iloc[train_end:val_end].copy()
    test_df = hourly_df.iloc[val_end:].copy()
    
    print(f"总样本数: {n}")
    print(f"训练集: {len(train_df)} 行 ({len(train_df)/n*100:.1f}%)")
    print(f"  时间范围: {train_df['time'].min()} 至 {train_df['time'].max()}")
    print(f"验证集: {len(val_df)} 行 ({len(val_df)/n*100:.1f}%)")
    print(f"  时间范围: {val_df['time'].min()} 至 {val_df['time'].max()}")
    print(f"测试集: {len(test_df)} 行 ({len(test_df)/n*100:.1f}%)")
    print(f"  时间范围: {test_df['time'].min()} 至 {test_df['time'].max()}")
    
    return train_df, val_df, test_df


def main():
    """主函数：执行完整数据清洗与特征构建流程"""
    print("="*60)
    print("空气质量数据清洗与特征构建流程")
    print("="*60)
    
    # 确保输出目录存在
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载数据
    print("\n" + "="*60)
    print("Step 1: 加载原始数据")
    print("="*60)
    
    reference_df = load_reference_data()
    selfbuild_df = load_selfbuild_data()
    
    # 2. 数据清洗
    print("\n" + "="*60)
    print("Step 2: 数据清洗")
    print("="*60)
    
    reference_clean = clean_reference_data(reference_df)
    selfbuild_clean = clean_selfbuild_data(selfbuild_df)
    
    # 保存清洗后数据
    save_cleaned_data(reference_clean, selfbuild_clean,
                     REFERENCE_CLEAN_FILE, SELFBUILD_CLEAN_FILE)
    
    # 3. 时间对齐与标准特征构建
    print("\n" + "="*60)
    print("Step 3: 时间对齐与标准特征构建")
    print("="*60)
    
    hourly_df = align_and_build_hourly_samples(reference_clean, selfbuild_clean)
    
    # 4. 重排列顺序
    hourly_df = reorder_columns(hourly_df)
    
    # 5. 缺失值处理
    hourly_df = handle_missing_features(hourly_df)
    
    # 6. 保存小时级特征表
    save_hourly_data(hourly_df, HOURLY_MERGED_FILE)
    
    # 7. 打印汇总信息
    print_feature_table_summary(hourly_df)
    
    # 8. 数据集划分
    train_df, val_df, test_df = split_dataset(
        hourly_df, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )
    
    # 保存划分后的数据集
    train_df.to_parquet(PROCESSED_DATA_DIR / "train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DATA_DIR / "val.parquet", index=False)
    test_df.to_parquet(PROCESSED_DATA_DIR / "test.parquet", index=False)
    print(f"\n已保存划分后的数据集至 {PROCESSED_DATA_DIR}/")
    
    print("\n" + "="*60)
    print("数据清洗与特征构建流程完成！")
    print("="*60)
    
    return hourly_df, train_df, val_df, test_df


if __name__ == "__main__":
    hourly_df, train_df, val_df, test_df = main()