"""
问题1：定量分析自建点与国控点在PM2.5上的差异
计算指标：MAE、RMSE
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR


def calculate_mae(y_true, y_pred):
    """计算平均绝对误差 (MAE)"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))


def calculate_rmse(y_true, y_pred):
    """计算均方根误差 (RMSE)"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))


def calculate_r2(y_true, y_pred):
    """计算决定系数 R²"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t = y_true[mask]
    y_p = y_pred[mask]
    
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)


def calculate_mape(y_true, y_pred):
    """计算平均绝对百分比误差 (MAPE)"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | (y_true == 0))
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def analyze_pm25_difference(hourly_df):
    """
    分析自建点与国控点在PM2.5上的差异
    """
    print("=" * 70)
    print("问题1：自建点与国控点PM2.5差异分析")
    print("=" * 70)
    
    # 国控点PM2.5真实值
    y_pm25 = hourly_df["y_pm25"].values
    
    # 自建点PM2.5特征（两种：最近点和窗口均值）
    x_pm25_near = hourly_df["x_pm25_near"].values  # 最近点
    x_pm25_mean = hourly_df["x_pm25_mean"].values  # 窗口均值
    
    # 有效样本数
    valid_near = ~(np.isnan(y_pm25) | np.isnan(x_pm25_near))
    valid_mean = ~(np.isnan(y_pm25) | np.isnan(x_pm25_mean))
    
    print(f"\n数据概况:")
    print(f"  总小时数: {len(hourly_df)}")
    print(f"  最近点有效配对数: {valid_near.sum()}")
    print(f"  窗口均值有效配对数: {valid_mean.sum()}")
    
    print(f"\n" + "-" * 70)
    print("1. 使用最近点特征 (x_pm25_near) 的差异分析")
    print("-" * 70)
    
    # 基本统计
    print(f"\n基本统计:")
    print(f"  国控点PM2.5均值: {np.nanmean(y_pm25):.2f} μg/m³")
    print(f"  自建点最近点PM2.5均值: {np.nanmean(x_pm25_near):.2f} μg/m³")
    print(f"  差异均值 (自建点 - 国控点): {np.nanmean(x_pm25_near - y_pm25):.2f} μg/m³")
    
    # 计算误差指标
    mae_near = calculate_mae(y_pm25, x_pm25_near)
    rmse_near = calculate_rmse(y_pm25, x_pm25_near)
    r2_near = calculate_r2(y_pm25, x_pm25_near)
    mape_near = calculate_mape(y_pm25, x_pm25_near)
    
    print(f"\n误差指标:")
    print(f"  MAE (平均绝对误差): {mae_near:.2f} μg/m³")
    print(f"  RMSE (均方根误差): {rmse_near:.2f} μg/m³")
    print(f"  R² (决定系数): {r2_near:.4f}")
    print(f"  MAPE (平均绝对百分比误差): {mape_near:.2f}%")
    
    print(f"\n" + "-" * 70)
    print("2. 使用窗口均值特征 (x_pm25_mean) 的差异分析")
    print("-" * 70)
    
    # 基本统计
    print(f"\n基本统计:")
    print(f"  国控点PM2.5均值: {np.nanmean(y_pm25):.2f} μg/m³")
    print(f"  自建点窗口均值PM2.5均值: {np.nanmean(x_pm25_mean):.2f} μg/m³")
    print(f"  差异均值 (自建点 - 国控点): {np.nanmean(x_pm25_mean - y_pm25):.2f} μg/m³")
    
    # 计算误差指标
    mae_mean = calculate_mae(y_pm25, x_pm25_mean)
    rmse_mean = calculate_rmse(y_pm25, x_pm25_mean)
    r2_mean = calculate_r2(y_pm25, x_pm25_mean)
    mape_mean = calculate_mape(y_pm25, x_pm25_mean)
    
    print(f"\n误差指标:")
    print(f"  MAE (平均绝对误差): {mae_mean:.2f} μg/m³")
    print(f"  RMSE (均方根误差): {rmse_mean:.2f} μg/m³")
    print(f"  R² (决定系数): {r2_mean:.4f}")
    print(f"  MAPE (平均绝对百分比误差): {mape_mean:.2f}%")
    
    print(f"\n" + "-" * 70)
    print("3. 综合对比")
    print("-" * 70)
    
    print(f"\n指标对比汇总表:")
    print(f"{'指标':<25} {'最近点特征':<20} {'窗口均值特征':<20}")
    print("-" * 65)
    print(f"{'MAE (μg/m³)':<25} {mae_near:<20.2f} {mae_mean:<20.2f}")
    print(f"{'RMSE (μg/m³)':<25} {rmse_near:<20.2f} {rmse_mean:<20.2f}")
    print(f"{'R²':<25} {r2_near:<20.4f} {r2_mean:<20.4f}")
    print(f"{'MAPE (%)':<25} {mape_near:<20.2f} {mape_mean:<20.2f}")
    
    print(f"\n结论:")
    if mae_near < mae_mean:
        print(f"  - 最近点特征的MAE ({mae_near:.2f}) 优于 窗口均值特征 ({mae_mean:.2f})")
    else:
        print(f"  - 窗口均值特征的MAE ({mae_mean:.2f}) 优于 最近点特征 ({mae_near:.2f})")
    
    if rmse_near < rmse_mean:
        print(f"  - 最近点特征的RMSE ({rmse_near:.2f}) 优于 窗口均值特征 ({rmse_mean:.2f})")
    else:
        print(f"  - 窗口均值特征的RMSE ({rmse_mean:.2f}) 优于 最近点特征 ({rmse_near:.2f})")
    
    print(f"\n  R² 均接近 {max(r2_near, r2_mean):.4f}，说明自建点与国控点")
    print(f"  PM2.5 存在较强的线性相关关系，但存在系统性偏差。")
    
    return {
        "nearest": {"mae": mae_near, "rmse": rmse_near, "r2": r2_near, "mape": mape_near},
        "mean": {"mae": mae_mean, "rmse": rmse_mean, "r2": r2_mean, "mape": mape_mean},
    }


def analyze_pm25_by_range(hourly_df):
    """
    按PM2.5浓度区间分段分析
    """
    print(f"\n" + "=" * 70)
    print("4. 按PM2.5浓度区间分段分析")
    print("=" * 70)
    
    y_pm25 = hourly_df["y_pm25"].values
    x_pm25_near = hourly_df["x_pm25_near"].values
    
    # 定义浓度区间
    ranges = [
        ("优 (0-35)", 0, 35),
        ("良 (35-75)", 35, 75),
        ("轻度污染 (75-115)", 75, 115),
        ("中度污染 (115-150)", 115, 150),
        ("重度及以上 (>150)", 150, 9999),
    ]
    
    print(f"\n{'浓度区间':<25} {'样本数':<10} {'MAE':<12} {'RMSE':<12} {'R²':<10}")
    print("-" * 70)
    
    for name, low, high in ranges:
        mask = (y_pm25 >= low) & (y_pm25 < high)
        mask = mask & ~np.isnan(x_pm25_near)
        
        if mask.sum() > 0:
            y_sub = y_pm25[mask]
            x_sub = x_pm25_near[mask]
            
            mae = calculate_mae(y_sub, x_sub)
            rmse = calculate_rmse(y_sub, x_sub)
            r2 = calculate_r2(y_sub, x_sub)
            
            print(f"{name:<25} {mask.sum():<10} {mae:<12.2f} {rmse:<12.2f} {r2:<10.4f}")


def main():
    """主函数"""
    print("\n加载小时级特征数据...")
    hourly_df = pd.read_parquet(PROCESSED_DATA_DIR / "hourly_merged.parquet")
    print(f"已加载 {len(hourly_df)} 条记录")
    print(f"时间范围: {hourly_df['time'].min()} 至 {hourly_df['time'].max()}")
    
    # 执行分析
    results = analyze_pm25_difference(hourly_df)
    
    # 分段分析
    analyze_pm25_by_range(hourly_df)
    
    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
