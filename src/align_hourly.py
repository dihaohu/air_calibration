"""
时间对齐与小时级特征构建模块：

最终生成"每小时一行"的特征表，字段结构如下：

第一组：标签列（国控点真实值）
  y_pm25, y_pm10, y_co, y_no2, y_so2, y_o3

第二组：自建点最近点特征
  x_pm25_near, x_pm10_near, x_co_near, x_no2_near, x_so2_near, x_o3_near
  wind_near, pressure_near, rain_near, temp_near, rh_near

第三组：自建点窗口统计特征（六个污染物）
  x_pm25_mean, x_pm25_std, x_pm25_min, x_pm25_max, x_pm25_median, x_pm25_slope
  x_pm10_mean, x_pm10_std, ...
  ... (其他污染物同理)

第四组：气象窗口特征
  wind_mean, wind_std
  pressure_mean, pressure_std
  temp_mean, temp_std
  rh_mean, rh_std
  rain_mean, rain_sum, rain_delta

第五组：时间特征
  hour, weekday, month, t_idx
  sin_hour, cos_hour, sin_month, cos_month
"""

import pandas as pd
import numpy as np
from scipy import stats
from .config import (
    TIME_WINDOW_MINUTES,
    MIN_RECORDS_IN_WINDOW,
    SELFBUILD_NUMERIC_COLS,
    SELFBUILD_POLLUTANT_COLS,
    SELFBUILD_WEATHER_COLS,
)


def calculate_slope(y_values, x_values=None):
    """
    计算线性趋势斜率（时间序列的线性拟合斜率）
    
    若有效数据点 < 3，返回 NaN
    若斜率计算失败（如所有值相同），返回 0
    """
    if len(y_values) < 3:
        return np.nan
    
    # x 默认为时间序号
    if x_values is None:
        x_values = np.arange(len(y_values))
    
    # 去除 NaN
    mask = ~np.isnan(y_values)
    y_valid = y_values[mask]
    x_valid = x_values[mask]
    
    if len(y_valid) < 3:
        return np.nan
    
    try:
        # 线性回归斜率
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
        return slope
    except Exception:
        return 0.0


def build_nearest_neighbor_feature(selfbuild_df, target_time):
    """
    为目标时间找到最近的自建点记录（±5分钟内）
    """
    time_diff = np.abs((selfbuild_df["time"] - target_time).dt.total_seconds())
    within_window = time_diff <= 5 * 60  # 5分钟 = 300秒
    
    if not within_window.any():
        return None
    
    nearest_idx = time_diff[within_window].idxmin()
    nearest_record = selfbuild_df.loc[nearest_idx]
    
    return nearest_record


def build_window_statistics(selfbuild_df, target_time, window_minutes,
                          numeric_cols, min_records):
    """
    计算目标时间窗口 [t-30min, t+30min] 内的统计特征
    
    返回特征包括：
    - mean, std, min, max, median, slope（对所有 numeric_cols）
    - rain_sum, rain_delta（降水量特殊统计）
    - record_count
    """
    half_window = pd.Timedelta(minutes=window_minutes)
    window_mask = (
        (selfbuild_df["time"] >= target_time - half_window) &
        (selfbuild_df["time"] <= target_time + half_window)
    )
    window_data = selfbuild_df[window_mask].copy()
    valid_count = len(window_data)
    
    stats = {}
    stats["record_count"] = valid_count
    
    if valid_count == 0:
        # 无数据时，所有统计量设为 NaN
        for col in numeric_cols:
            for stat in ["mean", "std", "min", "max", "median", "slope"]:
                stats[f"{col}_{stat}"] = np.nan
        # 降水特殊统计
        stats["rain_sum"] = np.nan
        stats["rain_delta"] = np.nan
        stats["window_data"] = window_data  # 返回空DataFrame以便后续处理
        return stats, window_data
    
    # 按时间排序
    window_data = window_data.sort_values("time")
    
    for col in numeric_cols:
        col_data = window_data[col].dropna()
        
        # 有效记录数少于阈值，该变量统计特征记为缺失
        if len(col_data) < min_records:
            for stat in ["mean", "std", "min", "max", "median", "slope"]:
                stats[f"{col}_{stat}"] = np.nan
        else:
            # 基本统计量
            stats[f"{col}_mean"] = col_data.mean()
            stats[f"{col}_std"] = col_data.std()
            stats[f"{col}_min"] = col_data.min()
            stats[f"{col}_max"] = col_data.max()
            stats[f"{col}_median"] = col_data.median()
            
            # 斜率：计算时间序列的线性趋势
            values = col_data.values
            stats[f"{col}_slope"] = calculate_slope(values)
    
    # 降水特殊统计量
    if "rain" in window_data.columns:
        rain_valid = window_data["rain"].dropna()
        if len(rain_valid) >= min_records:
            stats["rain_sum"] = rain_valid.sum()
            stats["rain_delta"] = rain_valid.max() - rain_valid.min()
        else:
            stats["rain_sum"] = np.nan
            stats["rain_delta"] = np.nan
    else:
        stats["rain_sum"] = np.nan
        stats["rain_delta"] = np.nan
    
    stats["window_data"] = window_data  # 返回窗口数据供后续使用
    return stats, window_data


def add_weather_window_features(window_stats, window_data):
    """
    为气象变量添加额外的窗口特征
    - rain_sum: 窗口内降水量总和
    - rain_delta: 窗口内降水量变化（max - min）
    """
    # rain_sum 和 rain_delta 已在 build_window_statistics 中计算
    # 这里确保它们存在
    if "rain_sum" not in window_stats:
        if len(window_data) > 0 and "rain" in window_data.columns:
            rain_valid = window_data["rain"].dropna()
            if len(rain_valid) > 0:
                window_stats["rain_sum"] = rain_valid.sum()
                window_stats["rain_delta"] = rain_valid.max() - rain_valid.min()
            else:
                window_stats["rain_sum"] = np.nan
                window_stats["rain_delta"] = np.nan
        else:
            window_stats["rain_sum"] = np.nan
            window_stats["rain_delta"] = np.nan
    
    return window_stats


def add_time_features(df, time_col="time"):
    """
    添加时间特征
    - hour: 0-23
    - weekday: 0-6 (周一=0, 周日=6)
    - month: 1-12
    - t_idx: 从起点开始累计的小时序号
    - sin_hour, cos_hour: 日周期编码
    - sin_month, cos_month: 年周期编码
    """
    df = df.copy()
    
    # 基本时间特征
    df["hour"] = df[time_col].dt.hour
    df["weekday"] = df[time_col].dt.dayofweek
    df["month"] = df[time_col].dt.month
    
    # 累计小时序号（从数据集起始时间开始）
    min_time = df[time_col].min()
    df["t_idx"] = ((df[time_col] - min_time).dt.total_seconds() / 3600).astype(int)
    
    # 日周期编码（sin/cos）
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    # 年周期编码（sin/cos，以月为单位）
    # 将月份映射到 [0, 12) 范围
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    
    return df


def align_and_build_hourly_samples(reference_df, selfbuild_df):
    """
    将自建点高频数据与国控点每小时数据对齐，构建标准化小时级样本表
    
    Returns:
        hourly_df: 小时级样本表，标准列名格式
    """
    print("\n" + "="*60)
    print("时间对齐与小时级特征构建（标准列名）")
    print("="*60)
    
    # 列名映射：从旧名（前缀 ws_/nn_）到新名（标准名）
    pollutant_rename = {
        "pm25": "pm25",
        "pm10": "pm10",
        "co": "co",
        "no2": "no2",
        "so2": "so2",
        "o3": "o3",
    }
    
    # 用于存储结果
    hourly_records = []
    total_hours = len(reference_df)
    print(f"共需处理 {total_hours} 个整点...")
    
    for i, row in reference_df.iterrows():
        target_time = row["time"]
        
        # 1. 最近点特征（±5分钟）
        nearest_record = build_nearest_neighbor_feature(selfbuild_df, target_time)
        
        # 2. 窗口统计特征（±30分钟）
        window_stats, window_data = build_window_statistics(
            selfbuild_df, target_time, TIME_WINDOW_MINUTES, 
            SELFBUILD_NUMERIC_COLS, MIN_RECORDS_IN_WINDOW
        )
        
        # 构建记录
        record = {
            "time": target_time,
        }
        
        # === 第一组：标签列（国控点真实值） ===
        record["y_pm25"] = row["pm25"]
        record["y_pm10"] = row["pm10"]
        record["y_co"] = row["co"]
        record["y_no2"] = row["no2"]
        record["y_so2"] = row["so2"]
        record["y_o3"] = row["o3"]
        
        # === 第二组：最近点特征 ===
        if nearest_record is not None:
            for col in SELFBUILD_POLLUTANT_COLS:
                record[f"x_{col}_near"] = nearest_record[col]
            for col in SELFBUILD_WEATHER_COLS:
                record[f"{col}_near"] = nearest_record[col]
        else:
            for col in SELFBUILD_POLLUTANT_COLS:
                record[f"x_{col}_near"] = np.nan
            for col in SELFBUILD_WEATHER_COLS:
                record[f"{col}_near"] = np.nan
        
        # === 第三组：污染物窗口统计特征 ===
        for col in SELFBUILD_POLLUTANT_COLS:
            for stat in ["mean", "std", "min", "max", "median", "slope"]:
                key = f"{col}_{stat}"
                record[f"x_{col}_{stat}"] = window_stats.get(key, np.nan)
        
        # === 第四组：气象窗口特征 ===
        for col in SELFBUILD_WEATHER_COLS:
            for stat in ["mean", "std", "min", "max", "median", "slope"]:
                record[f"{col}_{stat}"] = window_stats.get(f"{col}_{stat}", np.nan)
        
        # 气象特殊统计量
        record["rain_sum"] = window_stats.get("rain_sum", np.nan)
        record["rain_delta"] = window_stats.get("rain_delta", np.nan)
        
        # 记录数
        record["record_count"] = window_stats.get("record_count", 0)
        
        hourly_records.append(record)
        
        # 进度显示
        if (i + 1) % 500 == 0:
            print(f"  已处理 {i+1}/{total_hours} 个整点...")
    
    # 转换为DataFrame
    hourly_df = pd.DataFrame(hourly_records)
    
    # === 第五组：时间特征 ===
    hourly_df = add_time_features(hourly_df, "time")
    
    print(f"\n小时级特征表构建完成: {len(hourly_df)} 行, {len(hourly_df.columns)} 列")
    
    # 统计信息
    print(f"\n特征统计:")
    print(f"  最近点可用率: {hourly_df['x_pm25_near'].notna().mean()*100:.2f}%")
    print(f"  窗口平均记录数: {hourly_df['record_count'].mean():.2f}")
    
    return hourly_df


def reorder_columns(df):
    """
    按照标准顺序重排列：
    1. time
    2. y_* (标签列)
    3. x_* (特征列)
    4. 时间特征
    """
    # 定义列顺序
    time_cols = ["time"]
    
    y_cols = ["y_pm25", "y_pm10", "y_co", "y_no2", "y_so2", "y_o3"]
    
    # 最近点特征
    x_near_cols = [f"x_{col}_near" for col in SELFBUILD_POLLUTANT_COLS]
    weather_near_cols = [f"{col}_near" for col in SELFBUILD_WEATHER_COLS]
    
    # 污染物窗口统计特征
    pollutant_stat_cols = []
    for col in SELFBUILD_POLLUTANT_COLS:
        for stat in ["mean", "std", "min", "max", "median", "slope"]:
            pollutant_stat_cols.append(f"x_{col}_{stat}")
    
    # 气象窗口特征
    weather_stat_cols = []
    for col in SELFBUILD_WEATHER_COLS:
        for stat in ["mean", "std", "min", "max", "median", "slope"]:
            weather_stat_cols.append(f"{col}_{stat}")
    
    # 气象特殊统计量
    weather_extra_cols = ["rain_sum", "rain_delta", "record_count"]
    
    # 时间特征
    time_feature_cols = ["hour", "weekday", "month", "t_idx",
                         "sin_hour", "cos_hour", "sin_month", "cos_month"]
    
    # 组合顺序
    ordered_cols = (
        time_cols +
        y_cols +
        x_near_cols +
        weather_near_cols +
        pollutant_stat_cols +
        weather_stat_cols +
        weather_extra_cols +
        time_feature_cols
    )
    
    # 只保留存在的列
    final_cols = [c for c in ordered_cols if c in df.columns]
    
    return df[final_cols]


def handle_missing_features(hourly_df):
    """
    对小时级样本表中的缺失值进行处理
    - 对于特征列，使用中位数填补
    """
    print("\n" + "="*60)
    print("缺失值处理（中位数填补）")
    print("="*60)
    
    # 排除 time 列和 y_ 列（标签）后的特征列
    exclude_prefixes = ["time", "y_"]
    feature_cols = [c for c in hourly_df.columns 
                   if not any(c.startswith(p) for p in exclude_prefixes)]
    
    # 统计填补前缺失情况
    print("填补前缺失值统计:")
    missing_before = hourly_df[feature_cols].isnull().sum()
    cols_with_missing = missing_before[missing_before > 0]
    if len(cols_with_missing) > 0:
        print(f"  {len(cols_with_missing)} 个特征列存在缺失")
        print(cols_with_missing.head(10))
    else:
        print("  无缺失值")
    
    # 中位数填补
    fill_count = 0
    for col in feature_cols:
        if hourly_df[col].isnull().any():
            median_val = hourly_df[col].median()
            hourly_df[col] = hourly_df[col].fillna(median_val)
            fill_count += 1
    
    print(f"\n  已对 {fill_count} 个特征列进行中位数填补")
    
    print("填补后缺失值统计:")
    missing_after = hourly_df[feature_cols].isnull().sum()
    remaining_missing = missing_after[missing_after > 0]
    print(remaining_missing if len(remaining_missing) > 0 else "无缺失值")
    
    return hourly_df


def save_hourly_data(hourly_df, output_path):
    """保存小时级样本表"""
    print("\n" + "="*60)
    print("保存小时级特征表")
    print("="*60)
    
    hourly_df.to_parquet(output_path, index=False)
    print(f"已保存至: {output_path}")
    
    # 显示列信息
    print(f"\n列结构 ({len(hourly_df.columns)} 列):")
    print("  标签列: y_pm25, y_pm10, y_co, y_no2, y_so2, y_o3")
    print("  最近点特征: x_*_near (污染物), *_near (气象)")
    print("  窗口统计: *_mean/std/min/max/median/slope")
    print("  时间特征: hour, weekday, month, t_idx, sin/cos编码")
    
    return hourly_df


def print_feature_table_summary(hourly_df):
    """打印特征表汇总"""
    print("\n" + "="*60)
    print("小时级特征表汇总")
    print("="*60)
    
    print(f"\n形状: {hourly_df.shape}")
    print(f"时间范围: {hourly_df['time'].min()} 至 {hourly_df['time'].max()}")
    
    print(f"\n列分组统计:")
    col_groups = {
        "标签列 (y_)": [c for c in hourly_df.columns if c.startswith("y_")],
        "最近点特征": [c for c in hourly_df.columns if c.endswith("_near")],
        "污染物统计": [c for c in hourly_df.columns if c.startswith("x_") and any(s in c for s in ["_mean", "_std", "_min", "_max", "_median", "_slope"])],
        "气象统计": [c for c in hourly_df.columns if c and not c.startswith("y_") and not c.startswith("x_") and not c.endswith("_near") and any(s in c for s in ["_mean", "_std", "_min", "_max", "_median", "_slope"])],
        "时间特征": ["hour", "weekday", "month", "t_idx", "sin_hour", "cos_hour", "sin_month", "cos_month"],
    }
    
    for name, cols in col_groups.items():
        existing = [c for c in cols if c in hourly_df.columns]
        print(f"  {name}: {len(existing)} 列")
    
    print(f"\n国控点标签统计:")
    y_cols = [c for c in hourly_df.columns if c.startswith("y_")]
    print(hourly_df[y_cols].describe().round(2))
    
    print(f"\nPM2.5 相关特征（前几条）:")
    pm25_cols = ["y_pm25", "x_pm25_near", "x_pm25_mean", "x_pm25_median"]
    existing_pm25_cols = [c for c in pm25_cols if c in hourly_df.columns]
    print(hourly_df[existing_pm25_cols].describe().round(2))