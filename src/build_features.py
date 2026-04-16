"""
特征构建模块：定义三层特征集合和设备老化索引

三层特征集：
1. 最小集合（第一组）：该污染物本身的 near, mean, std, slope
2. 扩展集合（第二组）：最小集合 + 气象均值特征 + 日周期项
3. 完整集合（第三组）：扩展集合 + 其他污染物特征 + 漂移项
"""

import numpy as np
import pandas as pd

# 污染物列表
POLLUTANTS = ["pm25", "pm10", "co", "no2", "so2", "o3"]

# 气象特征列表
WEATHER_COLS = ["wind", "pressure", "rain", "temp", "rh"]

# 时间特征列表
TIME_COLS = ["hour", "weekday", "month", "sin_hour", "cos_hour"]


def get_device_drift_index(df, time_col="time"):
    """
    计算设备老化索引 u(t) = (t - t0) / T
    其中 t0 是起始时间，T 是总时长
    u(t) 从 0 线性增长到 1
    """
    t0 = df[time_col].min()
    T = (df[time_col].max() - t0).total_seconds()
    df = df.copy()
    df["drift_idx"] = (df[time_col] - t0).dt.total_seconds() / T
    return df


def get_feature_set_1(pollutant):
    """
    第一组（最小集合）：该污染物本身的近邻值和窗口统计
    用于一元线性基线模型
    """
    return [
        f"x_{pollutant}_near",
    ]


def get_feature_set_2(pollutant):
    """
    第二组（扩展集合）：污染物特征 + 气象 + 日周期
    用于多元静态校准模型
    """
    features = [
        # 该污染物本身特征
        f"x_{pollutant}_near",
        f"x_{pollutant}_mean",
        f"x_{pollutant}_std",
        f"x_{pollutant}_slope",
    ]
    
    # 气象均值特征
    for w in WEATHER_COLS:
        features.append(f"{w}_mean")
    
    # 日周期项
    features.extend(["sin_hour", "cos_hour"])
    
    return features


def get_feature_set_3(pollutant):
    """
    第三组（完整集合）：扩展集合 + 其他污染物 + 漂移项
    用于动态校准主模型
    """
    features = get_feature_set_2(pollutant)
    
    # 其他污染物近邻特征（交叉干扰）
    for p in POLLUTANTS:
        if p != pollutant:
            features.append(f"x_{p}_near")
    
    # 设备老化索引
    features.append("drift_idx")
    
    # 量程漂移交互项（需要在使用时动态生成）
    # features.append(f"drift_x_{pollutant}_near")  # drift_idx * x_pollutant_near
    
    return features


def get_all_features(pollutant):
    """
    获取某污染物的完整特征集（用于XGBoost对照模型）
    """
    features = get_feature_set_3(pollutant)
    
    # 添加其他污染物的mean特征（如果有的话）
    for p in POLLUTANTS:
        if p != pollutant:
            features.append(f"x_{p}_mean")
    
    return features


def build_interaction_features(df, pollutant):
    """
    构建交互特征：漂移项与污染物读数的交互
    用于动态校准模型
    """
    df = df.copy()
    
    # 量程漂移交互项
    if "drift_idx" in df.columns and f"x_{pollutant}_near" in df.columns:
        df[f"drift_x_{pollutant}_near"] = df["drift_idx"] * df[f"x_{pollutant}_near"]
    
    return df


def prepare_model_data(df, pollutant, feature_cols, target_col="y_pm25"):
    """
    准备模型数据，过滤有效样本
    
    Parameters:
    -----------
    df : DataFrame
        小时级特征表
    pollutant : str
        污染物名称
    feature_cols : list
        特征列名列表
    target_col : str
        目标列名（如 y_pm25）
    
    Returns:
    --------
    X : DataFrame
        特征矩阵
    y : Series
        目标向量
    valid_mask : Series
        有效样本掩码
    """
    # 添加漂移交互特征
    df = build_interaction_features(df, pollutant)
    
    # 过滤掉目标为NaN的样本
    valid_mask = df[target_col].notna()
    
    # 确保所有特征列存在
    available_cols = [c for c in feature_cols if c in df.columns]
    
    # 处理漂移交互项（动态生成）
    if "drift_idx" in available_cols and f"x_{pollutant}_near" in df.columns:
        if f"drift_x_{pollutant}_near" not in df.columns:
            df[f"drift_x_{pollutant}_near"] = df["drift_idx"] * df[f"x_{pollutant}_near"]
        available_cols.append(f"drift_x_{pollutant}_near")
    
    # 去除重复列
    available_cols = list(dict.fromkeys(available_cols))
    
    # 获取有效样本的特征和目标
    X = df.loc[valid_mask, available_cols].copy()
    y = df.loc[valid_mask, target_col].copy()
    
    # 统计信息
    n_total = len(df)
    n_valid = valid_mask.sum()
    n_missing = n_total - n_valid
    
    print(f"  污染物: {pollutant}")
    print(f"  目标列: {target_col}")
    print(f"  特征数: {len(available_cols)}")
    print(f"  有效样本: {n_valid}/{n_total} ({n_valid/n_total*100:.1f}%)")
    
    return X, y, valid_mask, available_cols


def get_model_data_summary(X, y):
    """打印模型数据摘要"""
    print(f"\n  特征矩阵形状: {X.shape}")
    print(f"  目标向量形状: {y.shape}")
    print(f"  目标统计: 均值={y.mean():.2f}, 标准差={y.std():.2f}, "
          f"最小={y.min():.2f}, 最大={y.max():.2f}")
    
    # 特征缺失情况
    missing_per_col = X.isnull().sum()
    if missing_per_col.sum() > 0:
        cols_with_missing = missing_per_col[missing_per_col > 0]
        print(f"  存在缺失的特征列: {len(cols_with_missing)} 列")
    else:
        print(f"  无缺失值")
    
    return {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "target_mean": y.mean(),
        "target_std": y.std(),
        "target_min": y.min(),
        "target_max": y.max(),
    }


def filter_valid_samples(X, y, fill_na_method="median"):
    """
    过滤或填补缺失值
    
    Parameters:
    -----------
    X : DataFrame
        特征矩阵
    y : Series
        目标向量
    fill_na_method : str
        缺失值填补方法，"median" 或 "mean" 或 None
    
    Returns:
    --------
    X_clean : DataFrame
    y_clean : Series
    """
    # 联合有效掩码：X和y都必须非空
    # 对于X中的NaN，用中位数填补
    if fill_na_method is not None:
        X_clean = X.copy()
        for col in X_clean.columns:
            if X_clean[col].isnull().any():
                if fill_na_method == "median":
                    fill_val = X_clean[col].median()
                else:
                    fill_val = X_clean[col].mean()
                X_clean[col] = X_clean[col].fillna(fill_val)
    else:
        # 只保留完全有效的行
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()
        return X_clean, y_clean
    
    y_clean = y.copy()
    return X_clean, y_clean


# ============================================================================
# 特征重要性映射（用于线性模型系数解释）
# ============================================================================

FEATURE_CATEGORIES = {
    "本体响应": [f"x_{p}_{stat}" for p in POLLUTANTS 
                 for stat in ["near", "mean", "std", "slope"]],
    "气象修正": [f"{w}_mean" for w in WEATHER_COLS] + 
                [f"{w}_std" for w in WEATHER_COLS] + 
                ["rain_sum", "rain_delta"],
    "日周期": ["sin_hour", "cos_hour", "sin_month", "cos_month"],
    "交叉干扰": [f"x_{p}_near" for p in POLLUTANTS] + 
                [f"x_{p}_mean" for p in POLLUTANTS],
    "漂移项": ["drift_idx", "drift_x_pm25_near", "drift_x_pm10_near", 
               "drift_x_co_near", "drift_x_no2_near", "drift_x_so2_near", 
               "drift_x_o3_near"],
}


def get_feature_category(feature_name):
    """获取特征所属类别"""
    for category, features in FEATURE_CATEGORIES.items():
        if feature_name in features:
            return category
    return "其他"


def print_feature_summary(feature_cols, coefs=None):
    """打印特征摘要"""
    print("\n  特征类别分布:")
    category_counts = {}
    for col in feature_cols:
        cat = get_feature_category(col)
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count} 个特征")
    
    if coefs is not None:
        print("\n  特征系数（绝对值排序）:")
        coef_df = pd.DataFrame({
            "feature": feature_cols,
            "coef": coefs
        })
        coef_df["abs_coef"] = np.abs(coef_df["coef"])
        coef_df["category"] = coef_df["feature"].apply(get_feature_category)
        coef_df = coef_df.sort_values("abs_coef", ascending=False)
        
        for _, row in coef_df.head(15).iterrows():
            print(f"    {row['feature']:<30} {row['coef']:>10.4f}  ({row['category']})")
