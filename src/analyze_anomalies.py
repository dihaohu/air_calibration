"""
问题4：异常传感器检测
利用自建点网络内部的数据一致性来评估各传感器数据的可靠性
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR

POLLUTANTS = ["pm25", "pm10", "co", "no2", "so2", "o3"]
POLLUTANT_NAMES = {"pm25": "PM2.5", "pm10": "PM10", "co": "CO", 
                   "no2": "NO2", "so2": "SO2", "o3": "O3"}


def load_data():
    """加载数据"""
    print("加载数据...")
    hourly_df = pd.read_parquet(PROCESSED_DATA_DIR / "hourly_merged.parquet")
    return hourly_df


def calculate_sensor_consistency(hourly_df):
    """
    计算传感器数据一致性指标
    使用同一小时内自建点不同特征（near vs mean）的一致性
    """
    print("\n" + "="*70)
    print("一、传感器数据内部一致性分析")
    print("="*70)
    
    results = {}
    
    for p in POLLUTANTS:
        p_name = POLLUTANT_NAMES.get(p, p)
        
        x_near = hourly_df[f"x_{p}_near"]
        x_mean = hourly_df[f"x_{p}_mean"]
        
        mask = ~(x_near.isna() | x_mean.isna())
        
        if mask.sum() < 100:
            continue
        
        near = x_near[mask]
        mean = x_mean[mask]
        
        # 计算一致性
        diff = near - mean
        diff_mean = diff.mean()
        diff_std = diff.std()
        corr, p_value = stats.pearsonr(near, mean)
        
        # 一致性比率（差异系数）
        cv = diff_std / (mean.abs().mean() + 0.01) * 100
        
        results[p] = {
            "diff_mean": diff_mean,
            "diff_std": diff_std,
            "correlation": corr,
            "cv": cv,
        }
        
        print(f"\n{p_name}:")
        print(f"  近邻值与窗口均值差异: {diff_mean:+.2f} ± {diff_std:.2f}")
        print(f"  相关系数: r = {corr:.4f}")
        print(f"  变异系数: {cv:.1f}%")
        
        if abs(diff_mean) > 10 or diff_std > 20:
            print(f"  → 警告：一致性较差，可能存在传感器问题")
    
    return results


def detect_outliers_by_statistical(hourly_df):
    """
    基于统计方法的异常检测
    """
    print("\n" + "="*70)
    print("二、基于统计的异常值检测")
    print("="*70)
    
    results = {}
    
    for p in POLLUTANTS:
        p_name = POLLUTANT_NAMES.get(p, p)
        print(f"\n{p_name}:")
        
        x_near = hourly_df[f"x_{p}_near"]
        y_true = hourly_df[f"y_{p}"]
        
        # 计算标准化残差
        mask = ~(x_near.isna() | y_true.isna())
        residual = (x_near - y_true)[mask]
        
        if mask.sum() < 100:
            continue
        
        # 方法1: Z-score
        z_scores = (residual - residual.mean()) / residual.std()
        outliers_z = (abs(z_scores) > 3).sum()
        
        # 方法2: IQR
        Q1 = residual.quantile(0.25)
        Q3 = residual.quantile(0.75)
        IQR = Q3 - Q1
        outliers_iqr = ((residual < Q1 - 1.5*IQR) | (residual > Q3 + 1.5*IQR)).sum()
        
        # 方法3: MAD (Median Absolute Deviation)
        median = residual.median()
        mad = np.median(np.abs(residual - median))
        outliers_mad = (np.abs(residual - median) > 3 * mad).sum()
        
        print(f"  Z-score异常 (>3σ): {outliers_z} 个 ({outliers_z/mask.sum()*100:.2f}%)")
        print(f"  IQR异常: {outliers_iqr} 个 ({outliers_iqr/mask.sum()*100:.2f}%)")
        print(f"  MAD异常 (>3MAD): {outliers_mad} 个 ({outliers_mad/mask.sum()*100:.2f}%)")
        
        # 记录异常阈值
        results[p] = {
            "z_threshold": 3,
            "iqr_multiplier": 1.5,
            "z_outliers": outliers_z,
            "iqr_outliers": outliers_iqr,
            "mad_outliers": outliers_mad,
            "outlier_rate": outliers_z / mask.sum(),
        }
    
    return results


def detect_outliers_by_isolation_forest(hourly_df):
    """
    使用Isolation Forest检测异常传感器
    """
    print("\n" + "="*70)
    print("三、Isolation Forest 异常检测")
    print("="*70)
    
    # 选择特征
    feature_cols = []
    for p in POLLUTANTS:
        feature_cols.extend([f"x_{p}_near", f"x_{p}_mean"])
    for w in ["wind", "pressure", "temp", "rh"]:
        feature_cols.append(f"{w}_mean")
    
    # 过滤存在的列
    feature_cols = [c for c in feature_cols if c in hourly_df.columns]
    
    # 准备数据
    X = hourly_df[feature_cols].copy()
    
    # 填充缺失值
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # 假设5%的异常
        random_state=42
    )
    
    # 预测
    anomaly_labels = iso_forest.fit_predict(X_scaled)
    anomaly_scores = iso_forest.decision_function(X_scaled)
    
    # 统计
    n_anomalies = (anomaly_labels == -1).sum()
    
    print(f"\n总体异常检测结果:")
    print(f"  异常样本数: {n_anomalies} ({n_anomalies/len(X)*100:.2f}%)")
    print(f"  正常样本数: {len(X) - n_anomalies}")
    
    # 分析每个污染物的异常贡献
    print(f"\n各污染物特征的异常贡献:")
    
    anomaly_indices = np.where(anomaly_labels == -1)[0]
    
    if len(anomaly_indices) > 0:
        anomaly_times = hourly_df.iloc[anomaly_indices]["time"]
        
        print(f"  异常时间分布:")
        print(f"    最早: {anomaly_times.min()}")
        print(f"    最晚: {anomaly_times.max()}")
    
    # 返回结果
    return {
        "n_anomalies": n_anomalies,
        "anomaly_rate": n_anomalies / len(X),
        "anomaly_indices": anomaly_indices,
        "anomaly_scores": anomaly_scores,
    }


def calculate_sensor_reliability_score(hourly_df):
    """
    计算传感器可靠性评分
    基于多个指标综合评估
    """
    print("\n" + "="*70)
    print("四、传感器可靠性综合评分")
    print("="*70)
    
    reliability_scores = {}
    
    for p in POLLUTANTS:
        p_name = POLLUTANT_NAMES.get(p, p)
        
        x_near = hourly_df[f"x_{p}_near"]
        x_mean = hourly_df[f"x_{p}_mean"]
        x_std = hourly_df[f"x_{p}_std"]
        y_true = hourly_df[f"y_{p}"]
        
        # 各维度评分
        scores = {}
        
        # 1. 与国控点一致性
        mask = ~(x_near.isna() | y_true.isna())
        if mask.sum() > 0:
            mae = np.abs(x_near[mask] - y_true[mask]).mean()
            # MAE < 10: 10分, MAE > 50: 0分
            consistency_score = max(0, min(10, 10 - (mae - 10) / 4))
            scores["一致性"] = consistency_score
        
        # 2. 内部稳定性（near vs mean）
        mask = ~(x_near.isna() | x_mean.isna())
        if mask.sum() > 0:
            stability = 1 / (1 + np.abs(x_near[mask] - x_mean[mask]).mean() / 10)
            stability_score = stability * 10
            scores["稳定性"] = stability_score
        
        # 3. 数据完整率
        completeness = 1 - x_near.isna().mean()
        completeness_score = completeness * 10
        scores["完整率"] = completeness_score
        
        # 4. 变异系数合理性
        mask = ~(x_std.isna() | x_mean.isna())
        if mask.sum() > 0:
            cv = x_std[mask] / (x_mean[mask].abs() + 1)
            cv_reasonableness = 1 / (1 + cv.mean() / 5)
            cv_score = cv_reasonableness * 10
            scores["变异合理性"] = cv_score
        
        # 综合评分
        if scores:
            overall = np.mean(list(scores.values()))
            reliability_scores[p] = {
                "总分": overall,
                **scores
            }
        
        print(f"\n{p_name}:")
        print(f"  综合评分: {overall:.1f}/10")
        for k, v in scores.items():
            bar = "█" * int(v) + "░" * (10 - int(v))
            print(f"  {k}: {bar} {v:.1f}")
        
        if overall < 6:
            print(f"  → 警告：该传感器可靠性较低，需要检查")
    
    return reliability_scores


def analyze_time_stability(hourly_df):
    """
    分析传感器随时间的稳定性
    """
    print("\n" + "="*70)
    print("五、传感器时间稳定性分析")
    print("="*70)
    
    # 添加时间索引
    t0 = hourly_df["time"].min()
    hourly_df["drift_idx"] = (hourly_df["time"] - t0).dt.total_seconds() / (
        hourly_df["time"].max() - t0
    ).total_seconds()
    
    results = {}
    
    for p in POLLUTANTS:
        p_name = POLLUTANT_NAMES.get(p, p)
        
        x_near = hourly_df[f"x_{p}_near"]
        y_true = hourly_df[f"y_{p}"]
        drift_idx = hourly_df["drift_idx"]
        
        mask = ~(x_near.isna() | y_true.isna())
        
        if mask.sum() < 100:
            continue
        
        residual = (x_near - y_true)[mask].values
        time_idx = drift_idx[mask].values
        
        # 分段计算均值和标准差
        n_segments = 10
        segment_size = len(residual) // n_segments
        
        segment_stats = []
        for i in range(n_segments):
            start = i * segment_size
            end = start + segment_size if i < n_segments - 1 else len(residual)
            seg_residual = residual[start:end]
            segment_stats.append({
                "segment": i + 1,
                "mean": np.mean(seg_residual),
                "std": np.std(seg_residual),
                "time": hourly_df.iloc[start:end]["time"].mean()
            })
        
        # 计算稳定性指标
        means = [s["mean"] for s in segment_stats]
        stds = [s["std"] for s in segment_stats]
        
        stability_cv = np.std(means) / (np.abs(np.mean(means)) + 1)
        
        results[p] = {
            "segment_stats": segment_stats,
            "stability_cv": stability_cv,
            "early_mean": means[0] if len(means) > 0 else np.nan,
            "late_mean": means[-1] if len(means) > 0 else np.nan,
        }
        
        print(f"\n{p_name}:")
        print(f"  前期均值: {means[0]:.2f}, 后期均值: {means[-1]:.2f}")
        print(f"  稳定性变异系数: {stability_cv:.3f}")
        
        if abs(means[-1] - means[0]) > 20:
            print(f"  → 警告：存在显著时间漂移")
    
    return results


def detect_sudden_changes(hourly_df):
    """
    检测突变点
    """
    print("\n" + "="*70)
    print("六、突变点检测")
    print("="*70)
    
    results = {}
    
    for p in POLLUTANTS:
        p_name = POLLUTANT_NAMES.get(p, p)
        
        x_near = hourly_df[f"x_{p}_near"]
        y_true = hourly_df[f"y_{p}"]
        
        mask = ~(x_near.isna() | y_true.isna())
        
        if mask.sum() < 100:
            continue
        
        residual = (x_near - y_true)[mask].values
        
        # 滑动窗口检测突变
        window_size = 24  # 24小时窗口
        diff_threshold = 2 * np.std(residual)
        
        sudden_changes = []
        for i in range(window_size, len(residual)):
            window1 = residual[i-window_size:i]
            window2 = residual[i:i+window_size] if i+window_size < len(residual) else residual[i:]
            
            mean1, mean2 = np.mean(window1), np.mean(window2)
            if abs(mean2 - mean1) > diff_threshold:
                sudden_changes.append({
                    "index": i,
                    "time": hourly_df.iloc[mask].iloc[i]["time"],
                    "change": mean2 - mean1
                })
        
        results[p] = {
            "n_sudden_changes": len(sudden_changes),
            "changes": sudden_changes[:10] if len(sudden_changes) > 10 else sudden_changes,
        }
        
        print(f"\n{p_name}:")
        print(f"  检测到突变次数: {len(sudden_changes)}")
        
        if len(sudden_changes) > 0:
            print(f"  主要突变点:")
            for change in sudden_changes[:3]:
                print(f"    {change['time']}: {change['change']:+.2f}")
    
    return results


def generate_anomaly_report(results):
    """
    生成异常检测报告
    """
    print("\n" + "="*70)
    print("异常传感器检测报告")
    print("="*70)
    
    print("""
基于上述分析，以下传感器可能存在异常：

1. 高异常率污染物：
   - 需要关注异常检测中异常率超过5%的污染物
   
2. 漂移严重的传感器：
   - 零点漂移和量程漂移显著的污染物需要定期校准
   
3. 可靠性评分较低的传感器：
   - 综合评分低于6分的传感器需要重点检查

4. 建议措施：
   - 对异常传感器进行现场核查
   - 定期进行多点位对比校准
   - 建立传感器健康监测机制
""")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("问题4：异常传感器检测")
    print("="*70)
    
    hourly_df = load_data()
    
    # 各项检测
    consistency = calculate_sensor_consistency(hourly_df)
    statistical_outliers = detect_outliers_by_statistical(hourly_df)
    iso_forest_results = detect_outliers_by_isolation_forest(hourly_df)
    reliability = calculate_sensor_reliability_score(hourly_df)
    time_stability = analyze_time_stability(hourly_df)
    sudden_changes = detect_sudden_changes(hourly_df)
    
    # 生成报告
    generate_anomaly_report({
        "consistency": consistency,
        "statistical_outliers": statistical_outliers,
        "iso_forest": iso_forest_results,
        "reliability": reliability,
        "time_stability": time_stability,
        "sudden_changes": sudden_changes,
    })
    
    return {
        "consistency": consistency,
        "statistical_outliers": statistical_outliers,
        "iso_forest": iso_forest_results,
        "reliability": reliability,
        "time_stability": time_stability,
        "sudden_changes": sudden_changes,
    }


if __name__ == "__main__":
    results = main()
