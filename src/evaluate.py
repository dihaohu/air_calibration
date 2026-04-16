"""
统一评估模块：计算回归指标、打印诊断报告
"""

import numpy as np
import pandas as pd
from pathlib import Path


def calculate_metrics(y_true, y_pred):
    """
    计算回归评估指标
    
    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值
    
    Returns:
    --------
    dict : 包含 MAE, RMSE, R2, MAPE, MB 的字典
    """
    # 过滤掉 NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t = np.array(y_true)[mask]
    y_p = np.array(y_pred)[mask]
    
    if len(y_t) == 0:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "MAPE": np.nan,
            "MB": np.nan,  # Mean Bias
            "n_samples": 0,
        }
    
    # MAE
    mae = np.mean(np.abs(y_t - y_p))
    
    # RMSE
    rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
    
    # R2
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    
    # MAPE (避免除零)
    mask_mape = mask & (y_t != 0)
    if mask_mape.sum() > 0:
        mape = np.mean(np.abs((y_t[mask_mape] - y_p[mask_mape]) / y_t[mask_mape])) * 100
    else:
        mape = np.nan
    
    # MB (Mean Bias，平均偏差)
    mb = np.mean(y_p - y_t)
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "MB": mb,
        "n_samples": len(y_t),
    }


def calculate_improvement(metrics_before, metrics_after):
    """
    计算校准前后的改善比例
    
    Parameters:
    -----------
    metrics_before : dict
        校准前的指标
    metrics_after : dict
        校准后的指标
    
    Returns:
    --------
    dict : 包含各指标改善百分比的字典
    """
    improvement = {}
    for key in ["MAE", "RMSE", "MAPE"]:
        if key in metrics_before and key in metrics_after:
            if metrics_before[key] != 0 and not np.isnan(metrics_before[key]):
                improvement[f"{key}_improvement"] = (
                    (metrics_before[key] - metrics_after[key]) / metrics_before[key] * 100
                )
            else:
                improvement[f"{key}_improvement"] = np.nan
    return improvement


def evaluate_model(model_name, y_true, y_pred, y_baseline=None, verbose=True):
    """
    评估单个模型
    
    Parameters:
    -----------
    model_name : str
        模型名称
    y_true : array-like
        真实值
    y_pred : array-like
        预测值
    y_baseline : array-like, optional
        基线预测值（用于计算改善比例）
    verbose : bool
        是否打印结果
    
    Returns:
    --------
    dict : 评估结果字典
    """
    metrics = calculate_metrics(y_true, y_pred)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"模型: {model_name}")
        print(f"{'='*60}")
        print(f"  样本数: {metrics['n_samples']}")
        print(f"  MAE (平均绝对误差): {metrics['MAE']:.3f} μg/m³")
        print(f"  RMSE (均方根误差): {metrics['RMSE']:.3f} μg/m³")
        print(f"  R² (决定系数): {metrics['R2']:.4f}")
        print(f"  MAPE (平均百分比误差): {metrics['MAPE']:.2f}%")
        print(f"  MB (平均偏差): {metrics['MB']:+.3f} μg/m³ "
              f"({'低估' if metrics['MB'] < 0 else '高估' if metrics['MB'] > 0 else '无偏'})")
    
    if y_baseline is not None:
        baseline_metrics = calculate_metrics(y_true, y_baseline)
        improvement = calculate_improvement(baseline_metrics, metrics)
        
        if verbose:
            print(f"\n  --- 相对于基线的改善 ---")
            print(f"  MAE 改善: {improvement.get('MAE_improvement', 0):+.2f}%")
            print(f"  RMSE 改善: {improvement.get('RMSE_improvement', 0):+.2f}%")
        
        metrics.update(improvement)
        metrics["baseline_MAE"] = baseline_metrics["MAE"]
        metrics["baseline_RMSE"] = baseline_metrics["RMSE"]
    
    return metrics


def compare_models(results_dict):
    """
    比较多个模型的评估结果
    
    Parameters:
    -----------
    results_dict : dict
        {model_name: metrics_dict} 的字典
    
    Returns:
    --------
    DataFrame : 比较表格
    """
    rows = []
    for model_name, metrics in results_dict.items():
        row = {"模型": model_name}
        row.update(metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 排序列
    cols_order = ["模型", "n_samples", "MAE", "RMSE", "R2", "MAPE", "MB"]
    available_cols = [c for c in cols_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in available_cols]
    df = df[available_cols + other_cols]
    
    return df


def print_comparison_table(results_dict):
    """
    打印模型比较表格
    """
    df = compare_models(results_dict)
    
    print("\n" + "="*80)
    print("模型比较汇总")
    print("="*80)
    
    # 格式化输出
    display_cols = ["模型", "n_samples", "MAE", "RMSE", "R2", "MAPE", "MB"]
    display_cols = [c for c in display_cols if c in df.columns]
    df_display = df[display_cols].copy()
    
    # 数值列格式化
    for col in ["MAE", "RMSE", "MB"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    
    if "MAPE" in df_display.columns:
        df_display["MAPE"] = df_display["MAPE"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    
    if "R2" in df_display.columns:
        df_display["R2"] = df_display["R2"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    
    print(df_display.to_string(index=False))
    
    return df


def calculate_residuals(y_true, y_pred):
    """计算残差"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return y_true[mask] - y_pred[mask]


def diagnose_residuals(y_true, y_pred, verbose=True):
    """
    残差诊断分析
    
    检查：
    1. 残差分布是否正态
    2. 残差是否随预测值变化（异方差）
    3. 残差自相关
    """
    residuals = calculate_residuals(y_true, y_pred)
    
    if verbose:
        print("\n" + "-"*50)
        print("残差诊断")
        print("-"*50)
    
    results = {}
    
    # 残差统计
    results["residual_mean"] = np.mean(residuals)
    results["residual_std"] = np.std(residuals)
    results["residual_skewness"] = pd.Series(residuals).skew()
    results["residual_kurtosis"] = pd.Series(residuals).kurtosis()
    
    if verbose:
        print(f"  均值: {results['residual_mean']:.4f} (应接近0)")
        print(f"  标准差: {results['residual_std']:.4f}")
        print(f"  偏度: {results['residual_skewness']:.4f} (正值表示右偏)")
        print(f"  峰度: {results['residual_kurtosis']:.4f} (正值表示厚尾)")
    
    # 残差分位数
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    if verbose:
        print(f"\n  分位数:")
        for q in quantiles:
            val = np.quantile(residuals, q)
            print(f"    {q*100:.0f}%: {val:.4f}")
    
    return results


def calculate_metrics_by_range(y_true, y_pred, thresholds=None):
    """
    按浓度区间计算指标
    
    Parameters:
    -----------
    y_true : array-like
    y_pred : array-like
    thresholds : list
        区间边界，如 [35, 75, 115, 150]
    
    Returns:
    --------
    DataFrame : 各区间指标
    """
    if thresholds is None:
        thresholds = [35, 75, 115, 150]
    
    labels = ["优(0-35)", "良(35-75)", "轻度污染(75-115)", 
               "中度污染(115-150)", "重度及以上(>150)"]
    
    ranges = [(0, thresholds[0])]
    for i in range(len(thresholds)-1):
        ranges.append((thresholds[i], thresholds[i+1]))
    ranges.append((thresholds[-1], float('inf')))
    
    results = []
    for (low, high), label in zip(ranges, labels):
        mask = (y_true >= low) & (y_true < high)
        y_t = y_true[mask]
        y_p = y_pred[mask]
        
        if len(y_t) > 0:
            metrics = calculate_metrics(y_t, y_p)
            metrics["range"] = label
            metrics["n_samples"] = mask.sum()
            results.append(metrics)
    
    return pd.DataFrame(results)


def print_metrics_by_range(y_true, y_pred, pollutant_name="PM2.5", thresholds=None):
    """打印按浓度区间的指标"""
    if thresholds is None:
        # PM2.5 的分级标准
        thresholds_map = {
            "PM2.5": [35, 75, 115, 150],
            "PM10": [50, 150, 250, 350],
        }
        thresholds = thresholds_map.get(pollutant_name, [35, 75, 115, 150])
    
    df = calculate_metrics_by_range(y_true, y_pred, thresholds)
    
    print(f"\n{'='*70}")
    print(f"按{pollutant_name}浓度区间的误差分析")
    print(f"{'='*70}")
    print(f"{'区间':<20} {'样本数':<10} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
    print("-"*70)
    
    for _, row in df.iterrows():
        print(f"{row['range']:<20} {int(row['n_samples']):<10} "
              f"{row['MAE']:<10.2f} {row['RMSE']:<10.2f} {row['R2']:<10.4f}")
    
    return df


def save_results(results_dict, output_path):
    """
    保存评估结果到文件
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = compare_models(results_dict)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存至: {output_path}")


# ============================================================================
# 可视化数据准备
# ============================================================================

def prepare_scatter_data(y_true, y_pred, n_bins=10):
    """
    准备散点图数据（按分位数分箱统计）
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t = np.array(y_true)[mask]
    y_p = np.array(y_pred)[mask]
    
    # 计算分位数边界
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(y_t, quantiles)
    bin_edges[0] = 0  # 确保包含0
    bin_edges[-1] = np.inf  # 确保包含最大值
    
    bins_data = []
    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i+1]
        bin_mask = (y_t >= low) & (y_t < high)
        
        if bin_mask.sum() > 0:
            bins_data.append({
                "bin_center": (low + high) / 2,
                "y_true_mean": np.mean(y_t[bin_mask]),
                "y_pred_mean": np.mean(y_p[bin_mask]),
                "y_true_std": np.std(y_t[bin_mask]),
                "y_pred_std": np.std(y_p[bin_mask]),
                "n_samples": bin_mask.sum(),
            })
    
    return pd.DataFrame(bins_data)
