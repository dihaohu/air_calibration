"""
主训练脚本：整合所有校准模型

执行流程：
1. 加载数据
2. 构建设备漂移索引
3. 对六种污染物分别训练：
   - 模型A：一元线性校准（基线）
   - 模型B：多元静态线性校准
   - 模型C：动态校准主模型（带漂移项）
   - 模型D：XGBoost 对照模型
4. 模型选择与评估
5. 保存结果
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')

# 项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR
from src.build_features import (
    get_device_drift_index,
    get_feature_set_1, get_feature_set_2, get_feature_set_3,
    build_interaction_features,
    prepare_model_data, get_model_data_summary, filter_valid_samples,
)
from src.evaluate import (
    calculate_metrics, evaluate_model, compare_models,
    print_comparison_table, diagnose_residuals, print_metrics_by_range,
)
from src.train_linear import (
    train_linear_baseline, train_multivariate_static, 
    train_all_linear_models, select_best_linear_model,
)
from src.train_dynamic import (
    train_dynamic_regression, train_dynamic_with_tuning,
    interpret_drift_coefficients, add_drift_features,
)
from src.train_xgb import (
    train_xgboost, train_xgboost_with_tuning,
    compare_linear_vs_tree, HAS_XGB,
)

# 污染物列表
POLLUTANTS = ["pm25", "pm10", "co", "no2", "so2", "o3"]
POLLUTANT_NAMES = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "co": "CO",
    "no2": "NO2",
    "so2": "SO2",
    "o3": "O3",
}


def load_and_prepare_data():
    """加载数据并添加漂移索引"""
    print("="*70)
    print("步骤1：加载数据")
    print("="*70)
    
    hourly_df = pd.read_parquet(PROCESSED_DATA_DIR / "hourly_merged.parquet")
    print(f"已加载 {len(hourly_df)} 条小时级记录")
    print(f"时间范围: {hourly_df['time'].min()} 至 {hourly_df['time'].max()}")
    
    # 添加设备漂移索引
    hourly_df = get_device_drift_index(hourly_df)
    print(f"\n已添加设备漂移索引 drift_idx")
    print(f"  范围: {hourly_df['drift_idx'].min():.4f} ~ {hourly_df['drift_idx'].max():.4f}")
    
    # 添加交互特征
    for p in POLLUTANTS:
        hourly_df = add_drift_features(hourly_df, p)
    
    return hourly_df


def split_data(df, train_ratio=0.7, val_ratio=0.15):
    """按时间顺序划分数据集"""
    print("\n" + "="*70)
    print("步骤2：数据集划分（按时间顺序）")
    print("="*70)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"总样本数: {n}")
    print(f"训练集: {len(train_df)} 行 ({len(train_df)/n*100:.1f}%)")
    print(f"  时间: {train_df['time'].min()} ~ {train_df['time'].max()}")
    print(f"验证集: {len(val_df)} 行 ({len(val_df)/n*100:.1f}%)")
    print(f"  时间: {val_df['time'].min()} ~ {val_df['time'].max()}")
    print(f"测试集: {len(test_df)} 行 ({len(test_df)/n*100:.1f}%)")
    print(f"  时间: {test_df['time'].min()} ~ {test_df['time'].max()}")
    
    return train_df, val_df, test_df


def train_pollutant_models(pollutant, train_df, val_df, test_df):
    """
    对单一污染物训练所有模型
    """
    p_name = POLLUTANT_NAMES.get(pollutant, pollutant.upper())
    print(f"\n{'#'*70}")
    print(f"## {p_name} 校准模型训练")
    print(f"{'#'*70}")
    
    # 获取目标列名
    target_col = f"y_{pollutant}"
    
    # 获取特征集
    features_1 = get_feature_set_1(pollutant)  # 一元线性
    features_2 = get_feature_set_2(pollutant)  # 多元静态
    features_3 = get_feature_set_3(pollutant)   # 动态完整
    
    # 确保所有特征存在
    available_features_1 = [f for f in features_1 if f in train_df.columns]
    available_features_2 = [f for f in features_2 if f in train_df.columns]
    available_features_3 = [f for f in features_3 if f in train_df.columns]
    
    # 添加交互特征到可用特征
    if f"drift_x_{pollutant}_near" in train_df.columns:
        available_features_3.append(f"drift_x_{pollutant}_near")
    
    all_results = {}
    
    # ===== 模型A：一元线性基线 =====
    print(f"\n{'='*50}")
    print(f"模型A：一元线性校准")
    print(f"{'='*50}")
    
    if len(available_features_1) > 0:
        X_train = train_df[available_features_1].copy()
        X_val = val_df[available_features_1].copy()
        y_train = train_df[target_col]
        y_val = val_df[target_col]
        
        model_a, scaler_a, results_a = train_linear_baseline(
            X_train, y_train, X_val, y_val
        )
        all_results["A_baseline"] = {
            "model": model_a,
            "scaler": scaler_a,
            "results": results_a,
            "features": available_features_1,
        }
    else:
        print("  警告: 无法获取近邻特征，跳过模型A")
    
    # ===== 模型B：多元静态线性 =====
    print(f"\n{'='*50}")
    print(f"模型B：多元静态线性校准")
    print(f"{'='*50}")
    
    if len(available_features_2) > 0:
        X_train = train_df[available_features_2].copy()
        X_val = val_df[available_features_2].copy()
        y_train = train_df[target_col]
        y_val = val_df[target_col]
        
        # 填补缺失值
        X_train, y_train = filter_valid_samples(X_train, y_train)
        X_val, y_val = filter_valid_samples(X_val, y_val)
        
        for model_type in ["ridge", "huber"]:
            try:
                model_b, scaler_b, results_b = train_multivariate_static(
                    X_train, y_train, X_val, y_val, model_type
                )
                all_results[f"B_static_{model_type}"] = {
                    "model": model_b,
                    "scaler": scaler_b,
                    "results": results_b,
                    "features": available_features_2,
                }
            except Exception as e:
                print(f"  警告: {model_type} 训练失败 - {e}")
    else:
        print("  警告: 特征不足，跳过模型B")
    
    # ===== 模型C：动态校准主模型 =====
    print(f"\n{'='*50}")
    print(f"模型C：动态校准主模型")
    print(f"{'='*50}")
    
    if len(available_features_3) > 0:
        X_train = train_df[available_features_3].copy()
        X_val = val_df[available_features_3].copy()
        y_train = train_df[target_col]
        y_val = val_df[target_col]
        
        # 填补缺失值
        X_train, y_train = filter_valid_samples(X_train, y_train)
        X_val, y_val = filter_valid_samples(X_val, y_val)
        
        # 添加漂移交互特征
        X_train = build_interaction_features(X_train, pollutant)
        X_val = build_interaction_features(X_val, pollutant)
        
        # 确保交互特征存在
        interaction_col = f"drift_x_{pollutant}_near"
        if interaction_col not in X_train.columns:
            X_train[interaction_col] = train_df.loc[X_train.index, "drift_idx"] * train_df.loc[X_train.index, f"x_{pollutant}_near"]
            X_val[interaction_col] = val_df.loc[X_val.index, "drift_idx"] * val_df.loc[X_val.index, f"x_{pollutant}_near"]
        
        # 训练动态模型
        model_c, scaler_c, results_c = train_dynamic_with_tuning(
            X_train, y_train, X_val, y_val, pollutant
        )
        all_results["C_dynamic"] = {
            "model": model_c,
            "scaler": scaler_c,
            "results": results_c,
            "features": list(X_train.columns),
        }
        
        # 解释漂移项
        interpret_drift_coefficients(results_c)
    else:
        print("  警告: 特征不足，跳过模型C")
    
    # ===== 模型D：XGBoost 对照 =====
    print(f"\n{'='*50}")
    print(f"模型D：XGBoost 对照模型")
    print(f"{'='*50}")
    
    if HAS_XGB and len(available_features_3) > 0:
        X_train = train_df[available_features_3].copy()
        X_val = val_df[available_features_3].copy()
        y_train = train_df[target_col]
        y_val = val_df[target_col]
        
        model_d, results_d = train_xgboost_with_tuning(
            X_train, y_train, X_val, y_val, pollutant
        )
        all_results["D_xgboost"] = {
            "model": model_d,
            "scaler": None,
            "results": results_d,
            "features": available_features_3,
        }
    elif not HAS_XGB:
        print("  XGBoost 未安装，跳过模型D")
    
    return all_results


def select_best_model(all_results, metric="RMSE"):
    """选择验证集上最优的模型"""
    best_name = None
    best_score = np.inf
    best_info = None
    
    for name, info in all_results.items():
        results = info["results"]
        if "val_metrics" in results:
            score = results["val_metrics"].get(metric, np.inf)
            if score < best_score:
                best_score = score
                best_name = name
                best_info = info
    
    return best_name, best_info, best_score


def evaluate_on_test(all_results, test_df, pollutant):
    """在测试集上评估所有模型"""
    target_col = f"y_{pollutant}"
    y_test = test_df[target_col]
    
    test_predictions = {}
    
    for model_name, info in all_results.items():
        model = info["model"]
        scaler = info["scaler"]
        features = info["features"]
        
        if model is None:
            continue
        
        X_test = test_df[features].copy()
        
        # 添加交互特征
        X_test = build_interaction_features(X_test, pollutant)
        interaction_col = f"drift_x_{pollutant}_near"
        if interaction_col not in X_test.columns and interaction_col in model.feature_names if hasattr(model, 'feature_names') else False:
            pass
        
        # 填补缺失值
        for col in X_test.columns:
            if X_test[col].isnull().any():
                X_test[col] = X_test[col].fillna(X_test[col].median())
        
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
        
        test_predictions[model_name] = y_pred
    
    return test_predictions, y_test


def print_summary_table(all_results, test_predictions, y_test):
    """打印所有模型在测试集上的对比表"""
    print("\n" + "="*70)
    print("测试集评估结果汇总")
    print("="*70)
    
    rows = []
    for model_name, y_pred in test_predictions.items():
        metrics = calculate_metrics(y_test.values, y_pred)
        metrics["模型"] = model_name
        rows.append(metrics)
    
    df = pd.DataFrame(rows)
    
    # 排序列
    display_cols = ["模型", "MAE", "RMSE", "R2", "n_samples"]
    df_display = df[[c for c in display_cols if c in df.columns]]
    
    print(df_display.to_string(index=False))
    
    return df


def main():
    """主函数"""
    print("\n" + "="*70)
    print("空气质量传感器校准模型训练")
    print("="*70)
    
    # 1. 加载数据
    hourly_df = load_and_prepare_data()
    
    # 2. 划分数据集
    train_df, val_df, test_df = split_data(hourly_df)
    
    # 3. 对每种污染物训练模型
    all_pollutant_results = {}
    
    for pollutant in POLLUTANTS:
        all_results = train_pollutant_models(pollutant, train_df, val_df, test_df)
        
        # 选择最优模型
        best_name, best_info, best_score = select_best_model(all_results)
        print(f"\n{pollutant.upper()} 最优模型: {best_name} (验证集RMSE: {best_score:.3f})")
        
        # 测试集评估
        test_preds, y_test = evaluate_on_test(all_results, test_df, pollutant)
        print_summary_table(all_results, test_preds, y_test)
        
        all_pollutant_results[pollutant] = {
            "all_models": all_results,
            "best_model": best_name,
            "best_info": best_info,
            "test_predictions": test_preds,
            "y_test": y_test,
        }
    
    # 4. 汇总所有污染物的结果
    print("\n" + "="*70)
    print("所有污染物校准结果汇总")
    print("="*70)
    
    summary_rows = []
    for pollutant, results in all_pollutant_results.items():
        best_name = results["best_model"]
        test_preds = results["test_predictions"]
        y_test = results["y_test"]
        
        if best_name in test_preds:
            y_pred = test_preds[best_name]
            metrics = calculate_metrics(y_test.values, y_pred)
            metrics["污染物"] = POLLUTANT_NAMES.get(pollutant, pollutant)
            metrics["最优模型"] = best_name
            summary_rows.append(metrics)
    
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df[["污染物", "最优模型", "MAE", "RMSE", "R2"]].to_string(index=False))
    
    # 5. 保存结果
    output_dir = PROCESSED_DATA_DIR / "models"
    output_dir.mkdir(exist_ok=True)
    
    # 保存汇总
    summary_df.to_csv(output_dir / "calibration_summary.csv", index=False)
    print(f"\n结果已保存至: {output_dir / 'calibration_summary.csv'}")
    
    # 打印最终结论
    print("\n" + "="*70)
    print("校准效果总结")
    print("="*70)
    
    for _, row in summary_df.iterrows():
        print(f"\n{row['污染物']}:")
        print(f"  最优模型: {row['最优模型']}")
        print(f"  测试集 MAE: {row['MAE']:.3f} μg/m³")
        print(f"  测试集 RMSE: {row['RMSE']:.3f} μg/m³")
        print(f"  测试集 R²: {row['R2']:.4f}")
    
    return all_pollutant_results, summary_df


if __name__ == "__main__":
    all_results, summary_df = main()
