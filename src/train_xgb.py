"""
XGBoost 对照模型：模型D

作为非线性性能对照，验证：
1. 天气因素和交叉干扰是否存在强非线性
2. 线性动态模型是否有非线性改进空间

树模型不做标准化，但保持输入列一致便于比较
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# XGBoost 可能未安装，先检查
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("警告: XGBoost 未安装，将跳过树模型训练")


def calculate_metrics(y_true, y_pred):
    """计算回归指标"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t = np.array(y_true)[mask]
    y_p = np.array(y_pred)[mask]
    
    if len(y_t) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "n_samples": 0}
    
    mae = np.mean(np.abs(y_t - y_p))
    rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
    
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "n_samples": len(y_t),
    }


def train_xgboost(X_train, y_train, X_val=None, y_val=None,
                   pollutant="pm25", params=None):
    """
    模型D：XGBoost 对照模型
    
    Parameters:
    -----------
    X_train, y_train : 训练数据
    X_val, y_val : 验证数据
    pollutant : str
    params : dict, XGBoost 参数
    
    Returns:
    --------
    model, results
    """
    if not HAS_XGB:
        print("\nXGBoost 未安装，跳过树模型")
        return None, None
    
    print("\n" + "="*60)
    print(f"模型D：XGBoost 对照模型")
    print("="*60)
    
    # 默认参数
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }
    
    # 创建模型
    model = xgb.XGBRegressor(**params)
    
    # 处理缺失值（XGBoost 可以处理NaN，但为了一致性，先填补）
    X_train_clean = X_train.copy()
    X_val_clean = X_val.copy() if X_val is not None else None
    
    for col in X_train_clean.columns:
        if X_train_clean[col].isnull().any():
            median_val = X_train_clean[col].median()
            X_train_clean[col] = X_train_clean[col].fillna(median_val)
            if X_val_clean is not None:
                X_val_clean[col] = X_val_clean[col].fillna(median_val)
    
    # 训练
    model.fit(X_train_clean, y_train)
    
    # 特征重要性
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": X_train_clean.columns,
        "importance": importance,
    }).sort_values("importance", ascending=False)
    
    print(f"\n特征重要性（Top 15）:")
    for _, row in importance_df.head(15).iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"  {row['feature']:<30} {row['importance']:.4f} {bar}")
    
    # 训练集指标
    y_train_pred = model.predict(X_train_clean)
    train_metrics = calculate_metrics(y_train, y_train_pred)
    
    print(f"\n训练集指标:")
    print(f"  MAE: {train_metrics['MAE']:.3f}")
    print(f"  RMSE: {train_metrics['RMSE']:.3f}")
    print(f"  R²: {train_metrics['R2']:.4f}")
    
    results = {
        "type": "xgboost",
        "pollutant": pollutant,
        "params": params,
        "train_metrics": train_metrics,
        "feature_importance": importance_df,
        "feature_names": X_train_clean.columns.tolist(),
        "n_features": len(X_train_clean.columns),
    }
    
    # 验证集指标
    if X_val is not None and y_val is not None:
        y_val_pred = model.predict(X_val_clean)
        val_metrics = calculate_metrics(y_val, y_val_pred)
        results["val_metrics"] = val_metrics
        print(f"\n验证集指标:")
        print(f"  MAE: {val_metrics['MAE']:.3f}")
        print(f"  RMSE: {val_metrics['RMSE']:.3f}")
        print(f"  R²: {val_metrics['R2']:.4f}")
    
    return model, results


def train_xgboost_with_tuning(X_train, y_train, X_val=None, y_val=None,
                               pollutant="pm25"):
    """
    XGBoost + 简单超参数调优
    """
    if not HAS_XGB:
        return None, None
    
    print("\n" + "="*60)
    print(f"模型D：XGBoost + 超参数调优")
    print("="*60)
    
    # 参数网格
    param_grid = [
        {"max_depth": 3, "n_estimators": 50, "learning_rate": 0.1},
        {"max_depth": 5, "n_estimators": 100, "learning_rate": 0.1},
        {"max_depth": 7, "n_estimators": 100, "learning_rate": 0.1},
        {"max_depth": 5, "n_estimators": 200, "learning_rate": 0.05},
    ]
    
    # 填补缺失值
    X_train_clean = X_train.copy()
    fill_values = {}
    for col in X_train_clean.columns:
        if X_train_clean[col].isnull().any():
            fill_values[col] = X_train_clean[col].median()
            X_train_clean[col] = X_train_clean[col].fillna(fill_values[col])
    
    if X_val is not None:
        X_val_clean = X_val.copy()
        for col in X_val_clean.columns:
            if col in fill_values:
                X_val_clean[col] = X_val_clean[col].fillna(fill_values[col])
    else:
        X_val_clean = None
    
    best_val_rmse = np.inf
    best_params = None
    best_model = None
    
    for i, params in enumerate(param_grid):
        params["random_state"] = 42
        params["n_jobs"] = -1
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_clean, y_train)
        
        if X_val_clean is not None and y_val is not None:
            y_val_pred = model.predict(X_val_clean)
            val_metrics = calculate_metrics(y_val, y_val_pred)
            val_rmse = val_metrics["RMSE"]
            
            print(f"  配置 {i+1}: max_depth={params['max_depth']}, "
                  f"n_estimators={params['n_estimators']}, "
                  f"RMSE={val_rmse:.3f}")
            
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_params = params.copy()
                best_model = model
    
    if best_params:
        print(f"\n最优参数: {best_params}")
    
    # 计算最终指标
    y_train_pred = best_model.predict(X_train_clean)
    train_metrics = calculate_metrics(y_train, y_train_pred)
    
    results = {
        "type": "xgboost_tuned",
        "pollutant": pollutant,
        "params": best_params,
        "train_metrics": train_metrics,
        "feature_names": X_train_clean.columns.tolist(),
    }
    
    if X_val_clean is not None and y_val is not None:
        y_val_pred = best_model.predict(X_val_clean)
        val_metrics = calculate_metrics(y_val, y_val_pred)
        results["val_metrics"] = val_metrics
    
    return best_model, results


def compare_linear_vs_tree(val_results_linear, val_results_xgb):
    """
    比较线性模型和树模型的验证集表现
    """
    print("\n" + "="*60)
    print("线性模型 vs XGBoost 对比")
    print("="*60)
    
    metrics_table = []
    
    # 线性模型结果
    if val_results_linear:
        metrics_table.append({
            "模型": "线性动态模型",
            "MAE": val_results_linear.get("MAE", np.nan),
            "RMSE": val_results_linear.get("RMSE", np.nan),
            "R²": val_results_linear.get("R2", np.nan),
        })
    
    # XGBoost 结果
    if val_results_xgb:
        metrics_table.append({
            "模型": "XGBoost",
            "MAE": val_results_xgb.get("MAE", np.nan),
            "RMSE": val_results_xgb.get("RMSE", np.nan),
            "R²": val_results_xgb.get("R2", np.nan),
        })
    
    df = pd.DataFrame(metrics_table)
    print(df.to_string(index=False))
    
    # 计算改善
    if val_results_linear and val_results_xgb:
        mae_improvement = (val_results_linear["MAE"] - val_results_xgb["MAE"]) / val_results_linear["MAE"] * 100
        rmse_improvement = (val_results_linear["RMSE"] - val_results_xgb["RMSE"]) / val_results_linear["RMSE"] * 100
        
        print(f"\n非线性改进空间:")
        print(f"  MAE 改善: {mae_improvement:+.2f}% "
              f"({'XGBoost更优' if mae_improvement > 0 else '线性模型更优'})")
        print(f"  RMSE 改善: {rmse_improvement:+.2f}% "
              f"({'XGBoost更优' if rmse_improvement > 0 else '线性模型更优'})")
        
        return mae_improvement, rmse_improvement
    
    return None, None
