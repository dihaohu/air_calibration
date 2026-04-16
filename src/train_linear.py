"""
基线线性校准模型：模型A（一元线性）+ 模型B（多元静态线性）

模型A：一元线性校准
    Y_p(t) = a + b * x_p^near(t)

模型B：多元静态线性校准
    Y_p(t) = β0 + β1*x_p^near + β2*x_p^mean + β3*x_p^std + β4*x_p^slope
            + Σ δ_r * w_r(t) + φ1*sin_hour + φ2*cos_hour

包含OLS、Ridge、Lasso、Huber四种回归器
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def train_linear_baseline(X_train, y_train, X_val=None, y_val=None):
    """
    模型A：一元线性校准
    Y_p(t) = a + b * x_p^near(t)
    
    Parameters:
    -----------
    X_train : DataFrame
    y_train : Series
    X_val : DataFrame, optional
    y_val : Series, optional
    
    Returns:
    --------
    model : LinearRegression
    scaler : StandardScaler (None for baseline)
    results : dict
    """
    print("\n" + "="*60)
    print("模型A：一元线性校准")
    print("Y_p(t) = a + b * x_p^near(t)")
    print("="*60)
    
    # 只用近邻值
    feature_name = X_train.columns[0]  # x_p_near
    X_train_simple = X_train[[feature_name]].values
    X_val_simple = X_val[[feature_name]].values if X_val is not None else None
    
    # 训练
    model = LinearRegression()
    model.fit(X_train_simple, y_train)
    
    # 系数解读
    a, b = model.intercept_, model.coef_[0]
    print(f"\n模型参数:")
    print(f"  截距 (a/零点偏移): {a:.4f}")
    print(f"  斜率 (b/灵敏度): {b:.4f}")
    
    if abs(a) > 1:
        print(f"  → 存在{'正' if a > 0 else '负'}零点偏差: {a:.2f} μg/m³")
    if abs(b - 1) > 0.05:
        print(f"  → 存在{'放大' if b > 1 else '缩小'}效应: 灵敏度为 {b:.4f}")
    
    # 训练集预测
    y_train_pred = model.predict(X_train_simple)
    
    # 验证集预测
    if X_val is not None:
        y_val_pred = model.predict(X_val_simple)
    else:
        y_val_pred = None
    
    # 计算指标
    train_metrics = calculate_regression_metrics(y_train, y_train_pred)
    
    print(f"\n训练集指标:")
    print(f"  MAE: {train_metrics['MAE']:.3f}")
    print(f"  RMSE: {train_metrics['RMSE']:.3f}")
    print(f"  R²: {train_metrics['R2']:.4f}")
    
    results = {
        "type": "baseline",
        "intercept": a,
        "slope": b,
        "train_metrics": train_metrics,
        "feature_used": feature_name,
    }
    
    if X_val is not None:
        val_metrics = calculate_regression_metrics(y_val, y_val_pred)
        results["val_metrics"] = val_metrics
        print(f"\n验证集指标:")
        print(f"  MAE: {val_metrics['MAE']:.3f}")
        print(f"  RMSE: {val_metrics['RMSE']:.3f}")
        print(f"  R²: {val_metrics['R2']:.4f}")
    
    return model, None, results


def train_multivariate_static(X_train, y_train, X_val=None, y_val=None,
                               model_type="ridge"):
    """
    模型B：多元静态线性校准
    Y_p(t) = β0 + β1*x_p^near + β2*x_p^mean + β3*x_p^std + β4*x_p^slope
            + Σ δ_r * w_r(t) + φ1*sin_hour + φ2*cos_hour
    
    Parameters:
    -----------
    X_train, y_train : 训练数据
    X_val, y_val : 验证数据
    model_type : str, "ols", "ridge", "lasso", "huber"
    
    Returns:
    --------
    model : 训练好的模型
    scaler : StandardScaler
    results : dict
    """
    print("\n" + "="*60)
    print(f"模型B：多元静态线性校准 ({model_type.upper()})")
    print("="*60)
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
    else:
        X_val_scaled = None
    
    # 选择模型
    if model_type == "ols":
        model = LinearRegression()
    elif model_type == "ridge":
        model = Ridge(alpha=1.0)
    elif model_type == "lasso":
        model = Lasso(alpha=0.1, max_iter=5000)
    elif model_type == "huber":
        model = HuberRegressor(epsilon=1.35, max_iter=500)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # 训练
    model.fit(X_train_scaled, y_train)
    
    # 系数解读（还原到原始尺度）
    coefs_original = model.coef_ / scaler.scale_
    intercept_original = model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_)
    
    print(f"\n模型系数（原始尺度）:")
    print(f"  截距: {intercept_original:.4f}")
    
    # 分类打印系数
    print_coefficients(X_train.columns.tolist(), coefs_original)
    
    # 训练集预测
    y_train_pred = model.predict(X_train_scaled)
    train_metrics = calculate_regression_metrics(y_train, y_train_pred)
    
    print(f"\n训练集指标:")
    print(f"  MAE: {train_metrics['MAE']:.3f}")
    print(f"  RMSE: {train_metrics['RMSE']:.3f}")
    print(f"  R²: {train_metrics['R2']:.4f}")
    
    results = {
        "type": "multivariate_static",
        "model_type": model_type,
        "intercept": intercept_original,
        "coefs": dict(zip(X_train.columns.tolist(), coefs_original)),
        "train_metrics": train_metrics,
        "feature_names": X_train.columns.tolist(),
    }
    
    if X_val is not None:
        y_val_pred = model.predict(X_val_scaled)
        val_metrics = calculate_regression_metrics(y_val, y_val_pred)
        results["val_metrics"] = val_metrics
        print(f"\n验证集指标:")
        print(f"  MAE: {val_metrics['MAE']:.3f}")
        print(f"  RMSE: {val_metrics['RMSE']:.3f}")
        print(f"  R²: {val_metrics['R2']:.4f}")
    
    return model, scaler, results


def train_all_linear_models(X_train, y_train, X_val=None, y_val=None,
                              pollutant_name="PM2.5"):
    """
    训练所有线性模型（A和B）
    
    Returns:
    --------
    dict : {model_name: (model, scaler, results)}
    """
    from .build_features import get_feature_set_1, get_feature_set_2
    
    all_results = {}
    
    # 模型A：一元线性
    print(f"\n{'#'*60}")
    print(f"## {pollutant_name} - 训练线性模型")
    print(f"{'#'*60}")
    
    # 获取特征集1（只用近邻）
    features_1 = get_feature_set_1(pollutant_name.lower().replace(".", ""))
    if len(features_1) > 0 and features_1[0] in X_train.columns:
        model_a, scaler_a, results_a = train_linear_baseline(
            X_train[features_1], y_train, 
            X_val[features_1] if X_val is not None else None,
            y_val
        )
        all_results["A_baseline"] = (model_a, scaler_a, results_a)
    else:
        print(f"  警告: 特征 {features_1} 不在数据中，跳过模型A")
    
    # 模型B：多元静态（尝试多种回归器）
    features_2 = get_feature_set_2(pollutant_name.lower().replace(".", ""))
    available_features_2 = [f for f in features_2 if f in X_train.columns]
    
    if len(available_features_2) > 0:
        X_train_b = X_train[available_features_2]
        X_val_b = X_val[available_features_2] if X_val is not None else None
        
        for model_type in ["ridge", "lasso", "huber"]:
            try:
                model, scaler, results = train_multivariate_static(
                    X_train_b, y_train, X_val_b, y_val, model_type
                )
                all_results[f"B_static_{model_type}"] = (model, scaler, results)
            except Exception as e:
                print(f"  警告: {model_type} 训练失败 - {e}")
    
    return all_results


def print_coefficients(feature_names, coefficients, top_n=15):
    """打印主要系数"""
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefficients,
    })
    coef_df["abs_coef"] = np.abs(coef_df["coef"])
    coef_df = coef_df.sort_values("abs_coef", ascending=False)
    
    print(f"\n主要系数（绝对值排序，前{top_n}个）:")
    for _, row in coef_df.head(top_n).iterrows():
        direction = "+" if row["coef"] > 0 else "-"
        significance = "***" if abs(row["coef"]) > 0.5 else "**" if abs(row["coef"]) > 0.2 else "*" if abs(row["coef"]) > 0.1 else ""
        print(f"  {row['feature']:<30} {direction}{abs(row['coef']):>8.4f} {significance}")


def calculate_regression_metrics(y_true, y_pred):
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


def select_best_linear_model(all_results, metric="RMSE"):
    """
    从多个模型中选择最优的
    
    Parameters:
    -----------
    all_results : dict
    metric : str, "RMSE" or "MAE"
    
    Returns:
    --------
    best_model_name, best_results
    """
    best_name = None
    best_score = np.inf
    best_results = None
    
    for name, (_, _, results) in all_results.items():
        if "val_metrics" in results:
            score = results["val_metrics"][metric]
            if score < best_score:
                best_score = score
                best_name = name
                best_results = results
    
    if best_name:
        print(f"\n最优线性模型: {best_name}")
        print(f"  验证集 {metric}: {best_score:.3f}")
    
    return best_name, best_results
