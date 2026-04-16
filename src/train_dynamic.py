"""
动态校准主模型：模型C（动态多元回归）

模型C：动态线性主模型
    Y_p(t) = β0 + β1*x_p^near + β2*x_p^mean + β3*x_p^std + β4*x_p^slope
            + Σ γ_q * x_q^near (其他污染物) + Σ δ_r * w_r(t) (气象)
            + θ1*u(t) (零点漂移) + θ2*u(t)*x_p^near (量程漂移)
            + φ1*sin(2πh/24) + φ2*cos(2πh/24) (日周期)

这是主模型，用于捕捉：
1. 零点漂移：设备基线随时间变化
2. 量程漂移：灵敏度随时间变化
3. 交叉干扰：其他污染物对测量的影响
4. 气象修正：温度、湿度等对测量的影响
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, HuberRegressor, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def add_drift_features(df, pollutant):
    """
    添加漂移特征到DataFrame
    
    Parameters:
    -----------
    df : DataFrame
    pollutant : str
    
    Returns:
    --------
    df : DataFrame with drift features added
    """
    df = df.copy()
    
    # 确保 drift_idx 存在
    if "drift_idx" not in df.columns:
        if "time" in df.columns:
            t0 = df["time"].min()
            T = (df["time"].max() - t0).total_seconds()
            df["drift_idx"] = (df["time"] - t0).dt.total_seconds() / T
        else:
            df["drift_idx"] = np.linspace(0, 1, len(df))
    
    # 量程漂移交互项：drift_idx * x_p_near
    near_col = f"x_{pollutant}_near"
    if near_col in df.columns:
        df[f"drift_x_{pollutant}_near"] = df["drift_idx"] * df[near_col]
    else:
        # 尝试其他命名
        for col in df.columns:
            if pollutant in col and "near" in col:
                df[f"drift_x_{pollutant}_near"] = df["drift_idx"] * df[col]
                break
    
    return df


def get_drift_features(pollutant, all_cols):
    """
    获取漂移相关特征列表
    """
    features = []
    
    # 漂移索引
    if "drift_idx" in all_cols:
        features.append("drift_idx")
    
    # 量程漂移交互项
    for col in all_cols:
        if f"drift_x_{pollutant}" in col or (pollutant in col and "near" in col and "drift" in col):
            features.append(col)
    
    return features


def train_dynamic_regression(X_train, y_train, X_val=None, y_val=None,
                              pollutant="pm25", model_type="ridge"):
    """
    模型C：动态校准主模型
    
    Parameters:
    -----------
    X_train, y_train : 训练数据（已包含漂移特征）
    X_val, y_val : 验证数据
    pollutant : str, 污染物名称（用于识别漂移特征）
    model_type : str, "ridge", "lasso", "huber", "elasticnet"
    
    Returns:
    --------
    model, scaler, results
    """
    print("\n" + "="*60)
    print(f"模型C：动态校准主模型 ({model_type.upper()})")
    print("="*60)
    print(f"特征数量: {len(X_train.columns)}")
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
    else:
        X_val_scaled = None
    
    # 选择模型
    if model_type == "ridge":
        model = Ridge(alpha=1.0)
    elif model_type == "lasso":
        model = Lasso(alpha=0.01, max_iter=10000)
    elif model_type == "huber":
        model = HuberRegressor(epsilon=1.35, max_iter=1000)
    elif model_type == "elasticnet":
        model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # 训练
    model.fit(X_train_scaled, y_train)
    
    # 还原系数到原始尺度
    coefs_original = model.coef_ / scaler.scale_
    intercept_original = model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_)
    
    print(f"\n模型参数:")
    print(f"  截距: {intercept_original:.4f}")
    print(f"  正则化系数 alpha: {model.alpha if hasattr(model, 'alpha') else 'N/A'}")
    
    # 打印系数
    print_coefficients_by_category(X_train.columns.tolist(), coefs_original)
    
    # 训练集指标
    y_train_pred = model.predict(X_train_scaled)
    train_metrics = calculate_metrics(y_train, y_train_pred)
    
    print(f"\n训练集指标:")
    print(f"  MAE: {train_metrics['MAE']:.3f}")
    print(f"  RMSE: {train_metrics['RMSE']:.3f}")
    print(f"  R²: {train_metrics['R2']:.4f}")
    
    results = {
        "type": "dynamic",
        "model_type": model_type,
        "pollutant": pollutant,
        "intercept": intercept_original,
        "coefs": dict(zip(X_train.columns.tolist(), coefs_original)),
        "train_metrics": train_metrics,
        "feature_names": X_train.columns.tolist(),
        "n_features": len(X_train.columns),
    }
    
    # 验证集指标
    if X_val is not None:
        y_val_pred = model.predict(X_val_scaled)
        val_metrics = calculate_metrics(y_val, y_val_pred)
        results["val_metrics"] = val_metrics
        print(f"\n验证集指标:")
        print(f"  MAE: {val_metrics['MAE']:.3f}")
        print(f"  RMSE: {val_metrics['RMSE']:.3f}")
        print(f"  R²: {val_metrics['R2']:.4f}")
        
        # 漂移项显著性检查
        drift_coefs = {k: v for k, v in results["coefs"].items() 
                      if "drift" in k.lower()}
        if drift_coefs:
            print(f"\n漂移项系数检验:")
            for feat, coef in drift_coefs.items():
                sig = "***" if abs(coef) > 0.5 else "**" if abs(coef) > 0.2 else "*" if abs(coef) > 0.05 else ""
                print(f"  {feat:<25} {coef:>+8.4f} {sig}")
    
    return model, scaler, results


def train_dynamic_with_tuning(X_train, y_train, X_val=None, y_val=None,
                               pollutant="pm25", alphas=None):
    """
    使用不同的正则化系数训练动态模型，选择最优
    
    Parameters:
    -----------
    X_train, y_train : 训练数据
    X_val, y_val : 验证数据
    pollutant : str
    alphas : list, 正则化系数候选值
    
    Returns:
    --------
    best_model, best_scaler, best_results
    """
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    
    print("\n" + "="*60)
    print(f"模型C：动态校准 + 超参数调优")
    print("="*60)
    
    best_alpha = None
    best_val_rmse = np.inf
    best_model = None
    best_scaler = None
    best_results = None
    
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)
        
        if X_val is not None:
            X_val_scaled = scaler.transform(X_val)
            y_val_pred = model.predict(X_val_scaled)
            val_metrics = calculate_metrics(y_val, y_val_pred)
            val_rmse = val_metrics["RMSE"]
            
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_alpha = alpha
                best_model = model
                best_scaler = scaler
    
    # 用最优alpha重新训练
    print(f"\n最优正则化系数: alpha = {best_alpha}")
    model = Ridge(alpha=best_alpha)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model.fit(X_train_scaled, y_train)
    
    # 还原系数
    coefs_original = model.coef_ / scaler.scale_
    intercept_original = model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_)
    
    print(f"\n最优模型系数（原始尺度）:")
    print(f"  截距: {intercept_original:.4f}")
    print_coefficients_by_category(X_train.columns.tolist(), coefs_original)
    
    # 计算指标
    y_train_pred = model.predict(X_train_scaled)
    train_metrics = calculate_metrics(y_train, y_train_pred)
    
    results = {
        "type": "dynamic_tuned",
        "model_type": "ridge_tuned",
        "pollutant": pollutant,
        "alpha": best_alpha,
        "intercept": intercept_original,
        "coefs": dict(zip(X_train.columns.tolist(), coefs_original)),
        "train_metrics": train_metrics,
        "feature_names": X_train.columns.tolist(),
    }
    
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        y_val_pred = model.predict(X_val_scaled)
        val_metrics = calculate_metrics(y_val, y_val_pred)
        results["val_metrics"] = val_metrics
    
    return model, scaler, results


def print_coefficients_by_category(feature_names, coefficients, top_n=20):
    """按类别打印系数"""
    # 定义特征类别
    categories = {
        "本体响应": lambda f: "near" in f or "mean" in f or "std" in f or "slope" in f,
        "气象修正": lambda f: any(w in f for w in ["wind", "pressure", "rain", "temp", "rh"]),
        "日周期": lambda f: "hour" in f or "month" in f,
        "交叉干扰": lambda f: any(p in f for p in ["pm25", "pm10", "co", "no2", "so2", "o3"]) 
                       and "near" in f and f"pm25" not in f if "pm25" in f else True,
        "漂移项": lambda f: "drift" in f.lower(),
    }
    
    # 创建系数DataFrame
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefficients,
    })
    coef_df["abs_coef"] = np.abs(coef_df["coef"])
    
    # 分类
    def get_category(feat):
        for cat, check_fn in categories.items():
            try:
                if check_fn(feat):
                    return cat
            except:
                pass
        return "其他"
    
    coef_df["category"] = coef_df["feature"].apply(get_category)
    coef_df = coef_df.sort_values("abs_coef", ascending=False)
    
    # 按类别打印
    print("\n系数详情（按类别分组）:")
    
    for cat in ["本体响应", "气象修正", "交叉干扰", "漂移项", "日周期", "其他"]:
        cat_df = coef_df[coef_df["category"] == cat]
        if len(cat_df) > 0:
            print(f"\n  [{cat}]")
            for _, row in cat_df.head(5).iterrows():
                direction = "+" if row["coef"] > 0 else "-"
                sig = "***" if row["abs_coef"] > 0.5 else "**" if row["abs_coef"] > 0.2 else "*" if row["abs_coef"] > 0.1 else ""
                print(f"    {row['feature']:<28} {direction}{row['abs_coef']:>8.4f} {sig}")


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


def interpret_drift_coefficients(results):
    """
    解释漂移系数的物理意义
    """
    if "coefs" not in results:
        return
    
    coefs = results["coefs"]
    
    print("\n" + "="*50)
    print("漂移项解释")
    print("="*50)
    
    # 零点漂移
    if "drift_idx" in coefs:
        drift_coef = coefs["drift_idx"]
        print(f"\n零点漂移项 (drift_idx): {drift_coef:+.4f}")
        if abs(drift_coef) > 1:
            print(f"  → 设备基线有{'显著上升' if drift_coef > 0 else '显著下降'}趋势")
            print(f"  → 在整个监测周期内，基线漂移约 {drift_coef:.2f} μg/m³")
        else:
            print(f"  → 基线相对稳定")
    
    # 量程漂移
    range_drift_cols = [k for k in coefs.keys() if "drift" in k.lower() and "near" in k.lower()]
    if range_drift_cols:
        for col in range_drift_cols:
            coef = coefs[col]
            print(f"\n量程漂移项 ({col}): {coef:+.4f}")
            if abs(coef) > 0.1:
                print(f"  → 灵敏度随时间有{'放大' if coef > 0 else '衰减'}趋势")
                print(f"  → 斜率变化率: {coef:.4f}（每单位设备老化，灵敏度变化）")
    
    # 交叉干扰
    pollutant_short = results.get("pollutant", "")
    cross_cols = [k for k in coefs.keys() 
                  if any(p in k for p in ["pm25", "pm10", "co", "no2", "so2", "o3"])
                  and "near" in k and pollutant_short not in k]
    
    if cross_cols:
        print(f"\n交叉干扰项 (其他污染物):")
        for col in cross_cols[:3]:  # 只显示前3个
            coef = coefs[col]
            if abs(coef) > 0.05:
                print(f"  {col}: {coef:+.4f}")
                if coef > 0:
                    print(f"    → 正向干扰：该污染物增加时测量值偏高")
                else:
                    print(f"    → 负向干扰：该污染物增加时测量值偏低")
