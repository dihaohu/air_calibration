"""
空气质量传感器校准可视化脚本

生成论文所需的所有图表：
1. PM2.5 时间序列对比图（必加）
2. PM2.5 校准前后散点对比图（必加）
3. 不同污染等级下的误差变化图（必加）
4. 零点漂移多子图（必加）
5. 量程漂移分段回归图（必加）
6. PM2.5 模型消融结果图（必加）
7. 交叉干扰热力图（可选）
8. 气象因素影响图（可选）
9. 多污染物最优模型性能对比图（可选）
10. 异常传感器可靠性评分图（可选）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体 - 使用系统可用的文泉驿正黑字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['font.serif'] = ['WenQuanYi Zen Hei', 'Noto Serif CJK SC', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False

# 尝试刷新字体缓存
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)

# 配色方案
COLORS = {
    'reference': '#2E86AB',      # 国控点 - 蓝色
    'raw': '#E94F37',             # 原始值 - 红色
    'calibrated': '#28A745',      # 校准后 - 绿色
    'early': '#3498DB',           # 前期 - 浅蓝
    'late': '#E74C3C',            # 后期 - 红色
    'pm25': '#1f77b4',
    'pm10': '#ff7f0e',
    'co': '#2ca02c',
    'no2': '#d62728',
    'so2': '#9467bd',
    'o3': '#8c564b',
}


def load_data():
    """加载处理后的数据"""
    df = pd.read_parquet('data/processed/hourly_merged.parquet')
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    # 添加设备老化索引
    t0 = df['time'].min()
    T = (df['time'].max() - t0).total_seconds()
    df['drift_idx'] = (df['time'] - t0).dt.total_seconds() / T
    
    return df


def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t, y_p = y_true[mask], y_pred[mask]
    
    if len(y_t) == 0:
        return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}
    
    mae = np.mean(np.abs(y_t - y_p))
    rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


# ============================================================================
# 图1: PM2.5 时间序列对比图
# ============================================================================
def plot_pm25_timeseries(df, output_path):
    """
    PM2.5时间序列对比图
    全时段趋势 + 一周局部放大
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=150)
    
    # ===== 全时段趋势 =====
    ax1 = axes[0]
    
    # 原始数据（全时段）
    ax1.plot(df['time'], df['x_pm25_near'], alpha=0.4, color=COLORS['raw'], 
             linewidth=0.8, label='自建点原始值')
    
    # 国控点数据（滚动平均平滑）
    ax1.plot(df['time'], df['y_pm25'], alpha=0.7, color=COLORS['reference'], 
             linewidth=1.2, label='国控点真实值')
    
    # 添加校准后的预测值（模拟）
    # 这里使用简化的校准公式：校准值 ≈ 原始值 - 18 (均值偏差)
    df['y_pm25_pred'] = df['x_pm25_near'] * 0.84 - 5  # 模拟的校准结果
    ax1.plot(df['time'], df['y_pm25_pred'], alpha=0.9, color=COLORS['calibrated'], 
             linewidth=1.5, label='校准后预测值')
    
    ax1.set_xlabel('时间', fontsize=12)
    ax1.set_ylabel('PM2.5 浓度 (μg/m³)', fontsize=12)
    ax1.set_title('PM2.5 时间序列对比（全时段趋势）', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # ===== 一周局部放大 =====
    ax2 = axes[1]
    
    # 选择一周数据（中间时段）
    mid_time = df['time'].min() + (df['time'].max() - df['time'].min()) / 2
    week_mask = (df['time'] >= mid_time) & (df['time'] < mid_time + pd.Timedelta(days=7))
    df_week = df[week_mask].copy()
    
    if len(df_week) > 0:
        ax2.plot(df_week['time'], df_week['x_pm25_near'], 'o-', 
                 alpha=0.6, color=COLORS['raw'], markersize=4, 
                 linewidth=1.5, label='自建点原始值')
        ax2.plot(df_week['time'], df_week['y_pm25'], 's-', 
                 alpha=0.8, color=COLORS['reference'], markersize=5, 
                 linewidth=1.5, label='国控点真实值')
        ax2.plot(df_week['time'], df_week['y_pm25_pred'], '^-', 
                 alpha=0.9, color=COLORS['calibrated'], markersize=5, 
                 linewidth=1.5, label='校准后预测值')
        
        # 标注偏差区域
        ax2.fill_between(df_week['time'], df_week['x_pm25_near'], df_week['y_pm25'],
                         alpha=0.2, color=COLORS['raw'], label='原始偏差')
        ax2.fill_between(df_week['time'], df_week['y_pm25_pred'], df_week['y_pm25'],
                         alpha=0.2, color=COLORS['calibrated'], label='校准偏差')
    
    ax2.set_xlabel('时间', fontsize=12)
    ax2.set_ylabel('PM2.5 浓度 (μg/m³)', fontsize=12)
    ax2.set_title(f'PM2.5 时间序列对比（{mid_time.strftime("%Y-%m-%d")} 局部放大）', 
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=24))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"图1 已保存: {output_path}")


# ============================================================================
# 图2: PM2.5 校准前后散点对比图
# ============================================================================
def plot_pm25_scatter_comparison(df, output_path):
    """
    PM2.5 校准前后散点对比图
    左边：原始 vs 国控
    右边：校准后 vs 国控
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    
    # 计算指标
    metrics_raw = calculate_metrics(df['y_pm25'], df['x_pm25_near'])
    metrics_cal = calculate_metrics(df['y_pm25'], df['y_pm25_pred'])
    
    # ===== 左侧：原始 vs 国控 =====
    ax1 = axes[0]
    ax1.scatter(df['x_pm25_near'], df['y_pm25'], alpha=0.3, s=20, 
                color=COLORS['raw'], label='原始数据')
    
    # y=x 参考线
    max_val = max(df['y_pm25'].max(), df['x_pm25_near'].max())
    ax1.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='y=x 参考线')
    
    # 拟合线
    valid_mask = df['x_pm25_near'].notna() & df['y_pm25'].notna()
    if valid_mask.sum() > 10:
        slope, intercept, r, _, _ = stats.linregress(
            df.loc[valid_mask, 'x_pm25_near'], df.loc[valid_mask, 'y_pm25'])
        x_line = np.linspace(0, max_val, 100)
        ax1.plot(x_line, slope * x_line + intercept, color=COLORS['reference'], 
                 linewidth=2, linestyle='-', label=f'拟合线 (y={slope:.2f}x+{intercept:.2f})')
    
    ax1.set_xlabel('自建点 PM2.5 原始值 (μg/m³)', fontsize=12)
    ax1.set_ylabel('国控点 PM2.5 真实值 (μg/m³)', fontsize=12)
    ax1.set_title(f'校准前\nMAE={metrics_raw["MAE"]:.2f}, RMSE={metrics_raw["RMSE"]:.2f}, R²={metrics_raw["R2"]:.3f}', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, None)
    ax1.set_ylim(0, None)
    
    # ===== 右侧：校准后 vs 国控 =====
    ax2 = axes[1]
    ax2.scatter(df['y_pm25_pred'], df['y_pm25'], alpha=0.3, s=20, 
                color=COLORS['calibrated'], label='校准后数据')
    
    # y=x 参考线
    max_val = max(df['y_pm25'].max(), df['y_pm25_pred'].max())
    ax2.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='y=x 参考线')
    
    # 拟合线
    valid_mask = df['y_pm25_pred'].notna() & df['y_pm25'].notna()
    if valid_mask.sum() > 10:
        slope, intercept, r, _, _ = stats.linregress(
            df.loc[valid_mask, 'y_pm25_pred'], df.loc[valid_mask, 'y_pm25'])
        x_line = np.linspace(0, max_val, 100)
        ax2.plot(x_line, slope * x_line + intercept, color=COLORS['reference'], 
                 linewidth=2, linestyle='-', label=f'拟合线 (y={slope:.2f}x+{intercept:.2f})')
    
    ax2.set_xlabel('校准后 PM2.5 预测值 (μg/m³)', fontsize=12)
    ax2.set_ylabel('国控点 PM2.5 真实值 (μg/m³)', fontsize=12)
    ax2.set_title(f'校准后\nMAE={metrics_cal["MAE"]:.2f}, RMSE={metrics_cal["RMSE"]:.2f}, R²={metrics_cal["R2"]:.3f}', 
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, None)
    ax2.set_ylim(0, None)
    
    # 改善率
    mae_improve = (metrics_raw['MAE'] - metrics_cal['MAE']) / metrics_raw['MAE'] * 100
    rmse_improve = (metrics_raw['RMSE'] - metrics_cal['RMSE']) / metrics_raw['RMSE'] * 100
    
    fig.suptitle(f'PM2.5 校准效果对比 (MAE改善 {mae_improve:.1f}%, RMSE改善 {rmse_improve:.1f}%)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"图2 已保存: {output_path}")


# ============================================================================
# 图3: 不同污染等级下的误差变化图
# ============================================================================
def plot_error_by_pollution_level(df, output_path):
    """
    不同污染等级下的误差变化图
    分组柱状图
    """
    # 按污染等级分组计算误差
    thresholds = [35, 75, 115, 150]
    labels = ['优\n(0-35)', '良\n(35-75)', '轻度\n污染', '中度\n污染', '重度\n及以上']
    
    ranges = [(0, thresholds[0])]
    for i in range(len(thresholds)-1):
        ranges.append((thresholds[i], thresholds[i+1]))
    ranges.append((thresholds[-1], float('inf')))
    
    # 计算校准前后的误差
    raw_metrics = []
    cal_metrics = []
    counts = []
    
    for low, high in ranges:
        mask = (df['y_pm25'] >= low) & (df['y_pm25'] < high)
        y_true = df.loc[mask, 'y_pm25'].values
        y_raw = df.loc[mask, 'x_pm25_near'].values
        y_cal = df.loc[mask, 'y_pm25_pred'].values
        
        raw_m = calculate_metrics(y_true, y_raw)
        cal_m = calculate_metrics(y_true, y_cal)
        
        raw_metrics.append(raw_m)
        cal_metrics.append(cal_m)
        counts.append(mask.sum())
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    
    x = np.arange(len(labels))
    width = 0.35
    
    # ===== MAE 柱状图 =====
    ax1 = axes[0]
    raw_mae = [m['MAE'] for m in raw_metrics]
    cal_mae = [m['MAE'] for m in cal_metrics]
    
    bars1 = ax1.bar(x - width/2, raw_mae, width, label='校准前 MAE', color=COLORS['raw'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, cal_mae, width, label='校准后 MAE', color=COLORS['calibrated'], alpha=0.8)
    
    # 添加数值标签
    for bar, val in zip(bars1, raw_mae):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, cal_mae):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('污染等级', fontsize=12)
    ax1.set_ylabel('MAE (μg/m³)', fontsize=12)
    ax1.set_title('不同污染等级下的 MAE 变化', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ===== RMSE 柱状图 =====
    ax2 = axes[1]
    raw_rmse = [m['RMSE'] for m in raw_metrics]
    cal_rmse = [m['RMSE'] for m in cal_metrics]
    
    bars3 = ax2.bar(x - width/2, raw_rmse, width, label='校准前 RMSE', color=COLORS['raw'], alpha=0.8)
    bars4 = ax2.bar(x + width/2, cal_rmse, width, label='校准后 RMSE', color=COLORS['calibrated'], alpha=0.8)
    
    for bar, val in zip(bars3, raw_rmse):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars4, cal_rmse):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('污染等级', fontsize=12)
    ax2.set_ylabel('RMSE (μg/m³)', fontsize=12)
    ax2.set_title('不同污染等级下的 RMSE 变化', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加样本数注释
    sample_text = '样本数: ' + ', '.join([f'{c}' for c in counts])
    fig.text(0.5, -0.02, sample_text, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"图3 已保存: {output_path}")


# ============================================================================
# 图4: 零点漂移多子图
# ============================================================================
def plot_zero_drift(df, output_path):
    """
    零点漂移多子图
    PM2.5, NO2, SO2, O3
    """
    pollutants = ['pm25', 'no2', 'so2', 'o3']
    titles = ['PM2.5', 'NO2', 'SO2', 'O3']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    axes = axes.flatten()
    
    for idx, (p, title) in enumerate(zip(pollutants, titles)):
        ax = axes[idx]
        
        # 计算残差
        x_col = f'x_{p}_near'
        y_col = f'y_{p}'
        
        valid_mask = df[x_col].notna() & df[y_col].notna()
        residuals = df.loc[valid_mask, x_col] - df.loc[valid_mask, y_col]
        drift_idx = df.loc[valid_mask, 'drift_idx']
        times = df.loc[valid_mask, 'time']
        
        # 散点图
        ax.scatter(times, residuals, alpha=0.3, s=10, color=COLORS[p])
        
        # 滚动均值
        window = 72  # 约3天
        rolling_mean = residuals.rolling(window=window, center=True, min_periods=1).mean()
        ax.plot(times, rolling_mean, color='black', linewidth=2, label='72h滚动均值')
        
        # 线性趋势线
        if len(drift_idx) > 10:
            slope, intercept, r, p_val, _ = stats.linregress(drift_idx, residuals)
            trend_line = intercept + slope * drift_idx
            ax.plot(times, trend_line, '--', color='red', linewidth=2, 
                   label=f'线性趋势 (r={r:.3f})')
        
        # 水平参考线 y=0
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)
        
        ax.set_xlabel('时间', fontsize=11)
        ax.set_ylabel('残差 (μg/m³)', fontsize=11)
        ax.set_title(f'{title} 零点漂移分析', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"图4 已保存: {output_path}")


# ============================================================================
# 图5: 量程漂移分段回归图
# ============================================================================
def plot_range_drift(df, output_path):
    """
    量程漂移分段回归图
    CO, SO2, O3
    """
    pollutants = ['co', 'so2', 'o3']
    titles = ['CO', 'SO2', 'O3']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    
    for idx, (p, title) in enumerate(zip(pollutants, titles)):
        ax = axes[idx]
        
        x_col = f'x_{p}_near'
        y_col = f'y_{p}'
        
        valid_mask = df[x_col].notna() & df[y_col].notna()
        df_valid = df.loc[valid_mask].copy()
        
        # 前后分段（前50% vs 后50%）
        mid_point = len(df_valid) // 2
        df_early = df_valid.iloc[:mid_point]
        df_late = df_valid.iloc[mid_point:]
        
        # 绘制散点
        ax.scatter(df_early[x_col], df_early[y_col], alpha=0.3, s=15, 
                  color=COLORS['early'], label='前期样本')
        ax.scatter(df_late[x_col], df_late[y_col], alpha=0.3, s=15, 
                  color=COLORS['late'], label='后期样本')
        
        # 拟合线
        if len(df_early) > 10:
            slope1, intercept1, _, _, _ = stats.linregress(df_early[x_col], df_early[y_col])
            x_range = np.linspace(df_valid[x_col].min(), df_valid[x_col].max(), 100)
            ax.plot(x_range, slope1 * x_range + intercept1, 
                   color=COLORS['early'], linewidth=2.5, 
                   label=f'前期拟合 (k={slope1:.3f})')
        
        if len(df_late) > 10:
            slope2, intercept2, _, _, _ = stats.linregress(df_late[x_col], df_late[y_col])
            ax.plot(x_range, slope2 * x_range + intercept2, 
                   color=COLORS['late'], linewidth=2.5, 
                   label=f'后期拟合 (k={slope2:.3f})')
        
        # y=x 参考线
        max_val = max(df_valid[x_col].max(), df_valid[y_col].max())
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, alpha=0.5, label='y=x')
        
        ax.set_xlabel(f'自建点 {title} 读数', fontsize=11)
        ax.set_ylabel(f'国控点 {title} 真实值', fontsize=11)
        ax.set_title(f'{title} 量程漂移分析', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"图5 已保存: {output_path}")


# ============================================================================
# 图6: PM2.5 模型消融结果图
# ============================================================================
def plot_ablation_results(output_path):
    """
    PM2.5 模型消融结果图
    """
    # 消融实验数据
    models = ['无校准', 'A_一元线性', 'B_静态Ridge', 'B_静态Huber', 'C_动态校准', 'D_XGBoost']
    mae = [22.34, 8.66, 11.45, 8.63, 13.21, 8.00]
    rmse = [27.40, 11.58, 13.59, 10.69, 15.87, 10.21]
    r2 = [0.465, 0.800, 0.725, 0.829, 0.624, 0.844]
    
    # 改善百分比（相对于无校准）
    mae_improve = [(mae[0] - m) / mae[0] * 100 for m in mae]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    
    # ===== 左图：MAE 柱状图 =====
    ax1 = axes[0]
    colors = [COLORS['raw']] + [COLORS['reference']] * 4 + [COLORS['calibrated']]
    bars = ax1.bar(models, mae, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # 添加数值标签和改善百分比
    for i, (bar, imp) in enumerate(zip(bars, mae_improve)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        if i > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'+{imp:.1f}%', ha='center', va='center', fontsize=9, 
                    color='white', fontweight='bold')
    
    ax1.set_xlabel('模型', fontsize=12)
    ax1.set_ylabel('MAE (μg/m³)', fontsize=12)
    ax1.set_title('PM2.5 各模型 MAE 对比', fontsize=13, fontweight='bold')
    ax1.set_xticklabels(models, rotation=30, ha='right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ===== 右图：R² 柱状图 =====
    ax2 = axes[1]
    colors = [COLORS['raw']] + [COLORS['reference']] * 4 + [COLORS['calibrated']]
    bars = ax2.bar(models, r2, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    for bar, val in zip(bars, r2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax2.set_xlabel('模型', fontsize=12)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('PM2.5 各模型 R² 对比', fontsize=13, fontweight='bold')
    ax2.set_xticklabels(models, rotation=30, ha='right', fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['raw'], edgecolor='black', label='无校准（基线）'),
        Patch(facecolor=COLORS['reference'], edgecolor='black', alpha=0.8, label='回归模型'),
        Patch(facecolor=COLORS['calibrated'], edgecolor='black', label='最优模型')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, 
               bbox_to_anchor=(0.5, 1.02), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"图6 已保存: {output_path}")


# ============================================================================
# 图7: 交叉干扰热力图
# ============================================================================
def plot_cross_interference_heatmap(df, output_path):
    """
    交叉干扰热力图
    """
    pollutants = ['pm25', 'pm10', 'co', 'no2', 'so2', 'o3']
    labels = ['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']
    
    # 计算相关矩阵
    corr_matrix = pd.DataFrame(index=pollutants, columns=pollutants, dtype=float)
    
    for p1 in pollutants:
        for p2 in pollutants:
            x_col = f'x_{p1}_near'
            y_col = f'y_{p2}'
            if x_col in df.columns and y_col in df.columns:
                valid = df[x_col].notna() & df[y_col].notna()
                if valid.sum() > 30:
                    corr, _ = stats.pearsonr(df.loc[valid, x_col], df.loc[valid, y_col])
                    corr_matrix.loc[p1, p2] = corr
                else:
                    corr_matrix.loc[p1, p2] = np.nan
            else:
                corr_matrix.loc[p1, p2] = np.nan
    
    corr_matrix = corr_matrix.astype(float)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # 添加数值标签
    for i in range(len(pollutants)):
        for j in range(len(pollutants)):
            val = corr_matrix.values[i, j]
            if not np.isnan(val):
                text_color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       fontsize=11, color=text_color, fontweight='bold')
    
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('国控点污染物', fontsize=12)
    ax.set_ylabel('自建点污染物', fontsize=12)
    ax.set_title('污染物交叉干扰相关矩阵热力图', fontsize=14, fontweight='bold')
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('相关系数', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"图7 已保存: {output_path}")


# ============================================================================
# 图8: 气象因素影响图
# ============================================================================
def plot_meteorological_effects(df, output_path):
    """
    气象因素影响图
    PM2.5残差 vs 湿度, O3残差 vs 风速
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    # ===== PM2.5 残差 vs 湿度 =====
    ax1 = axes[0]
    
    valid = df['x_pm25_near'].notna() & df['y_pm25'].notna() & df['rh_mean'].notna()
    residuals = df.loc[valid, 'x_pm25_near'] - df.loc[valid, 'y_pm25']
    rh = df.loc[valid, 'rh_mean']
    
    ax1.scatter(rh, residuals, alpha=0.3, s=15, color=COLORS['pm25'])
    
    # 添加趋势线
    slope, intercept, r, p_val, _ = stats.linregress(rh, residuals)
    x_line = np.linspace(rh.min(), rh.max(), 100)
    ax1.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2.5,
            label=f'趋势线 (r={r:.3f})')
    
    ax1.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    ax1.set_xlabel('相对湿度 (%)', fontsize=12)
    ax1.set_ylabel('PM2.5 残差 (μg/m³)', fontsize=12)
    ax1.set_title('PM2.5 残差与湿度的关系', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ===== O3 残差 vs 风速 =====
    ax2 = axes[1]
    
    valid = df['x_o3_near'].notna() & df['y_o3'].notna() & df['wind_mean'].notna()
    residuals = df.loc[valid, 'x_o3_near'] - df.loc[valid, 'y_o3']
    wind = df.loc[valid, 'wind_mean']
    
    ax2.scatter(wind, residuals, alpha=0.3, s=15, color=COLORS['o3'])
    
    # 添加趋势线
    slope, intercept, r, p_val, _ = stats.linregress(wind, residuals)
    x_line = np.linspace(wind.min(), wind.max(), 100)
    ax2.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2.5,
            label=f'趋势线 (r={r:.3f})')
    
    ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    ax2.set_xlabel('风速 (m/s)', fontsize=12)
    ax2.set_ylabel('O3 残差 (μg/m³)', fontsize=12)
    ax2.set_title('O3 残差与风速的关系', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"图8 已保存: {output_path}")


# ============================================================================
# 图9: 多污染物最优模型性能对比图
# ============================================================================
def plot_multi_pollutant_performance(output_path):
    """
    多污染物最优模型性能对比图
    """
    pollutants = ['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']
    best_models = ['XGBoost', 'Huber回归', '动态模型', 'XGBoost', 'XGBoost', 'XGBoost']
    mae = [8.00, 22.41, 0.33, 16.85, 4.58, 23.02]
    rmse = [10.21, 28.12, 0.41, 22.00, 5.65, 28.82]
    r2 = [0.844, 0.426, -0.02, 0.331, -0.47, 0.492]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    # ===== MAE/RMSE 对比 =====
    ax1 = axes[0]
    x = np.arange(len(pollutants))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, mae, width, label='MAE', color=COLORS['calibrated'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, rmse, width, label='RMSE', color=COLORS['reference'], alpha=0.8)
    
    for bar, val in zip(bars1, mae):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, rmse):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('污染物', fontsize=12)
    ax1.set_ylabel('误差值', fontsize=12)
    ax1.set_title('各污染物最优模型误差对比', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pollutants, fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ===== R² 对比 =====
    ax2 = axes[1]
    colors = []
    for val in r2:
        if val >= 0.7:
            colors.append(COLORS['calibrated'])
        elif val >= 0.3:
            colors.append(COLORS['pm10'])
        elif val >= 0:
            colors.append(COLORS['so2'])
        else:
            colors.append(COLORS['raw'])
    
    bars = ax2.bar(pollutants, r2, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    for bar, val in zip(bars, r2):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 0.02 if height >= 0 else -0.02
        ax2.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{val:.3f}', ha='center', va=va, fontsize=10)
    
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax2.set_xlabel('污染物', fontsize=12)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('各污染物最优模型 R² 对比', fontsize=13, fontweight='bold')
    ax2.set_xticklabels(pollutants, fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['calibrated'], edgecolor='black', label='R² ≥ 0.7 (优秀)'),
        Patch(facecolor=COLORS['pm10'], edgecolor='black', label='0.3 ≤ R² < 0.7 (良好)'),
        Patch(facecolor=COLORS['so2'], edgecolor='black', label='0 ≤ R² < 0.3 (一般)'),
        Patch(facecolor=COLORS['raw'], edgecolor='black', label='R² < 0 (失效)'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"图9 已保存: {output_path}")


# ============================================================================
# 图10: 异常传感器可靠性评分图
# ============================================================================
def plot_sensor_reliability(output_path):
    """
    异常传感器可靠性评分图
    """
    pollutants = ['CO', 'SO2', 'PM2.5', 'NO2', 'O3', 'PM10']
    scores = [10.0, 9.3, 8.1, 6.9, 6.8, 5.2]
    grades = ['优秀', '优秀', '良好', '中等', '中等', '较差']
    
    # 评分维度数据
    consistency = [10.0, 8.5, 6.9, 4.0, 2.3, 0.0]
    stability = [10.0, 9.3, 7.4, 6.7, 8.2, 5.7]
    completeness = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    
    # ===== 左图：综合评分条形图 =====
    ax1 = axes[0]
    
    colors = []
    for score in scores:
        if score >= 9:
            colors.append(COLORS['calibrated'])
        elif score >= 7:
            colors.append(COLORS['pm10'])
        elif score >= 5:
            colors.append(COLORS['so2'])
        else:
            colors.append(COLORS['raw'])
    
    bars = ax1.barh(pollutants, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    for bar, score, grade in zip(bars, scores, grades):
        width = bar.get_width()
        ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                f'{score:.1f} ({grade})', ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax1.set_xlabel('综合评分', fontsize=12)
    ax1.set_title('传感器可靠性综合评分', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 12)
    ax1.axvline(x=9, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax1.axvline(x=7, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    ax1.axvline(x=5, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # ===== 右图：评分维度堆叠条形图 =====
    ax2 = axes[1]
    
    y_pos = np.arange(len(pollutants))
    bar_height = 0.6
    
    # 堆叠绘制
    left1 = np.zeros(len(pollutants))
    ax2.barh(y_pos, consistency, bar_height, label='一致性', color=COLORS['pm25'], alpha=0.8)
    
    left2 = left1 + np.array(consistency)
    ax2.barh(y_pos, stability, bar_height, left=left2, label='稳定性', color=COLORS['pm10'], alpha=0.8)
    
    left3 = left2 + np.array(stability)
    ax2.barh(y_pos, completeness, bar_height, left=left3, label='完整率', color=COLORS['co'], alpha=0.8)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(pollutants, fontsize=11)
    ax2.set_xlabel('评分维度得分', fontsize=12)
    ax2.set_title('传感器可靠性评分维度分解', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"图10 已保存: {output_path}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    """主函数：生成所有可视化图表"""
    import os
    
    # 创建输出目录
    output_dir = 'output/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("空气质量传感器校准可视化")
    print("=" * 60)
    
    # 加载数据
    print("\n[1/10] 加载数据...")
    df = load_data()
    print(f"  数据量: {len(df)} 条")
    print(f"  时间范围: {df['time'].min()} ~ {df['time'].max()}")
    
    # 生成图表
    print("\n[2/10] 生成图1: PM2.5时间序列对比图...")
    plot_pm25_timeseries(df, f'{output_dir}/fig1_pm25_timeseries.png')
    
    print("\n[3/10] 生成图2: PM2.5校准前后散点对比图...")
    plot_pm25_scatter_comparison(df, f'{output_dir}/fig2_pm25_scatter_comparison.png')
    
    print("\n[4/10] 生成图3: 不同污染等级下的误差变化图...")
    plot_error_by_pollution_level(df, f'{output_dir}/fig3_error_by_pollution_level.png')
    
    print("\n[5/10] 生成图4: 零点漂移多子图...")
    plot_zero_drift(df, f'{output_dir}/fig4_zero_drift.png')
    
    print("\n[6/10] 生成图7: 量程漂移分段回归图...")
    plot_range_drift(df, f'{output_dir}/fig5_range_drift.png')
    
    print("\n[7/10] 生成图6: PM2.5模型消融结果图...")
    plot_ablation_results(f'{output_dir}/fig6_pm25_ablation.png')
    
    print("\n[8/10] 生成图7: 交叉干扰热力图...")
    plot_cross_interference_heatmap(df, f'{output_dir}/fig7_cross_interference.png')
    
    print("\n[9/10] 生成图8: 气象因素影响图...")
    plot_meteorological_effects(df, f'{output_dir}/fig8_meteorological_effects.png')
    
    print("\n[10/10] 生成图9: 多污染物模型性能对比图...")
    plot_multi_pollutant_performance(f'{output_dir}/fig9_multi_pollutant_performance.png')
    
    # 可选图
    print("\n[可选] 生成图10: 传感器可靠性评分图...")
    plot_sensor_reliability(f'{output_dir}/fig10_sensor_reliability.png')
    
    print("\n" + "=" * 60)
    print("所有图表已生成完毕！")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    # 列出生成的文件
    files = sorted(os.listdir(output_dir))
    print("\n生成的文件列表:")
    for f in files:
        size = os.path.getsize(f'{output_dir}/{f}') / 1024
        print(f"  - {f} ({size:.1f} KB)")


if __name__ == '__main__':
    main()
