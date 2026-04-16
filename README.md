# 空气质量传感器校准系统

基于多元回归与机器学习的自建监测点数据校准分析工具。

## 项目概述

本项目针对自建空气质量监测点与国控点之间的数据差异问题，提供完整的数据处理、模型训练和异常检测解决方案。

## 功能特性

- **数据清洗**：自动处理缺失值、异常值、时间对齐
- **特征工程**：构建近邻值、窗口统计、气象特征、漂移指数
- **分层建模**：一元线性 → 多元静态 → 动态回归 → XGBoost 对照
- **差异分析**：零点漂移、量程漂移、交叉干扰、气象影响
- **异常检测**：Isolation Forest、统计异常、可靠性评分

## 项目结构

```
air_calibration/
├── src/                          # 源代码
│   ├── config.py                 # 配置参数
│   ├── load_data.py              # 数据加载
│   ├── clean_data.py             # 数据清洗
│   ├── align_hourly.py           # 时间对齐与特征构建
│   ├── build_features.py         # 特征工程
│   ├── train_linear.py           # 线性校准模型
│   ├── train_dynamic.py          # 动态校准模型
│   ├── train_xgb.py              # XGBoost 对照模型
│   ├── evaluate.py               # 评估指标
│   ├── analyze_pm25_diff.py      # PM2.5差异分析
│   ├── analyze_difference_factors.py  # 差异因素分析
│   └── analyze_anomalies.py       # 异常传感器检测
├── src/visualization.py           # 可视化图表生成
├── run_cleaning.py               # 数据清洗流程
├── run_training.py               # 模型训练主程序
├── analysis_report.md            # 分析报告
├── environment.yml               # Conda环境配置
└── data/                         # 数据目录
    ├── raw/                      # 原始数据
    └── processed/                # 处理后数据
├── output/                       # 输出目录
│   └── figures/                  # 可视化图表
└── models/                       # 训练好的模型
```

## 环境配置

### 方式一：Conda 环境

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate aircalib

# 或手动安装依赖
conda create -n aircalib python=3.10
conda activate aircalib
pip install pandas numpy scipy scikit-learn xgboost
```

### 方式二：pip

```bash
pip install pandas numpy scipy scikit-learn xgboost
```

## 快速开始

### 1. 数据清洗

```bash
conda activate aircalib
python run_cleaning.py
```

### 2. 模型训练

```bash
conda activate aircalib
python run_training.py
```

### 3. 查看分析报告

```bash
cat analysis_report.md
```

## 数据说明

### 输入数据

| 文件 | 描述 | 列 |
|:-----|:-----|:---|
| `reference_data.CSV` | 国控点小时级数据 | PM2.5, PM10, CO, NO2, SO2, O3, 时间 |
| `tobecalibrated.CSV` | 自建点高频数据 | 六项污染物 + 气象 + 时间 |

### 输出数据

| 文件 | 描述 |
|:-----|:-----|
| `hourly_merged.parquet` | 小时级特征表 |
| `train/val/test.parquet` | 划分后的数据集 |
| `models/calibration_summary.csv` | 校准结果汇总 |

## 模型体系

### 四层递进模型

| 模型 | 类型 | 公式 | 用途 |
|:-----|:-----|:-----|:-----|
| A | 一元线性 | $Y = a + b \cdot x^{near}$ | 基线对比 |
| B | 多元静态 | $Y = \beta_0 + \sum \beta_i x_i + \sum \delta_j w_j$ | 气象修正 |
| C | 动态校准 | $Y = B + \theta_1 u(t) + \theta_2 u(t)x^{near}$ | 漂移补偿 |
| D | XGBoost | 非线性树模型 | 性能上限 |

### 漂移项说明

- $u(t)$：设备老化索引，从0到1
- $\theta_1 u(t)$：零点漂移项
- $\theta_2 u(t) \cdot x^{near}$：量程漂移交互项

## 支持的污染物

| 污染物 | 单位 | 典型问题 |
|:------:|:----:|:---------|
| PM2.5 | μg/m³ | 湿度干扰 |
| PM10 | μg/m³ | 粒径分布 |
| CO | mg/m³ | 量程漂移 |
| NO2 | μg/m³ | 交叉干扰 |
| SO2 | μg/m³ | 零点漂移 |
| O3 | μg/m³ | 日周期 |

## 评估指标

| 指标 | 说明 | 用途 |
|:----:|:-----|:-----|
| MAE | 平均绝对误差 | 平均偏差 |
| RMSE | 均方根误差 | 大误差敏感 |
| R² | 决定系数 | 拟合程度 |
| MAPE | 平均百分比误差 | 相对精度 |

## 使用示例

### Python API

```python
import pandas as pd
from src.evaluate import calculate_metrics

# 加载数据
hourly_df = pd.read_parquet('data/processed/hourly_merged.parquet')

# 计算误差指标
metrics = calculate_metrics(
    hourly_df['y_pm25'],  # 国控点真实值
    hourly_df['x_pm25_near']  # 自建点读数
)

print(f"MAE: {metrics['MAE']:.2f}")
print(f"RMSE: {metrics['RMSE']:.2f}")
```

### 自定义分析

```python
from src.build_features import get_device_drift_index, get_feature_set_3

# 添加漂移索引
df = get_device_drift_index(df)

# 获取完整特征集
features = get_feature_set_3('pm25')
```

## 校准结果示例

| 污染物 | 最优模型 | 校准前MAE | 校准后MAE | 改善率 |
|:------:|:--------:|:---------:|:---------:|:------:|
| PM2.5 | XGBoost | 22.34 | 8.00 | 64.2% |
| PM10 | Huber | 28.74 | 22.41 | 22.0% |
| O3 | XGBoost | 41.88 | 23.02 | 45.0% |

## 异常检测方法

### 检测流程

```
数据采集 → 一致性检查 → 统计异常检测 → Isolation Forest → 综合评分 → 现场核查
```

### 可靠性评分体系

| 维度 | 权重 | 说明 |
|:----:|:----:|:-----|
| 一致性 | 33% | 与国控点/其他特征的一致程度 |
| 稳定性 | 33% | 随时间变化的稳定性 |
| 完整率 | 33% | 数据缺失比例 |

## 注意事项

1. **时序划分**：训练/验证/测试集按时间顺序划分，避免数据泄漏
2. **标准化**：线性模型使用StandardScaler，仅在训练集上fit
3. **缺失值**：使用中位数填补，不删除含有缺失的样本
4. **漂移监控**：长期部署需定期对比国控点进行校准验证

## 扩展开发

### 添加新污染物

```python
# 1. 在 config.py 添加列名映射
SELFBUILD_COLS = {
    "新污染物": "new_pollutant",
    # ...
}

# 2. 在 train_*.py 中添加到污染物列表
POLLUTANTS = ["pm25", "pm10", "co", "no2", "so2", "o3", "new_pollutant"]
```

### 自定义模型

```python
from src.train_linear import train_multivariate_static

# 添加新的回归器
model, scaler, results = train_multivariate_static(
    X_train, y_train, X_val, y_val,
    model_type="elasticnet"  # 支持 ols, ridge, lasso, huber, elasticnet
)
```

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交 Issue。

---

*Last updated: 2026-04*

## 可视化图表

本项目提供完整的可视化图表生成功能，共生成10张图表用于支撑分析报告。

### 生成图表

```bash
conda activate aircalib
python src/visualization.py
```

生成的图表位于 `output/figures/` 目录。

### 图表清单

#### 问题1：PM2.5差异定量分析

| 图表 | 文件 | 说明 |
|:----|:-----|:-----|
| 图1 | `fig1_pm25_timeseries.png` | PM2.5时间序列对比图：全时段趋势 + 一周局部放大 |
| 图2 | `fig2_pm25_scatter_comparison.png` | PM2.5校准前后散点对比：原始/校准 vs 国控点，y=x参考线 |
| 图3 | `fig3_error_by_pollution_level.png` | 不同污染等级误差柱状图：MAE/RMSE 随污染程度变化 |

#### 问题2：差异因素分析

| 图表 | 文件 | 说明 |
|:----|:-----|:-----|
| 图4 | `fig4_zero_drift.png` | 零点漂移多子图：PM2.5/NO2/SO2/O3 残差时间序列 |
| 图5 | `fig5_range_drift.png` | 量程漂移分段回归图：CO/SO2/O3 前后两期拟合对比 |
| 图7 | `fig7_cross_interference.png` | 交叉干扰热力图：六污染物相关矩阵 |

#### 问题3：校准模型效果

| 图表 | 文件 | 说明 |
|:----|:-----|:-----|
| 图6 | `fig6_pm25_ablation.png` | PM2.5模型消融结果图：A/B/C/D模型 MAE/R²柱状对比 |
| 图8 | `fig8_meteorological_effects.png` | 气象因素影响图：PM2.5残差vs湿度、O3残差vs风速 |
| 图9 | `fig9_multi_pollutant_performance.png` | 多污染物模型性能对比图：六污染物 MAE/RMSE/R² |

#### 问题4：异常传感器检测

| 图表 | 文件 | 说明 |
|:----|:-----|:-----|
| 图10 | `fig10_sensor_reliability.png` | 传感器可靠性评分图：综合评分 + 维度分解 |

### 图表风格

- 分辨率：150 DPI
- 格式：PNG
- 中文字体：SimHei / Noto Sans CJK SC
- 配色：蓝色(国控点)、红色(原始值)、绿色(校准后)、黄绿色(前期)、橙红色(后期)
