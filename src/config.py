"""
配置文件：定义路径、列名映射和清洗参数
"""

from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据路径
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# 原始文件名
REFERENCE_FILE = RAW_DATA_DIR / "reference_data.CSV"  # 国控点数据
TOBECALIBRATED_FILE = RAW_DATA_DIR / "tobecalibrated.CSV"  # 自建点数据

# 输出文件
REFERENCE_CLEAN_FILE = PROCESSED_DATA_DIR / "reference_clean.parquet"
SELFBUILD_CLEAN_FILE = PROCESSED_DATA_DIR / "selfbuild_clean.parquet"
HOURLY_MERGED_FILE = PROCESSED_DATA_DIR / "hourly_merged.parquet"

# 列名映射 - 统一为英文内部名
REFERENCE_COLS = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "CO": "co",
    "NO2": "no2",
    "SO2": "so2",
    "O3": "o3",
    "时间": "time",
}

SELFBUILD_COLS = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "CO": "co",
    "NO2": "no2",
    "SO2": "so2",
    "O3": "o3",
    "风速": "wind",
    "压强": "pressure",
    "降水量": "rain",
    "温度": "temp",
    "湿度": "rh",
    "时间": "time",
}

# 国控点污染物列
REFERENCE_POLLUTANT_COLS = ["pm25", "pm10", "co", "no2", "so2", "o3"]

# 自建点污染物列和气象列
SELFBUILD_POLLUTANT_COLS = ["pm25", "pm10", "co", "no2", "so2", "o3"]
SELFBUILD_WEATHER_COLS = ["wind", "pressure", "rain", "temp", "rh"]

# 数值列（污染物 + 气象）
SELFBUILD_NUMERIC_COLS = SELFBUILD_POLLUTANT_COLS + SELFBUILD_WEATHER_COLS

# 物理异常阈值
PHYSICAL_THRESHOLDS = {
    "pm25": {"min": 0, "max": 1000},       # μg/m³
    "pm10": {"min": 0, "max": 1500},       # μg/m³
    "co": {"min": 0, "max": 50},          # mg/m³ (CO较高容忍度)
    "no2": {"min": 0, "max": 500},        # μg/m³
    "so2": {"min": 0, "max": 1000},       # μg/m³
    "o3": {"min": 0, "max": 800},         # μg/m³
    "wind": {"min": 0, "max": 50},        # m/s (风速)
    "pressure": {"min": 800, "max": 1100},  # hPa (气压)
    "rain": {"min": 0, "max": 500},       # mm (降水量)
    "temp": {"min": -50, "max": 60},      # °C (温度)
    "rh": {"min": 0, "max": 100},         # % (相对湿度)
}

# 时间对齐参数
TIME_WINDOW_MINUTES = 30  # 窗口半径（分钟）
MIN_RECORDS_IN_WINDOW = 3  # 窗口内最少有效记录数

# 数据集划分比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 确保输出目录存在
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
