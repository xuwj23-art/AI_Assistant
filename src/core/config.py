"""
项目配置文件
"""
from pathlib import Path

# 项目根目录（相对于此文件）
BASE_DIR = Path(__file__).parent.parent.parent

# 数据目录
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 模型目录
MODELS_DIR = BASE_DIR / "models"

# 确保目录存在
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)