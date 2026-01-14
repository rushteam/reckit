"""
特征配置：定义训练和推理时使用的特征列
"""
import os

# 特征列配置
FEATURE_COLUMNS = [
    # 物品特征
    "ctr",
    "cvr",
    "price",
    # 用户特征
    "age",
    "gender",
    # 交叉特征（可选，训练时计算）
    "age_x_ctr",
    "gender_x_price",
]

# 标签列
LABEL_COLUMN = "label"

# 模型保存路径（相对于项目根目录）
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.json")
FEATURE_META_PATH = os.path.join(MODEL_DIR, "feature_meta.json")
