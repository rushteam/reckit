"""
特征配置：定义训练和推理时使用的特征列

与 RPCNode 默认行为（StripFeaturePrefix=False）对齐：使用带前缀特征名。
- item_*：物品特征（EnrichNode ItemFeaturePrefix）
- user_*：用户特征（EnrichNode UserFeaturePrefix）
- cross_*：交叉特征（EnrichNode CrossFeaturePrefix）
"""
import os

# 特征列配置（带前缀，与 EnrichNode 产出及 RPCNode 默认不 strip 时一致）
FEATURE_COLUMNS = [
    # 物品特征
    "item_ctr",
    "item_cvr",
    "item_price",
    # 用户特征
    "user_age",
    "user_gender",
    # 交叉特征（训练时计算，命名与 defaultCrossFeatures 一致）
    "cross_age_x_ctr",
    "cross_gender_x_price",
]

# 标签列
LABEL_COLUMN = "label"

# 模型保存路径（相对于项目根目录）
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.json")
FEATURE_META_PATH = os.path.join(MODEL_DIR, "feature_meta.json")
