"""预测用例：批量特征 -> 分数/向量列表"""
from typing import Any


def run_batch_predict(loader: Any, features_list: list[dict[str, float]]) -> list:
    """
    批量预测：调用 loader.predict(features_list)，返回分数或 embedding 列表。
    loader 需实现 predict(features_list) -> list[float] 或 list[list[float]] 或 list[dict]。
    """
    return loader.predict(features_list)
