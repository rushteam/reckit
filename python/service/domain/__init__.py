# 领域层：协议约定与接口抽象，与 reckit 约束一致
from service.domain.protocol import TorchServePredictRequest, TorchServePredictResponse

__all__ = ["TorchServePredictRequest", "TorchServePredictResponse"]
