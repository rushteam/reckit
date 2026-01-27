package core

import "context"

// MLService 是机器学习服务的领域接口。
//
// 设计原则：
//   - 定义在领域层（core），由基础设施层（service）实现
//   - 遵循依赖倒置原则：领域层定义接口，基础设施层实现接口
//   - 避免循环依赖：领域层不依赖基础设施层
//
// 使用场景：
//   - 双塔模型召回：用户塔推理
//   - 排序模型：LR、DNN、DIN、Wide&Deep 等
//   - 外部模型服务：TensorFlow Serving、TorchServe、ONNX Runtime 等
//
// 实现：
//   - service.TFServingClient 实现此接口
//   - service.TorchServeClient 实现此接口
//   - 其他模型服务也可以实现此接口
type MLService interface {
	// Predict 批量预测
	Predict(ctx context.Context, req *MLPredictRequest) (*MLPredictResponse, error)

	// Health 健康检查
	Health(ctx context.Context) error

	// Close 关闭连接
	Close(ctx context.Context) error
}

// MLPredictRequest 预测请求
type MLPredictRequest struct {
	// Instances 特征实例列表（每个实例是一个特征向量）
	// 格式：[[f1, f2, f3, ...], [f1, f2, f3, ...], ...]
	Instances [][]float64

	// Features 特征字典列表（可选，与 Instances 二选一）
	// 格式：[{"feature1": 0.1, "feature2": 0.2}, ...]
	Features []map[string]float64

	// ModelName 模型名称（可选，如果服务支持多模型）
	ModelName string

	// ModelVersion 模型版本（可选）
	ModelVersion string

	// SignatureName 签名名称（可选，TF Serving 使用）
	SignatureName string

	// Params 额外参数（可选）
	Params map[string]interface{}
}

// MLPredictResponse 预测响应
type MLPredictResponse struct {
	// Predictions 预测结果列表（与请求实例一一对应）
	Predictions []float64

	// Outputs 原始输出（可选，用于调试）
	Outputs interface{}

	// ModelVersion 模型版本（如果服务返回）
	ModelVersion string
}
