package service

import (
	"context"
)

// MLService 是统一的机器学习服务接口，用于对接 TF Serving、TorchServe、自定义模型服务等。
//
// 设计目标：
//   - 统一不同模型服务的接口（TF Serving、TorchServe、自定义服务）
//   - 支持批量预测
//   - 支持超时控制
//   - 支持错误处理和重试
//
// 使用示例：
//
//	service := service.NewTFServingClient("localhost:8500", "model_name", "v1")
//	scores, err := service.Predict(ctx, &service.PredictRequest{
//	    Instances: [][]float64{features},
//	})
type MLService interface {
	// Predict 批量预测
	Predict(ctx context.Context, req *PredictRequest) (*PredictResponse, error)

	// Health 健康检查
	Health(ctx context.Context) error

	// Close 关闭连接
	Close() error
}

// PredictRequest 预测请求
type PredictRequest struct {
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

// PredictResponse 预测响应
type PredictResponse struct {
	// Predictions 预测结果列表（与请求实例一一对应）
	Predictions []float64

	// Outputs 原始输出（可选，用于调试）
	Outputs interface{}

	// ModelVersion 模型版本（如果服务返回）
	ModelVersion string
}

// ServiceType 服务类型
type ServiceType string

const (
	ServiceTypeTFServing ServiceType = "tf_serving"  // TensorFlow Serving
	ServiceTypeTorchServe ServiceType = "torch_serve" // TorchServe
	ServiceTypeCustom    ServiceType = "custom"      // 自定义服务
	ServiceTypeANN       ServiceType = "ann"          // ANN 服务
)

// ServiceConfig 服务配置
type ServiceConfig struct {
	// Type 服务类型
	Type ServiceType

	// Endpoint 服务端点
	// TF Serving: "localhost:8500" (gRPC) 或 "http://localhost:8501" (REST)
	// TorchServe: "http://localhost:8080"
	// Custom: "http://localhost:8080/predict"
	Endpoint string

	// ModelName 模型名称
	ModelName string

	// ModelVersion 模型版本
	ModelVersion string

	// Timeout 超时时间（秒）
	Timeout int

	// Auth 认证信息（可选）
	Auth *AuthConfig

	// Params 额外参数
	Params map[string]interface{}
}

// AuthConfig 认证配置
type AuthConfig struct {
	Type     string // "basic", "bearer", "api_key"
	Username string
	Password string
	Token    string
	APIKey   string
}
