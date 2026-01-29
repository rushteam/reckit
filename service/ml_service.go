package service

// 注意：此包只包含实现，接口定义在 core 包。
// 使用 core.MLService 接口，core.MLPredictRequest 和 core.MLPredictResponse 类型。
//
// 示例：
//   var mlService core.MLService = NewTorchServeClient(...)
//   req := &core.MLPredictRequest{...}
//   resp, err := mlService.Predict(ctx, req)

// ServiceType 服务类型
type ServiceType string

const (
	ServiceTypeTFServing  ServiceType = "tf_serving"  // TensorFlow Serving
	ServiceTypeTorchServe ServiceType = "torch_serve" // TorchServe
	// ServiceTypeANN 已移除：ANN（向量检索）应该使用 core.VectorService，而不是 core.MLService
)

// ServiceConfig 服务配置
type ServiceConfig struct {
	// Type 服务类型
	Type ServiceType

	// Endpoint 服务端点（不含路径，TorchServe 客户端会拼 /predictions/{ModelName}）
	// TF Serving: "localhost:8500" (gRPC) 或 "http://localhost:8501" (REST)
	// TorchServe / 自研 Python 服务: "http://localhost:8080"，推荐统一使用 POST /predictions/{model_name}
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
