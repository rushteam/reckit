package service

import (
	"context"
	"fmt"
	"time"

	"github.com/rushteam/reckit/core"
)

// 本包为 core.MLService 的实现；接口与请求/响应类型在 core 包。
// 创建实例请使用 NewMLService(config)，或直接 NewTorchServeClient / NewTFServingClient。

// ServiceType 服务类型
type ServiceType string

const (
	ServiceTypeTFServing  ServiceType = "tf_serving"  // TensorFlow Serving
	ServiceTypeTorchServe ServiceType = "torch_serve" // TorchServe / 自研 Python（POST /predictions/{model_name}）
)

// ServiceConfig 服务配置（工厂 NewMLService 入参）
type ServiceConfig struct {
	Type       ServiceType
	Endpoint   string // 不含路径；TorchServe 客户端会拼 /predictions/{ModelName}
	ModelName  string
	ModelVersion string
	Timeout    int    // 秒
	Auth       *AuthConfig
	Params     map[string]interface{}
}

// AuthConfig 认证配置
type AuthConfig struct {
	Type     string
	Username string
	Password string
	Token    string
	APIKey   string
}

// NewMLService 根据配置创建 MLService 实例（工厂方法）。
// 返回 core.MLService 接口。
func NewMLService(config *ServiceConfig) (core.MLService, error) {
	if config == nil {
		return nil, fmt.Errorf("service config is required")
	}

	timeout := time.Duration(config.Timeout) * time.Second
	if timeout == 0 {
		timeout = 30 * time.Second
	}

	switch config.Type {
	case ServiceTypeTFServing:
		opts := []TFServingOption{
			WithTFServingTimeout(timeout),
		}
		if config.ModelVersion != "" {
			opts = append(opts, WithTFServingVersion(config.ModelVersion))
		}
		if config.Auth != nil {
			opts = append(opts, WithTFServingAuth(config.Auth))
		}
		// 判断是 gRPC 还是 REST
		if isGRPCEndpoint(config.Endpoint) {
			opts = append(opts, WithTFServingGRPC())
		}
		return NewTFServingClient(config.Endpoint, config.ModelName, opts...), nil

	// ServiceTypeANN 已移除：ANN（向量检索）应该使用 core.VectorService，而不是 core.MLService
	// 如果需要向量检索，请使用 ext/vector/milvus 或 store.MemoryVectorService

	case ServiceTypeTorchServe:
		opts := []TorchServeOption{
			WithTorchServeTimeout(timeout),
		}
		if config.ModelVersion != "" {
			opts = append(opts, WithTorchServeVersion(config.ModelVersion))
		}
		if config.Auth != nil {
			opts = append(opts, WithTorchServeAuth(config.Auth))
		}
		return NewTorchServeClient(config.Endpoint, config.ModelName, opts...), nil

	default:
		return nil, fmt.Errorf("unsupported service type: %s", config.Type)
	}
}

// isGRPCEndpoint 判断端点是否为 gRPC（简单判断，不包含协议前缀则认为是 gRPC）
func isGRPCEndpoint(endpoint string) bool {
	// 如果包含 http:// 或 https://，则是 REST API
	// 否则认为是 gRPC（如 localhost:8500）
	return !hasHTTPPrefix(endpoint)
}

// hasHTTPPrefix 检查是否包含 HTTP 前缀
func hasHTTPPrefix(s string) bool {
	return len(s) > 7 && (s[:7] == "http://" || s[:8] == "https://")
}

// ValidateConfig 验证服务配置
func ValidateConfig(config *ServiceConfig) error {
	if config == nil {
		return fmt.Errorf("config is required")
	}
	if config.Endpoint == "" {
		return fmt.Errorf("endpoint is required")
	}
	if config.ModelName == "" {
		return fmt.Errorf("model name is required")
	}
	return nil
}

// TestConnection 测试服务连接
func TestConnection(ctx context.Context, svc core.MLService) error {
	if svc == nil {
		return fmt.Errorf("service is nil")
	}
	return svc.Health(ctx)
}
