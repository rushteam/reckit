package service

import (
	"context"
	"fmt"
	"time"
)

// NewMLService 根据配置创建 MLService 实例（工厂方法）。
func NewMLService(config *ServiceConfig) (MLService, error) {
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

	case ServiceTypeANN:
		opts := []ANNServiceOption{
			WithANNServiceTimeout(timeout),
		}
		if config.Auth != nil {
			opts = append(opts, WithANNServiceAuth(config.Auth))
		}
		return NewANNServiceClient(config.Endpoint, config.ModelName, opts...), nil

	case ServiceTypeCustom:
		// TODO: 实现自定义服务客户端
		return nil, fmt.Errorf("custom service not implemented")

	case ServiceTypeTorchServe:
		// TODO: 实现 TorchServe 客户端
		return nil, fmt.Errorf("torch serve not implemented")

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
	if config.ModelName == "" && config.Type != ServiceTypeANN {
		return fmt.Errorf("model name is required")
	}
	return nil
}

// TestConnection 测试服务连接
func TestConnection(ctx context.Context, service MLService) error {
	if service == nil {
		return fmt.Errorf("service is nil")
	}
	return service.Health(ctx)
}
