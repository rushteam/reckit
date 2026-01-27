package service

import (
	"context"
	"fmt"
	"time"

	"github.com/rushteam/reckit/core"
)

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

	case ServiceTypeANN:
		opts := []ANNServiceOption{
			WithANNServiceTimeout(timeout),
		}
		if config.Auth != nil {
			opts = append(opts, WithANNServiceAuth(config.Auth))
		}
		return NewANNServiceClient(config.Endpoint, config.ModelName, opts...), nil

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
	if config.ModelName == "" && config.Type != ServiceTypeANN {
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
