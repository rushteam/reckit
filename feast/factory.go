package feast

import (
	"context"
	"strconv"
	"strings"
	"time"
)

// NewClient 统一的客户端创建函数，根据配置自动选择实现。
//
// 这是推荐的创建方式，支持：
//   - HTTP 客户端（自定义实现，支持完整功能）
//   - gRPC 客户端（官方 SDK，性能更好）
//
// 参数：
//   - endpoint: 服务端点
//   - HTTP: "http://localhost:6566" 或 "https://localhost:6566"
//   - gRPC: "localhost:6565" 或 "grpc://localhost:6565"
//   - project: 项目名称
//   - opts: 配置选项
//
// 示例：
//
//	// 使用 HTTP 客户端
//	client, err := feast.NewClient("http://localhost:6566", "my_project")
//
//	// 使用 gRPC 客户端（官方 SDK）
//	client, err := feast.NewClient("localhost:6565", "my_project", feast.WithGRPC())
func NewClient(endpoint, project string, opts ...ClientOption) (Client, error) {
	ctx := context.Background()
	factory := &DefaultClientFactory{}
	return factory.NewClient(ctx, endpoint, project, opts...)
}

// NewClientWithContext 带上下文的客户端创建函数
func NewClientWithContext(ctx context.Context, endpoint, project string, opts ...ClientOption) (Client, error) {
	factory := &DefaultClientFactory{}
	return factory.NewClient(ctx, endpoint, project, opts...)
}

// WithTimeout 配置选项：设置超时时间
func WithTimeout(timeout time.Duration) ClientOption {
	return func(c *ClientConfig) {
		c.Timeout = timeout
	}
}

// WithAuth 配置选项：设置认证信息
func WithAuth(auth *AuthConfig) ClientOption {
	return func(c *ClientConfig) {
		c.Auth = auth
	}
}

// parseEndpoint 解析端点地址，返回 host 和 port
func parseEndpoint(endpoint string) (string, int) {
	// 移除协议前缀
	endpoint = strings.TrimPrefix(endpoint, "http://")
	endpoint = strings.TrimPrefix(endpoint, "https://")
	endpoint = strings.TrimPrefix(endpoint, "grpc://")

	// 分割 host:port
	parts := strings.Split(endpoint, ":")
	if len(parts) == 2 {
		port, err := strconv.Atoi(parts[1])
		if err == nil {
			return parts[0], port
		}
	}

	// 如果没有端口，返回默认值
	return endpoint, 0
}
