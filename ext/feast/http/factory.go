package http

import (
	"context"
	"time"
)

// NewClient 统一的客户端创建函数，创建 HTTP 客户端。
//
// 注意：此实现位于扩展包中，需要单独引入：
//   go get github.com/rushteam/reckit/ext/feast/http
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

// DefaultClientFactory 是默认的 Feast HTTP 客户端工厂。
type DefaultClientFactory struct{}

// NewClient 创建 Feast HTTP 客户端
func (f *DefaultClientFactory) NewClient(ctx context.Context, endpoint, project string, opts ...ClientOption) (Client, error) {
	config := &ClientConfig{
		Endpoint: endpoint,
		Project:  project,
		Timeout:  30 * time.Second,
		UseGRPC:  false, // HTTP 客户端
	}

	// 应用配置选项
	for _, opt := range opts {
		opt(config)
	}

	// 使用 HTTP 客户端
	return NewHTTPClient(endpoint, project, opts...)
}