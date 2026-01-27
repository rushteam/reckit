package feast

import (
	"context"
	"time"
)

// Client 是 Feast Feature Store 的客户端接口（遵循 DDD 原则，高内聚低耦合）。
//
// Feast 是一个开源的 Feature Store，提供：
//   - 离线特征存储（Offline Store）：用于训练数据
//   - 在线特征存储（Online Store）：用于实时预测
//   - Feature Server：提供特征服务
//   - 特征注册和管理
//
// 使用方式：
//   - 方式1：使用扩展包实现（推荐）
//     - HTTP 客户端：go get github.com/rushteam/reckit/ext/feast/http
//     - gRPC 客户端：go get github.com/rushteam/reckit/ext/feast/grpc
//   - 方式2：自行实现此接口（参考扩展包实现）
//
// 参考：https://github.com/feast-dev/feast
type Client interface {
	// GetOnlineFeatures 获取在线特征（用于实时预测）
	//
	// 参数：
	//   - features: 特征名称列表，例如 ["driver_hourly_stats:conv_rate", "driver_hourly_stats:acc_rate"]
	//   - entityRows: 实体行，例如 [{"driver_id": 1001}]
	//
	// 返回：
	//   - FeatureVector: 特征向量，key 为特征名称，value 为特征值
	//   - error: 错误信息
	GetOnlineFeatures(ctx context.Context, req *GetOnlineFeaturesRequest) (*GetOnlineFeaturesResponse, error)

	// GetHistoricalFeatures 获取历史特征（用于训练数据）
	//
	// 参数：
	//   - entityDF: 实体数据框（包含实体 ID 和时间戳）
	//   - features: 特征名称列表
	//   - startTime: 开始时间
	//   - endTime: 结束时间
	//
	// 返回：
	//   - DataFrame: 历史特征数据框
	//   - error: 错误信息
	GetHistoricalFeatures(ctx context.Context, req *GetHistoricalFeaturesRequest) (*GetHistoricalFeaturesResponse, error)

	// Materialize 将特征物化到在线存储
	//
	// 参数：
	//   - startTime: 开始时间
	//   - endTime: 结束时间
	//   - featureViews: 特征视图列表（可选，为空则物化所有）
	//
	// 返回：
	//   - error: 错误信息
	Materialize(ctx context.Context, req *MaterializeRequest) error

	// ListFeatures 列出所有可用的特征
	//
	// 返回：
	//   - []Feature: 特征列表
	//   - error: 错误信息
	ListFeatures(ctx context.Context) ([]Feature, error)

	// GetFeatureService 获取特征服务信息
	//
	// 返回：
	//   - FeatureServiceInfo: 特征服务信息
	//   - error: 错误信息
	GetFeatureService(ctx context.Context) (*FeatureServiceInfo, error)

	// Close 关闭客户端连接
	Close() error
}

// GetOnlineFeaturesRequest 获取在线特征请求
type GetOnlineFeaturesRequest struct {
	// Features 特征名称列表，例如 ["driver_hourly_stats:conv_rate", "driver_hourly_stats:acc_rate"]
	Features []string

	// EntityRows 实体行，例如 [{"driver_id": 1001}, {"driver_id": 1002}]
	EntityRows []map[string]interface{}

	// Project 项目名称（可选）
	Project string
}

// GetOnlineFeaturesResponse 获取在线特征响应
type GetOnlineFeaturesResponse struct {
	// FeatureVectors 特征向量列表，每个元素对应一个实体行
	FeatureVectors []FeatureVector

	// Metadata 元数据
	Metadata map[string]interface{}
}

// FeatureVector 特征向量
type FeatureVector struct {
	// Values 特征值，key 为特征名称，value 为特征值
	Values map[string]interface{}

	// EntityRow 对应的实体行
	EntityRow map[string]interface{}
}

// GetHistoricalFeaturesRequest 获取历史特征请求
type GetHistoricalFeaturesRequest struct {
	// EntityDF 实体数据框（包含实体 ID 和时间戳）
	// 格式：[]map[string]interface{}，例如：
	//   [{"driver_id": 1001, "event_timestamp": "2021-04-12T10:59:42Z"}]
	EntityDF []map[string]interface{}

	// Features 特征名称列表
	Features []string

	// StartTime 开始时间（可选）
	StartTime *time.Time

	// EndTime 结束时间（可选）
	EndTime *time.Time

	// Project 项目名称（可选）
	Project string
}

// GetHistoricalFeaturesResponse 获取历史特征响应
type GetHistoricalFeaturesResponse struct {
	// DataFrame 历史特征数据框
	// 格式：[]map[string]interface{}，包含实体列、时间戳列和特征列
	DataFrame []map[string]interface{}

	// Metadata 元数据
	Metadata map[string]interface{}
}

// MaterializeRequest 物化请求
type MaterializeRequest struct {
	// StartTime 开始时间
	StartTime time.Time

	// EndTime 结束时间
	EndTime time.Time

	// FeatureViews 特征视图列表（可选，为空则物化所有）
	FeatureViews []string

	// Project 项目名称（可选）
	Project string
}

// Feature 特征定义
type Feature struct {
	// Name 特征名称，例如 "driver_hourly_stats:conv_rate"
	Name string

	// FeatureView 特征视图名称，例如 "driver_hourly_stats"
	FeatureView string

	// ValueType 特征值类型，例如 "FLOAT", "INT64", "STRING"
	ValueType string

	// Description 特征描述
	Description string
}

// FeatureServiceInfo 特征服务信息
type FeatureServiceInfo struct {
	// Endpoint 服务端点
	Endpoint string

	// Project 项目名称
	Project string

	// FeatureViews 特征视图列表
	FeatureViews []string

	// OnlineStore 在线存储类型
	OnlineStore string

	// OfflineStore 离线存储类型
	OfflineStore string
}

// ClientFactory 是 Feast 客户端工厂接口（用于依赖注入）。
type ClientFactory interface {
	NewClient(ctx context.Context, endpoint string, project string, opts ...ClientOption) (Client, error)
}

// ClientOption Feast 客户端配置选项
type ClientOption func(*ClientConfig)

// ClientConfig Feast 客户端配置
type ClientConfig struct {
	// Endpoint 服务端点
	Endpoint string

	// Project 项目名称
	Project string

	// Timeout 超时时间
	Timeout time.Duration

	// UseGRPC 是否使用 gRPC（默认 false，使用 HTTP）
	UseGRPC bool

	// Auth 认证信息
	Auth *AuthConfig
}

// AuthConfig 认证配置
type AuthConfig struct {
	// Type 认证类型：basic, bearer, api_key, static
	// static 用于 gRPC 的静态 Token 认证
	Type string

	// Username 用户名（basic auth）
	Username string

	// Password 密码（basic auth）
	Password string

	// Token Token（bearer auth 或 static auth）
	Token string

	// APIKey API Key（api_key auth）
	APIKey string
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

// WithGRPC 配置选项：使用 gRPC 客户端（扩展包）
func WithGRPC() ClientOption {
	return func(c *ClientConfig) {
		c.UseGRPC = true
	}
}

// WithHTTP 配置选项：使用 HTTP 客户端（扩展包）
func WithHTTP() ClientOption {
	return func(c *ClientConfig) {
		c.UseGRPC = false
	}
}
