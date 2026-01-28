package feast

import (
	"context"
	"time"
)

// Client 是 Feast Feature Store 的客户端接口。
//
// 参考：https://github.com/feast-dev/feast
type Client interface {
	GetOnlineFeatures(ctx context.Context, req *GetOnlineFeaturesRequest) (*GetOnlineFeaturesResponse, error)
	GetHistoricalFeatures(ctx context.Context, req *GetHistoricalFeaturesRequest) (*GetHistoricalFeaturesResponse, error)
	Materialize(ctx context.Context, req *MaterializeRequest) error
	ListFeatures(ctx context.Context) ([]Feature, error)
	GetFeatureService(ctx context.Context) (*FeatureServiceInfo, error)
	Close() error
}

// GetOnlineFeaturesRequest 获取在线特征请求
type GetOnlineFeaturesRequest struct {
	Features   []string
	EntityRows []map[string]interface{}
	Project    string
}

// GetOnlineFeaturesResponse 获取在线特征响应
type GetOnlineFeaturesResponse struct {
	FeatureVectors []FeatureVector
	Metadata       map[string]interface{}
}

// FeatureVector 特征向量
type FeatureVector struct {
	Values    map[string]interface{}
	EntityRow map[string]interface{}
}

// GetHistoricalFeaturesRequest 获取历史特征请求
type GetHistoricalFeaturesRequest struct {
	EntityDF  []map[string]interface{}
	Features  []string
	StartTime *time.Time
	EndTime   *time.Time
	Project   string
}

// GetHistoricalFeaturesResponse 获取历史特征响应
type GetHistoricalFeaturesResponse struct {
	DataFrame []map[string]interface{}
	Metadata  map[string]interface{}
}

// MaterializeRequest 物化请求
type MaterializeRequest struct {
	StartTime    time.Time
	EndTime      time.Time
	FeatureViews []string
	Project      string
}

// Feature 特征定义
type Feature struct {
	Name        string
	FeatureView string
	ValueType   string
	Description string
}

// FeatureServiceInfo 特征服务信息
type FeatureServiceInfo struct {
	Endpoint     string
	Project      string
	FeatureViews []string
	OnlineStore  string
	OfflineStore string
}

// FeatureMapping 特征映射配置
type FeatureMapping struct {
	UserFeatures     []string
	ItemFeatures     []string
	RealtimeFeatures []string
	UserEntityKey    string
	ItemEntityKey    string
}

// ClientOption 客户端配置选项
type ClientOption func(*ClientConfig)

// ClientConfig 客户端配置
type ClientConfig struct {
	Endpoint string
	Project  string
	Timeout  time.Duration
	UseGRPC  bool
	Auth     *AuthConfig
}

// AuthConfig 认证配置
type AuthConfig struct {
	Type     string
	Username string
	Password string
	Token    string
	APIKey   string
}

// WithTimeout 设置超时时间
func WithTimeout(timeout time.Duration) ClientOption {
	return func(c *ClientConfig) {
		c.Timeout = timeout
	}
}

// WithAuth 设置认证信息
func WithAuth(auth *AuthConfig) ClientOption {
	return func(c *ClientConfig) {
		c.Auth = auth
	}
}

// WithHTTP 使用 HTTP 客户端
func WithHTTP() ClientOption {
	return func(c *ClientConfig) {
		c.UseGRPC = false
	}
}
