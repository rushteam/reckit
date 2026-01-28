package feast

import (
	"context"
	"fmt"
	"strconv"
	"time"

	feastsdk "github.com/feast-dev/feast/sdk/go"
)

// GrpcClient 是基于官方 Feast Go SDK 的 gRPC 客户端实现。
type GrpcClient struct {
	client   *feastsdk.GrpcClient
	Project  string
	Endpoint string
}

// NewGrpcClient 创建一个基于官方 SDK 的 Feast gRPC 客户端。
func NewGrpcClient(host string, port int, project string, opts ...ClientOption) (Client, error) {
	if port == 0 {
		port = 6565
	}
	config := &ClientConfig{
		Endpoint: fmt.Sprintf("%s:%d", host, port),
		Project:  project,
		Timeout:  30 * time.Second,
		UseGRPC:  true,
	}
	for _, opt := range opts {
		opt(config)
	}
	var client *feastsdk.GrpcClient
	var err error
	if config.Auth != nil && config.Auth.Type == "static" && config.Auth.Token != "" {
		credential := feastsdk.NewStaticCredential(config.Auth.Token)
		security := feastsdk.SecurityConfig{EnableTLS: false, Credential: credential}
		client, err = feastsdk.NewSecureGrpcClient(host, port, security)
	} else {
		client, err = feastsdk.NewGrpcClient(host, port)
	}
	if err != nil {
		return nil, fmt.Errorf("创建 Feast gRPC 客户端失败: %w", err)
	}
	return &GrpcClient{client: client, Project: project, Endpoint: config.Endpoint}, nil
}

// GetOnlineFeatures 获取在线特征
func (c *GrpcClient) GetOnlineFeatures(ctx context.Context, req *GetOnlineFeaturesRequest) (*GetOnlineFeaturesResponse, error) {
	if len(req.Features) == 0 {
		return nil, fmt.Errorf("features are required")
	}
	if len(req.EntityRows) == 0 {
		return nil, fmt.Errorf("entity rows are required")
	}
	project := req.Project
	if project == "" {
		project = c.Project
	}
	if project == "" {
		return nil, fmt.Errorf("project is required")
	}
	entityRows := make([]feastsdk.Row, len(req.EntityRows))
	for i, row := range req.EntityRows {
		entityRow := make(feastsdk.Row)
		for k, v := range row {
			switch val := v.(type) {
			case string:
				entityRow[k] = feastsdk.StrVal(val)
			case int:
				entityRow[k] = feastsdk.Int64Val(int64(val))
			case int64:
				entityRow[k] = feastsdk.Int64Val(val)
			case int32:
				entityRow[k] = feastsdk.Int64Val(int64(val))
			case float64:
				entityRow[k] = feastsdk.DoubleVal(val)
			case float32:
				entityRow[k] = feastsdk.FloatVal(val)
			case bool:
				entityRow[k] = feastsdk.BoolVal(val)
			case []byte:
				entityRow[k] = feastsdk.BytesVal(val)
			default:
				entityRow[k] = feastsdk.StrVal(fmt.Sprintf("%v", val))
			}
		}
		entityRows[i] = entityRow
	}
	sdkReq := &feastsdk.OnlineFeaturesRequest{
		Features: req.Features,
		Entities: entityRows,
		Project:  project,
	}
	sdkResp, err := c.client.GetOnlineFeatures(ctx, sdkReq)
	if err != nil {
		return nil, fmt.Errorf("feast get online features failed: %w", err)
	}
	rows := sdkResp.Rows()
	if len(rows) != len(req.EntityRows) {
		return nil, fmt.Errorf("response row count mismatch: expected %d, got %d", len(req.EntityRows), len(rows))
	}
	featureVectors := make([]FeatureVector, len(rows))
	for i := 0; i < len(rows); i++ {
		values := make(map[string]interface{})
		for _, featureName := range req.Features {
			if val, exists := rows[i][featureName]; exists {
				if cv := sdkValueToInterface(val); cv != nil {
					values[featureName] = cv
				}
			}
		}
		featureVectors[i] = FeatureVector{Values: values, EntityRow: req.EntityRows[i]}
	}
	return &GetOnlineFeaturesResponse{FeatureVectors: featureVectors, Metadata: make(map[string]interface{})}, nil
}

// GetHistoricalFeatures 获取历史特征（暂不支持）
func (c *GrpcClient) GetHistoricalFeatures(ctx context.Context, req *GetHistoricalFeaturesRequest) (*GetHistoricalFeaturesResponse, error) {
	return nil, fmt.Errorf("历史特征获取暂不支持，请使用 HTTP 客户端")
}

// Materialize 特征物化（暂不支持）
func (c *GrpcClient) Materialize(ctx context.Context, req *MaterializeRequest) error {
	return fmt.Errorf("特征物化暂不支持，请使用 HTTP 客户端")
}

// ListFeatures 列出特征（暂不支持）
func (c *GrpcClient) ListFeatures(ctx context.Context) ([]Feature, error) {
	return nil, fmt.Errorf("列出特征暂不支持，请使用 HTTP 客户端")
}

// GetFeatureService 获取特征服务信息
func (c *GrpcClient) GetFeatureService(ctx context.Context) (*FeatureServiceInfo, error) {
	return &FeatureServiceInfo{
		Endpoint:     c.Endpoint,
		Project:      c.Project,
		FeatureViews: []string{},
		OnlineStore:  "grpc",
		OfflineStore: "unknown",
	}, nil
}

// Close 关闭客户端连接
func (c *GrpcClient) Close() error {
	if c.client != nil {
		c.client = nil
	}
	return nil
}

func sdkValueToInterface(val interface{}) interface{} {
	if val == nil {
		return nil
	}
	switch v := val.(type) {
	case string:
		return v
	case int64:
		return float64(v)
	case int32:
		return float64(v)
	case int:
		return float64(v)
	case float64:
		return v
	case float32:
		return float64(v)
	case bool:
		if v {
			return float64(1)
		}
		return float64(0)
	case []byte:
		return string(v)
	default:
		strVal := fmt.Sprintf("%v", val)
		if f, err := strconv.ParseFloat(strVal, 64); err == nil {
			return f
		}
		return strVal
	}
}

var _ Client = (*GrpcClient)(nil)
