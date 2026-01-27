package grpc

import (
	"context"
	"fmt"
	"strconv"
	"time"

	feastsdk "github.com/feast-dev/feast/sdk/go"
	"github.com/rushteam/reckit/ext/feast/http"
)

// GrpcClient 是基于官方 Feast Go SDK 的 gRPC 客户端实现。
//
// 注意：此实现位于扩展包中，需要单独引入：
//   go get github.com/rushteam/reckit/ext/feast/grpc
type GrpcClient struct {
	client   *feastsdk.GrpcClient
	Project  string
	Endpoint string
}

// NewGrpcClient 创建一个基于官方 SDK 的 Feast gRPC 客户端。
func NewGrpcClient(host string, port int, project string, opts ...http.ClientOption) (*GrpcClient, error) {
	if port == 0 {
		port = 6565
	}

	config := &http.ClientConfig{
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
		security := feastsdk.SecurityConfig{
			EnableTLS:  false,
			Credential: credential,
		}
		client, err = feastsdk.NewSecureGrpcClient(host, port, security)
	} else {
		client, err = feastsdk.NewGrpcClient(host, port)
	}

	if err != nil {
		return nil, fmt.Errorf("创建 Feast gRPC 客户端失败: %w", err)
	}

	return &GrpcClient{
		client:   client,
		Project:  project,
		Endpoint: config.Endpoint,
	}, nil
}

// GetOnlineFeatures 实现 Client 接口
func (c *GrpcClient) GetOnlineFeatures(ctx context.Context, req *http.GetOnlineFeaturesRequest) (*http.GetOnlineFeaturesResponse, error) {
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

	featureVectors := make([]http.FeatureVector, len(rows))
	featureNames := req.Features

	for i := 0; i < len(rows); i++ {
		values := make(map[string]interface{})
		row := rows[i]

		for _, featureName := range featureNames {
			if val, exists := row[featureName]; exists {
				convertedVal := convertFromSDKValue(val)
				if convertedVal != nil {
					values[featureName] = convertedVal
				}
			}
		}

		featureVectors[i] = http.FeatureVector{
			Values:    values,
			EntityRow: req.EntityRows[i],
		}
	}

	return &http.GetOnlineFeaturesResponse{
		FeatureVectors: featureVectors,
		Metadata:       make(map[string]interface{}),
	}, nil
}

func (c *GrpcClient) GetHistoricalFeatures(ctx context.Context, req *http.GetHistoricalFeaturesRequest) (*http.GetHistoricalFeaturesResponse, error) {
	return nil, fmt.Errorf("历史特征获取暂不支持，请使用 HTTP 客户端")
}

func (c *GrpcClient) Materialize(ctx context.Context, req *http.MaterializeRequest) error {
	return fmt.Errorf("特征物化暂不支持，请使用 HTTP 客户端")
}

func (c *GrpcClient) ListFeatures(ctx context.Context) ([]http.Feature, error) {
	return nil, fmt.Errorf("列出特征暂不支持，请使用 HTTP 客户端")
}

func (c *GrpcClient) GetFeatureService(ctx context.Context) (*http.FeatureServiceInfo, error) {
	return &http.FeatureServiceInfo{
		Endpoint:     c.Endpoint,
		Project:      c.Project,
		FeatureViews: []string{},
		OnlineStore:  "grpc",
		OfflineStore: "unknown",
	}, nil
}

func (c *GrpcClient) Close(ctx context.Context) error {
	c.client = nil
	return nil
}

func convertFromSDKValue(val interface{}) interface{} {
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

var _ http.Client = (*GrpcClient)(nil)