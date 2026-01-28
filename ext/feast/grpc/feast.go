package grpc

import (
	"context"
	"fmt"
	"strconv"
	"time"

	feastsdk "github.com/feast-dev/feast/sdk/go"
	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/ext/feast/common"
)

// FeatureServiceAdapter 将 Feast gRPC Client 适配为 core.FeatureService 接口。
//
// 注意：此实现位于扩展包中，需要单独引入：
//
//	go get github.com/rushteam/reckit/ext/feast/grpc
//
// 这是推荐的使用方式：通过适配器将 Feast（基础设施层）适配为 core.FeatureService（领域层）。
type FeatureServiceAdapter struct {
	client         common.Client
	featureMapping *common.FeatureMapping
}

// NewFeatureServiceAdapter 创建一个新的 FeatureService 适配器。
//
// 参数：
//   - client: Feast gRPC 客户端（基础设施层）
//   - mapping: 特征映射配置
//
// 返回：
//   - *FeatureServiceAdapter: 实现了 core.FeatureService 接口的适配器
func NewFeatureServiceAdapter(client common.Client, mapping *common.FeatureMapping) *FeatureServiceAdapter {
	if mapping.UserEntityKey == "" {
		mapping.UserEntityKey = "user_id"
	}
	if mapping.ItemEntityKey == "" {
		mapping.ItemEntityKey = "item_id"
	}

	return &FeatureServiceAdapter{
		client:         client,
		featureMapping: mapping,
	}
}

// Name 返回特征服务名称
func (a *FeatureServiceAdapter) Name() string {
	return "feast-grpc"
}

// GetUserFeatures 获取用户特征
func (a *FeatureServiceAdapter) GetUserFeatures(ctx context.Context, userID string) (map[string]float64, error) {
	if len(a.featureMapping.UserFeatures) == 0 {
		return make(map[string]float64), nil
	}

	// 构建实体行
	entityRows := []map[string]interface{}{
		{a.featureMapping.UserEntityKey: userID},
	}

	// 调用 Feast 客户端
	req := &common.GetOnlineFeaturesRequest{
		Features:   a.featureMapping.UserFeatures,
		EntityRows: entityRows,
	}

	resp, err := a.client.GetOnlineFeatures(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("feast get user features failed: %w", err)
	}

	if len(resp.FeatureVectors) == 0 {
		return make(map[string]float64), nil
	}

	// 转换特征向量为 map[string]float64
	features := make(map[string]float64)
	for k, v := range resp.FeatureVectors[0].Values {
		if fv, ok := convertToFloat64(v); ok {
			features[k] = fv
		}
	}

	return features, nil
}

// BatchGetUserFeatures 批量获取用户特征
func (a *FeatureServiceAdapter) BatchGetUserFeatures(ctx context.Context, userIDs []string) (map[string]map[string]float64, error) {
	if len(a.featureMapping.UserFeatures) == 0 {
		result := make(map[string]map[string]float64)
		for _, userID := range userIDs {
			result[userID] = make(map[string]float64)
		}
		return result, nil
	}

	// 构建实体行
	entityRows := make([]map[string]interface{}, len(userIDs))
	for i, userID := range userIDs {
		entityRows[i] = map[string]interface{}{
			a.featureMapping.UserEntityKey: userID,
		}
	}

	// 调用 Feast 客户端
	req := &common.GetOnlineFeaturesRequest{
		Features:   a.featureMapping.UserFeatures,
		EntityRows: entityRows,
	}

	resp, err := a.client.GetOnlineFeatures(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("feast batch get user features failed: %w", err)
	}

	// 转换特征向量为 map[string]map[string]float64
	result := make(map[string]map[string]float64)
	for i, fv := range resp.FeatureVectors {
		if i >= len(userIDs) {
			break
		}
		userID := userIDs[i]
		features := make(map[string]float64)
		for k, v := range fv.Values {
			if fv, ok := convertToFloat64(v); ok {
				features[k] = fv
			}
		}
		result[userID] = features
	}

	return result, nil
}

// GetItemFeatures 获取物品特征
func (a *FeatureServiceAdapter) GetItemFeatures(ctx context.Context, itemID string) (map[string]float64, error) {
	if len(a.featureMapping.ItemFeatures) == 0 {
		return make(map[string]float64), nil
	}

	// 构建实体行
	entityRows := []map[string]interface{}{
		{a.featureMapping.ItemEntityKey: itemID},
	}

	// 调用 Feast 客户端
	req := &common.GetOnlineFeaturesRequest{
		Features:   a.featureMapping.ItemFeatures,
		EntityRows: entityRows,
	}

	resp, err := a.client.GetOnlineFeatures(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("feast get item features failed: %w", err)
	}

	if len(resp.FeatureVectors) == 0 {
		return make(map[string]float64), nil
	}

	// 转换特征向量为 map[string]float64
	features := make(map[string]float64)
	for k, v := range resp.FeatureVectors[0].Values {
		if fv, ok := convertToFloat64(v); ok {
			features[k] = fv
		}
	}

	return features, nil
}

// BatchGetItemFeatures 批量获取物品特征
func (a *FeatureServiceAdapter) BatchGetItemFeatures(ctx context.Context, itemIDs []string) (map[string]map[string]float64, error) {
	if len(a.featureMapping.ItemFeatures) == 0 {
		result := make(map[string]map[string]float64)
		for _, itemID := range itemIDs {
			result[itemID] = make(map[string]float64)
		}
		return result, nil
	}

	// 构建实体行
	entityRows := make([]map[string]interface{}, len(itemIDs))
	for i, itemID := range itemIDs {
		entityRows[i] = map[string]interface{}{
			a.featureMapping.ItemEntityKey: itemID,
		}
	}

	// 调用 Feast 客户端
	req := &common.GetOnlineFeaturesRequest{
		Features:   a.featureMapping.ItemFeatures,
		EntityRows: entityRows,
	}

	resp, err := a.client.GetOnlineFeatures(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("feast batch get item features failed: %w", err)
	}

	// 转换特征向量为 map[string]map[string]float64
	result := make(map[string]map[string]float64)
	for i, fv := range resp.FeatureVectors {
		if i >= len(itemIDs) {
			break
		}
		itemID := itemIDs[i]
		features := make(map[string]float64)
		for k, v := range fv.Values {
			if fv, ok := convertToFloat64(v); ok {
				features[k] = fv
			}
		}
		result[itemID] = features
	}

	return result, nil
}

// GetRealtimeFeatures 获取实时特征
func (a *FeatureServiceAdapter) GetRealtimeFeatures(ctx context.Context, userID, itemID string) (map[string]float64, error) {
	if len(a.featureMapping.RealtimeFeatures) == 0 {
		return make(map[string]float64), nil
	}

	// 构建实体行
	entityRows := []map[string]interface{}{
		{
			a.featureMapping.UserEntityKey: userID,
			a.featureMapping.ItemEntityKey: itemID,
		},
	}

	// 调用 Feast 客户端
	req := &common.GetOnlineFeaturesRequest{
		Features:   a.featureMapping.RealtimeFeatures,
		EntityRows: entityRows,
	}

	resp, err := a.client.GetOnlineFeatures(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("feast get realtime features failed: %w", err)
	}

	if len(resp.FeatureVectors) == 0 {
		return make(map[string]float64), nil
	}

	// 转换特征向量为 map[string]float64
	features := make(map[string]float64)
	for k, v := range resp.FeatureVectors[0].Values {
		if fv, ok := convertToFloat64(v); ok {
			features[k] = fv
		}
	}

	return features, nil
}

// BatchGetRealtimeFeatures 批量获取实时特征
func (a *FeatureServiceAdapter) BatchGetRealtimeFeatures(ctx context.Context, pairs []core.FeatureUserItemPair) (map[core.FeatureUserItemPair]map[string]float64, error) {
	if len(a.featureMapping.RealtimeFeatures) == 0 {
		result := make(map[core.FeatureUserItemPair]map[string]float64)
		for _, pair := range pairs {
			result[pair] = make(map[string]float64)
		}
		return result, nil
	}

	// 构建实体行
	entityRows := make([]map[string]interface{}, len(pairs))
	for i, pair := range pairs {
		entityRows[i] = map[string]interface{}{
			a.featureMapping.UserEntityKey: pair.UserID,
			a.featureMapping.ItemEntityKey: pair.ItemID,
		}
	}

	// 调用 Feast 客户端
	req := &common.GetOnlineFeaturesRequest{
		Features:   a.featureMapping.RealtimeFeatures,
		EntityRows: entityRows,
	}

	resp, err := a.client.GetOnlineFeatures(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("feast batch get realtime features failed: %w", err)
	}

	// 转换特征向量为 map[FeatureUserItemPair]map[string]float64
	result := make(map[core.FeatureUserItemPair]map[string]float64)
	for i, fv := range resp.FeatureVectors {
		if i >= len(pairs) {
			break
		}
		pair := pairs[i]
		features := make(map[string]float64)
		for k, v := range fv.Values {
			if fv, ok := convertToFloat64(v); ok {
				features[k] = fv
			}
		}
		result[pair] = features
	}

	return result, nil
}

// Close 关闭特征服务（实现 core.FeatureService 接口）
func (a *FeatureServiceAdapter) Close(ctx context.Context) error {
	return a.client.Close()
}

// GrpcClient 是基于官方 Feast Go SDK 的 gRPC 客户端实现。
//
// 注意：此实现位于扩展包中，需要单独引入：
//
//	go get github.com/rushteam/reckit/ext/feast/grpc
type GrpcClient struct {
	client   *feastsdk.GrpcClient
	Project  string
	Endpoint string
}

// NewClient 创建一个基于官方 SDK 的 Feast gRPC 客户端。
func NewClient(host string, port int, project string, opts ...common.ClientOption) (common.Client, error) {
	if port == 0 {
		port = 6565
	}

	config := &common.ClientConfig{
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

// GetOnlineFeatures 实现 common.Client 接口
func (c *GrpcClient) GetOnlineFeatures(ctx context.Context, req *common.GetOnlineFeaturesRequest) (*common.GetOnlineFeaturesResponse, error) {
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

	featureVectors := make([]common.FeatureVector, len(rows))
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

		featureVectors[i] = common.FeatureVector{
			Values:    values,
			EntityRow: req.EntityRows[i],
		}
	}

	return &common.GetOnlineFeaturesResponse{
		FeatureVectors: featureVectors,
		Metadata:       make(map[string]interface{}),
	}, nil
}

// GetHistoricalFeatures 获取历史特征（暂不支持）
func (c *GrpcClient) GetHistoricalFeatures(ctx context.Context, req *common.GetHistoricalFeaturesRequest) (*common.GetHistoricalFeaturesResponse, error) {
	return nil, fmt.Errorf("历史特征获取暂不支持，请使用 HTTP 客户端")
}

// Materialize 特征物化（暂不支持）
func (c *GrpcClient) Materialize(ctx context.Context, req *common.MaterializeRequest) error {
	return fmt.Errorf("特征物化暂不支持，请使用 HTTP 客户端")
}

// ListFeatures 列出特征（暂不支持）
func (c *GrpcClient) ListFeatures(ctx context.Context) ([]common.Feature, error) {
	return nil, fmt.Errorf("列出特征暂不支持，请使用 HTTP 客户端")
}

// GetFeatureService 获取特征服务信息
func (c *GrpcClient) GetFeatureService(ctx context.Context) (*common.FeatureServiceInfo, error) {
	return &common.FeatureServiceInfo{
		Endpoint:     c.Endpoint,
		Project:      c.Project,
		FeatureViews: []string{},
		OnlineStore:  "grpc",
		OfflineStore: "unknown",
	}, nil
}

// Close 关闭客户端连接（实现 common.Client 接口）
func (c *GrpcClient) Close() error {
	if c.client != nil {
		// Feast SDK 的 gRPC 客户端没有显式的 Close 方法，设置为 nil 即可
		c.client = nil
	}
	return nil
}

// convertFromSDKValue 将 SDK 值转换为通用类型
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

// convertToFloat64 将 interface{} 转换为 float64
func convertToFloat64(v interface{}) (float64, bool) {
	switch val := v.(type) {
	case float64:
		return val, true
	case float32:
		return float64(val), true
	case int:
		return float64(val), true
	case int64:
		return float64(val), true
	case int32:
		return float64(val), true
	default:
		return 0, false
	}
}

var (
	_ core.FeatureService = (*FeatureServiceAdapter)(nil)
	_ common.Client       = (*GrpcClient)(nil)
)
