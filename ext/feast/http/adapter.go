package http

import (
	"context"
	"fmt"

	"github.com/rushteam/reckit/core"
)

// FeatureServiceAdapter 将 Feast Client 适配为 core.FeatureService 接口。
//
// 注意：此实现位于扩展包中，需要单独引入：
//
//	go get github.com/rushteam/reckit/ext/feast/http
//
// 这是推荐的使用方式：通过适配器将 Feast（基础设施层）适配为 core.FeatureService（领域层）。
type FeatureServiceAdapter struct {
	client         Client
	featureMapping *FeatureMapping
}

// FeatureMapping 特征映射配置
type FeatureMapping struct {
	// UserFeatures 用户特征列表，例如 ["user_stats:age", "user_stats:gender"]
	UserFeatures []string

	// ItemFeatures 物品特征列表，例如 ["item_stats:price", "item_stats:category"]
	ItemFeatures []string

	// RealtimeFeatures 实时特征列表，例如 ["interaction:click_count", "interaction:view_count"]
	RealtimeFeatures []string

	// UserEntityKey 用户实体键名，默认 "user_id"
	UserEntityKey string

	// ItemEntityKey 物品实体键名，默认 "item_id"
	ItemEntityKey string
}

// NewFeatureServiceAdapter 创建一个新的 FeatureService 适配器。
//
// 参数：
//   - client: Feast 客户端（基础设施层）
//   - mapping: 特征映射配置
//
// 返回：
//   - *FeatureServiceAdapter: 实现了 core.FeatureService 接口的适配器
func NewFeatureServiceAdapter(client Client, mapping *FeatureMapping) *FeatureServiceAdapter {
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
	return "feast"
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
	req := &GetOnlineFeaturesRequest{
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
	req := &GetOnlineFeaturesRequest{
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
	req := &GetOnlineFeaturesRequest{
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
	req := &GetOnlineFeaturesRequest{
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
	req := &GetOnlineFeaturesRequest{
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
	req := &GetOnlineFeaturesRequest{
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
	// http.Client 接口的 Close() 不接受 context，直接调用即可
	return a.client.Close()
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

var _ core.FeatureService = (*FeatureServiceAdapter)(nil)
