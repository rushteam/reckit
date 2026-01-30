package feast

import (
	"context"
	"fmt"

	"github.com/rushteam/reckit/core"
)

// FeatureServiceAdapter 将 Feast Client 适配为 core.FeatureService 接口。
//
// 使用方式：通过适配器将 Feast 适配为 core.FeatureService。
type FeatureServiceAdapter struct {
	client         Client
	featureMapping *FeatureMapping
}

// NewFeatureServiceAdapter 创建一个新的 FeatureService 适配器。
// client 可为 NewHTTPClient 或 NewGrpcClient 返回的客户端。
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
	entityRows := []map[string]interface{}{{a.featureMapping.UserEntityKey: userID}}
	req := &GetOnlineFeaturesRequest{Features: a.featureMapping.UserFeatures, EntityRows: entityRows}
	resp, err := a.client.GetOnlineFeatures(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("feast get user features failed: %w", err)
	}
	if len(resp.FeatureVectors) == 0 {
		return make(map[string]float64), nil
	}
	features := make(map[string]float64)
	for k, v := range resp.FeatureVectors[0].Values {
		if fv, ok := valueToFloat64(v); ok {
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
	entityRows := make([]map[string]interface{}, len(userIDs))
	for i, userID := range userIDs {
		entityRows[i] = map[string]interface{}{a.featureMapping.UserEntityKey: userID}
	}
	req := &GetOnlineFeaturesRequest{Features: a.featureMapping.UserFeatures, EntityRows: entityRows}
	resp, err := a.client.GetOnlineFeatures(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("feast batch get user features failed: %w", err)
	}
	result := make(map[string]map[string]float64)
	for i, fv := range resp.FeatureVectors {
		if i >= len(userIDs) {
			break
		}
		features := make(map[string]float64)
		for k, v := range fv.Values {
			if fv, ok := valueToFloat64(v); ok {
				features[k] = fv
			}
		}
		result[userIDs[i]] = features
	}
	return result, nil
}

// GetItemFeatures 获取物品特征
func (a *FeatureServiceAdapter) GetItemFeatures(ctx context.Context, itemID string) (map[string]float64, error) {
	if len(a.featureMapping.ItemFeatures) == 0 {
		return make(map[string]float64), nil
	}
	entityRows := []map[string]interface{}{{a.featureMapping.ItemEntityKey: itemID}}
	req := &GetOnlineFeaturesRequest{Features: a.featureMapping.ItemFeatures, EntityRows: entityRows}
	resp, err := a.client.GetOnlineFeatures(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("feast get item features failed: %w", err)
	}
	if len(resp.FeatureVectors) == 0 {
		return make(map[string]float64), nil
	}
	features := make(map[string]float64)
	for k, v := range resp.FeatureVectors[0].Values {
		if fv, ok := valueToFloat64(v); ok {
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
	entityRows := make([]map[string]interface{}, len(itemIDs))
	for i, itemID := range itemIDs {
		entityRows[i] = map[string]interface{}{a.featureMapping.ItemEntityKey: itemID}
	}
	req := &GetOnlineFeaturesRequest{Features: a.featureMapping.ItemFeatures, EntityRows: entityRows}
	resp, err := a.client.GetOnlineFeatures(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("feast batch get item features failed: %w", err)
	}
	result := make(map[string]map[string]float64)
	for i, fv := range resp.FeatureVectors {
		if i >= len(itemIDs) {
			break
		}
		features := make(map[string]float64)
		for k, v := range fv.Values {
			if fv, ok := valueToFloat64(v); ok {
				features[k] = fv
			}
		}
		result[itemIDs[i]] = features
	}
	return result, nil
}

// Close 关闭特征服务
func (a *FeatureServiceAdapter) Close(ctx context.Context) error {
	return a.client.Close()
}

// valueToFloat64 将 interface{} 转为 float64（包内共用）
func valueToFloat64(v interface{}) (float64, bool) {
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
