package feature

import (
	"context"

	"reckit/core"
)

// DefaultFallbackStrategy 是默认降级策略，当特征服务不可用时，
// 从 RecommendContext 和 Item 中提取基础特征。
type DefaultFallbackStrategy struct{}

func NewDefaultFallbackStrategy() *DefaultFallbackStrategy {
	return &DefaultFallbackStrategy{}
}

func (f *DefaultFallbackStrategy) GetUserFeatures(ctx context.Context, userID int64, rctx *core.RecommendContext) (map[string]float64, error) {
	features := make(map[string]float64)

	if rctx == nil {
		return features, nil
	}

	// 基础用户特征
	features["user_id"] = float64(userID)

	// 从 UserProfile 提取
	if rctx.UserProfile != nil {
		for k, v := range rctx.UserProfile {
			if fv, ok := toFloat64(v); ok {
				features[k] = fv
			}
		}
	}

	// 从 Realtime 提取
	if rctx.Realtime != nil {
		for k, v := range rctx.Realtime {
			if fv, ok := toFloat64(v); ok {
				features["realtime_"+k] = fv
			}
		}
	}

	return features, nil
}

func (f *DefaultFallbackStrategy) GetItemFeatures(ctx context.Context, itemID int64, item *core.Item) (map[string]float64, error) {
	features := make(map[string]float64)

	if item == nil {
		return features, nil
	}

	// 基础物品特征
	features["item_id"] = float64(itemID)

	// 从 Item.Features 提取
	if item.Features != nil {
		for k, v := range item.Features {
			features[k] = v
		}
	}

	// 从 Item.Meta 提取
	if item.Meta != nil {
		for k, v := range item.Meta {
			if fv, ok := toFloat64(v); ok {
				features["meta_"+k] = fv
			}
		}
	}

	return features, nil
}

func (f *DefaultFallbackStrategy) GetRealtimeFeatures(ctx context.Context, userID, itemID int64, rctx *core.RecommendContext, item *core.Item) (map[string]float64, error) {
	features := make(map[string]float64)

	// 组合用户和物品的基础特征
	userFeatures, _ := f.GetUserFeatures(ctx, userID, rctx)
	itemFeatures, _ := f.GetItemFeatures(ctx, itemID, item)

	// 简单的实时特征：用户-物品交互
	for uk, uv := range userFeatures {
		for ik, iv := range itemFeatures {
			// 只对关键特征做交叉
			if isKeyFeature(uk) && isKeyFeature(ik) {
				key := uk + "_x_" + ik
				features[key] = uv * iv
			}
		}
	}

	return features, nil
}

// toFloat64 将值转换为 float64
func toFloat64(v interface{}) (float64, bool) {
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
	case bool:
		if val {
			return 1.0, true
		}
		return 0.0, true
	default:
		return 0, false
	}
}

// isKeyFeature 判断是否为关键特征
func isKeyFeature(name string) bool {
	keyFeatures := []string{"age", "gender", "user_id", "ctr", "cvr", "price", "score"}
	for _, kf := range keyFeatures {
		if name == kf {
			return true
		}
	}
	return false
}
