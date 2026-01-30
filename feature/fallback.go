package feature

import (
	"context"

	"github.com/rushteam/reckit/core"
)

// DefaultFallbackStrategy 是默认降级策略，当特征服务不可用时，
// 从 RecommendContext 和 Item 中提取基础特征。
type DefaultFallbackStrategy struct{}

func NewDefaultFallbackStrategy() *DefaultFallbackStrategy {
	return &DefaultFallbackStrategy{}
}

func (f *DefaultFallbackStrategy) GetUserFeatures(ctx context.Context, userID string, rctx *core.RecommendContext) (map[string]float64, error) {
	features := make(map[string]float64)

	if rctx == nil {
		return features, nil
	}

	// 基础用户特征
	// 注意：userID 是 string 类型，不能直接作为 float64 特征
	// 如果需要用户 ID 特征，可以通过 hash 或其他方式转换为数值

	// 从 UserProfile 提取
	if rctx.UserProfile != nil {
		for k, v := range rctx.UserProfile {
			if fv, ok := toFloat64(v); ok {
				features[k] = fv
			}
		}
	}

	// 从 Params 提取
	if rctx.Params != nil {
		for k, v := range rctx.Params {
			if fv, ok := toFloat64(v); ok {
				features[k] = fv
			}
		}
	}

	return features, nil
}

func (f *DefaultFallbackStrategy) GetItemFeatures(ctx context.Context, itemID string, item *core.Item) (map[string]float64, error) {
	features := make(map[string]float64)

	if item == nil {
		return features, nil
	}

	// 基础物品特征
	// 注意：itemID 是 string 类型，不能直接作为 float64 特征
	// 如果需要物品 ID 特征，可以通过 hash 或其他方式转换为数值

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

// GetCrossFeatures 获取交叉特征（用户-物品交互特征）
func (f *DefaultFallbackStrategy) GetCrossFeatures(ctx context.Context, userID, itemID string, rctx *core.RecommendContext, item *core.Item) (map[string]float64, error) {
	features := make(map[string]float64)

	// 组合用户和物品的基础特征
	userFeatures, _ := f.GetUserFeatures(ctx, userID, rctx)
	itemFeatures, _ := f.GetItemFeatures(ctx, itemID, item)

	// 简单的交叉特征：用户-物品交互
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
