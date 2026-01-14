package feature

import (
	"context"
	"fmt"
	"strings"

	"reckit/core"
	"reckit/pipeline"
)

// EnrichNode 是特征注入节点，将用户特征、物品特征、交叉特征组合。
// 支持千人千面的个性化推荐。
type EnrichNode struct {
	// UserFeatureExtractor 从 RecommendContext 提取用户特征
	UserFeatureExtractor func(rctx *core.RecommendContext) map[string]float64

	// ItemFeatureExtractor 从 Item 提取物品特征（可选，默认使用 item.Features）
	ItemFeatureExtractor func(item *core.Item) map[string]float64

	// CrossFeatureExtractor 生成交叉特征（用户-物品交叉特征）
	CrossFeatureExtractor func(userFeatures map[string]float64, itemFeatures map[string]float64) map[string]float64

	// FeaturePrefix 特征前缀，用于区分不同类型的特征
	// 例如：user_*, item_*, cross_*
	UserFeaturePrefix  string
	ItemFeaturePrefix  string
	CrossFeaturePrefix string
}

func (n *EnrichNode) Name() string {
	return "feature.enrich"
}

func (n *EnrichNode) Kind() pipeline.Kind {
	return pipeline.KindPostProcess
}

func (n *EnrichNode) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if len(items) == 0 {
		return items, nil
	}

	// 提取用户特征
	var userFeatures map[string]float64
	if n.UserFeatureExtractor != nil {
		userFeatures = n.UserFeatureExtractor(rctx)
	} else {
		userFeatures = n.defaultUserFeatureExtractor(rctx)
	}

	// 用户特征前缀
	userPrefix := n.UserFeaturePrefix
	if userPrefix == "" {
		userPrefix = "user_"
	}

	// 物品特征前缀
	itemPrefix := n.ItemFeaturePrefix
	if itemPrefix == "" {
		itemPrefix = "item_"
	}

	// 交叉特征前缀
	crossPrefix := n.CrossFeaturePrefix
	if crossPrefix == "" {
		crossPrefix = "cross_"
	}

	// 为每个物品注入特征
	for _, item := range items {
		if item == nil {
			continue
		}

		// 提取物品特征
		var itemFeatures map[string]float64
		if n.ItemFeatureExtractor != nil {
			itemFeatures = n.ItemFeatureExtractor(item)
		} else {
			itemFeatures = item.Features
			if itemFeatures == nil {
				itemFeatures = make(map[string]float64)
			}
		}

		// 合并用户特征（带前缀）
		for k, v := range userFeatures {
			key := userPrefix + k
			item.Features[key] = v
		}

		// 合并物品特征（带前缀）
		// 注意：避免重复添加前缀（如果特征名已包含前缀则不再添加）
		for k, v := range itemFeatures {
			key := k
			// 如果特征名不包含前缀，则添加前缀
			if !strings.HasPrefix(k, itemPrefix) && !strings.HasPrefix(k, userPrefix) && !strings.HasPrefix(k, crossPrefix) {
				key = itemPrefix + k
			}
			// 如果已存在，保留原值（物品特征优先）
			if _, exists := item.Features[key]; !exists {
				item.Features[key] = v
			}
		}

		// 生成交叉特征
		if n.CrossFeatureExtractor != nil {
			crossFeatures := n.CrossFeatureExtractor(userFeatures, itemFeatures)
			for k, v := range crossFeatures {
				key := crossPrefix + k
				item.Features[key] = v
			}
		} else {
			// 默认交叉特征：用户-物品特征组合
			crossFeatures := n.defaultCrossFeatures(userFeatures, itemFeatures)
			for k, v := range crossFeatures {
				key := crossPrefix + k
				item.Features[key] = v
			}
		}
	}

	return items, nil
}

// defaultUserFeatureExtractor 默认用户特征提取器
func (n *EnrichNode) defaultUserFeatureExtractor(rctx *core.RecommendContext) map[string]float64 {
	features := make(map[string]float64)

	// 基础用户特征
	if rctx != nil {
		features["user_id"] = float64(rctx.UserID)

		// 从 UserProfile 提取特征
		if rctx.UserProfile != nil {
			for k, v := range rctx.UserProfile {
				if fv, ok := n.toFloat64(v); ok {
					features[k] = fv
				}
			}
		}

		// 从 Realtime 提取实时特征
		if rctx.Realtime != nil {
			for k, v := range rctx.Realtime {
				if fv, ok := n.toFloat64(v); ok {
					features["realtime_"+k] = fv
				}
			}
		}
	}

	return features
}

// defaultCrossFeatures 默认交叉特征生成
func (n *EnrichNode) defaultCrossFeatures(userFeatures, itemFeatures map[string]float64) map[string]float64 {
	crossFeatures := make(map[string]float64)

	// 简单的交叉特征：用户特征 × 物品特征
	// 只对关键特征做交叉，避免生成过多无意义特征
	keyUserFeatures := []string{"age", "gender", "user_id"}
	keyItemFeatures := []string{"ctr", "cvr", "price", "score"}

	for _, uk := range keyUserFeatures {
		uv, uok := userFeatures[uk]
		if !uok {
			continue
		}
		for _, ik := range keyItemFeatures {
			iv, iok := itemFeatures[ik]
			if !iok {
				continue
			}
			// 生成交叉特征
			key := fmt.Sprintf("%s_x_%s", uk, ik)
			crossFeatures[key] = uv * iv
		}
	}

	return crossFeatures
}

// toFloat64 将值转换为 float64
func (n *EnrichNode) toFloat64(v interface{}) (float64, bool) {
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

// isNumericFeature 判断特征名是否为数值特征
func (n *EnrichNode) isNumericFeature(key string) bool {
	// 简单判断：排除明显的分类特征
	nonNumeric := []string{"gender", "category", "type", "status"}
	for _, nn := range nonNumeric {
		if key == nn {
			return false
		}
	}
	return true
}
