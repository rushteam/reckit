package feature

import (
	"context"
	"fmt"
	"hash/fnv"
	"strconv"
	"strings"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

// EnrichNode 是特征注入节点，将用户特征、物品特征、交叉特征组合。
// 支持千人千面的个性化推荐。
// 支持两种模式：
// 1. 传统模式：使用自定义提取器（UserFeatureExtractor、ItemFeatureExtractor）
// 2. 特征服务模式：使用 FeatureService 统一获取特征（推荐）
type EnrichNode struct {
	// FeatureService 特征服务（推荐使用，统一特征获取接口）
	// 如果设置了 FeatureService，将优先使用它获取特征
	FeatureService core.FeatureService

	// UserFeatureExtractor 从 RecommendContext 提取用户特征（传统模式）
	// 如果设置了 FeatureService，此选项将被忽略
	UserFeatureExtractor func(rctx *core.RecommendContext) map[string]float64

	// ItemFeatureExtractor 从 Item 提取物品特征（传统模式，可选，默认使用 item.Features）
	// 如果设置了 FeatureService，此选项将被忽略
	ItemFeatureExtractor func(item *core.Item) map[string]float64

	// SceneFeatureExtractor 从 RecommendContext 提取场景特征（传统模式）
	// 场景特征用于区分不同的推荐场景（如 feed、search、detail 等）
	SceneFeatureExtractor func(rctx *core.RecommendContext) map[string]float64

	// CrossFeatureExtractor 生成交叉特征（用户-物品交叉特征）
	CrossFeatureExtractor func(userFeatures map[string]float64, itemFeatures map[string]float64) map[string]float64

	// FeaturePrefix 特征前缀，用于区分不同类型的特征
	// 例如：user_*, item_*, scene_*, cross_*
	// 如果为空，使用默认值（user_, item_, scene_, cross_）
	UserFeaturePrefix  string
	ItemFeaturePrefix  string
	SceneFeaturePrefix string
	CrossFeaturePrefix string

	// KeyUserFeatures 关键用户特征列表（用于交叉特征生成）
	// 如果未设置，使用默认列表：["age", "gender", "user_id"]
	KeyUserFeatures []string

	// KeyItemFeatures 关键物品特征列表（用于交叉特征生成）
	// 如果未设置，使用默认列表：["ctr", "cvr", "price", "score"]
	KeyItemFeatures []string

	// GlobalFeatureConfig 全局特征配置（可选）
	// 如果设置，可以从全局配置读取默认前缀
	GlobalFeatureConfig *FeatureConfig
}

// FeatureConfig 是特征相关的全局配置。
type FeatureConfig struct {
	// DefaultUserPrefix 默认用户特征前缀
	DefaultUserPrefix string

	// DefaultItemPrefix 默认物品特征前缀
	DefaultItemPrefix string

	// DefaultScenePrefix 默认场景特征前缀
	DefaultScenePrefix string

	// DefaultCrossPrefix 默认交叉特征前缀
	DefaultCrossPrefix string
}

// DefaultFeatureConfig 返回默认的特征配置。
func DefaultFeatureConfig() *FeatureConfig {
	return &FeatureConfig{
		DefaultUserPrefix:  "user_",
		DefaultItemPrefix:  "item_",
		DefaultScenePrefix: "scene_",
		DefaultCrossPrefix: "cross_",
	}
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
	var err error

	// 优先使用 FeatureService
	if n.FeatureService != nil {
		userFeatures, err = n.FeatureService.GetUserFeatures(ctx, rctx.UserID)
		if err != nil {
			// 特征服务获取失败，回退到传统模式
			userFeatures = n.defaultUserFeatureExtractor(rctx)
		}
	} else if n.UserFeatureExtractor != nil {
		// 传统模式：使用自定义提取器
		userFeatures = n.UserFeatureExtractor(rctx)
	} else {
		// 默认提取器
		userFeatures = n.defaultUserFeatureExtractor(rctx)
	}

	// 用户特征前缀
	userPrefix := n.UserFeaturePrefix
	if userPrefix == "" {
		if n.GlobalFeatureConfig != nil {
			userPrefix = n.GlobalFeatureConfig.DefaultUserPrefix
		}
		if userPrefix == "" {
			userPrefix = "user_" // 最终默认值
		}
	}

	// 物品特征前缀
	itemPrefix := n.ItemFeaturePrefix
	if itemPrefix == "" {
		if n.GlobalFeatureConfig != nil {
			itemPrefix = n.GlobalFeatureConfig.DefaultItemPrefix
		}
		if itemPrefix == "" {
			itemPrefix = "item_" // 最终默认值
		}
	}

	// 场景特征前缀
	scenePrefix := n.SceneFeaturePrefix
	if scenePrefix == "" {
		if n.GlobalFeatureConfig != nil {
			scenePrefix = n.GlobalFeatureConfig.DefaultScenePrefix
		}
		if scenePrefix == "" {
			scenePrefix = "scene_" // 最终默认值
		}
	}

	// 交叉特征前缀
	crossPrefix := n.CrossFeaturePrefix
	if crossPrefix == "" {
		if n.GlobalFeatureConfig != nil {
			crossPrefix = n.GlobalFeatureConfig.DefaultCrossPrefix
		}
		if crossPrefix == "" {
			crossPrefix = "cross_" // 最终默认值
		}
	}

	// 提取场景特征
	var sceneFeatures map[string]float64
	if n.SceneFeatureExtractor != nil {
		sceneFeatures = n.SceneFeatureExtractor(rctx)
	} else {
		sceneFeatures = n.defaultSceneFeatureExtractor(rctx)
	}

	// 批量获取物品特征（如果使用 FeatureService）
	var itemFeaturesMap map[string]map[string]float64
	if n.FeatureService != nil {
		itemIDs := make([]string, 0, len(items))
		for _, item := range items {
			if item != nil {
				itemIDs = append(itemIDs, item.ID)
			}
		}
		if len(itemIDs) > 0 {
			itemFeaturesMap, _ = n.FeatureService.BatchGetItemFeatures(ctx, itemIDs)
		}
	}

	// 为每个物品注入特征
	for _, item := range items {
		if item == nil {
			continue
		}

		// 提取物品特征
		var itemFeatures map[string]float64

		// 优先使用 FeatureService
		if n.FeatureService != nil && itemFeaturesMap != nil {
			if features, ok := itemFeaturesMap[item.ID]; ok {
				itemFeatures = features
			} else {
				// 特征服务未返回该物品的特征，使用默认值
				itemFeatures = make(map[string]float64)
			}
		} else if n.ItemFeatureExtractor != nil {
			// 传统模式：使用自定义提取器
			itemFeatures = n.ItemFeatureExtractor(item)
		} else {
			// 默认：使用 item.Features
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
			if !strings.HasPrefix(k, itemPrefix) && !strings.HasPrefix(k, userPrefix) && !strings.HasPrefix(k, scenePrefix) && !strings.HasPrefix(k, crossPrefix) {
				key = itemPrefix + k
			}
			// 如果已存在，保留原值（物品特征优先）
			if _, exists := item.Features[key]; !exists {
				item.Features[key] = v
			}
		}

		// 合并场景特征（带前缀）
		for k, v := range sceneFeatures {
			key := scenePrefix + k
			item.Features[key] = v
		}

		// 生成交叉特征（包含用户、物品、场景特征）
		if n.CrossFeatureExtractor != nil {
			crossFeatures := n.CrossFeatureExtractor(userFeatures, itemFeatures)
			for k, v := range crossFeatures {
				key := crossPrefix + k
				item.Features[key] = v
			}
		} else {
			// 默认交叉特征：用户-物品-场景特征组合
			crossFeatures := n.defaultCrossFeatures(userFeatures, itemFeatures, sceneFeatures)
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
		features["user_id_str"] = 0 // 注意：这里原先是 float64(rctx.UserID)，改为 string 后不能直接转 float64，根据业务处理
		// 建议如果是数值 ID 仍可转换，否则作为分类特征
		if id, err := strconv.ParseFloat(rctx.UserID, 64); err == nil {
			features["user_id"] = id
		}

		// 从 UserProfile 提取特征
		if rctx.UserProfile != nil {
			for k, v := range rctx.UserProfile {
				if fv, ok := n.toFloat64(v); ok {
					features[k] = fv
				}
			}
		}

		// 从 Params 提取上下文特征
		if rctx.Params != nil {
			for k, v := range rctx.Params {
				if fv, ok := n.toFloat64(v); ok {
					features[k] = fv
				}
			}
		}
	}

	return features
}

// defaultSceneFeatureExtractor 默认场景特征提取器
func (n *EnrichNode) defaultSceneFeatureExtractor(rctx *core.RecommendContext) map[string]float64 {
	features := make(map[string]float64)

	if rctx != nil {
		// 场景ID哈希特征
		if rctx.Scene != "" {
			// 将场景字符串转换为数值特征
			h := fnv.New32a()
			h.Write([]byte(rctx.Scene))
			sceneHash := h.Sum32()
			features["scene_id"] = float64(sceneHash)
			features["scene_id_hash"] = float64(sceneHash % 1000) // 归一化到 0-1000

			// 场景字符串长度特征
			features["scene_len"] = float64(len(rctx.Scene))
		}

		// 从 Params 中提取场景相关特征
		if rctx.Params != nil {
			if sceneType, ok := rctx.Params["scene_type"]; ok {
				if fv, ok := n.toFloat64(sceneType); ok {
					features["scene_type"] = fv
				}
			}
			if pageType, ok := rctx.Params["page_type"]; ok {
				if fv, ok := n.toFloat64(pageType); ok {
					features["page_type"] = fv
				}
			}
		}
	}

	return features
}

// defaultCrossFeatures 默认交叉特征生成（包含场景特征）
func (n *EnrichNode) defaultCrossFeatures(userFeatures, itemFeatures, sceneFeatures map[string]float64) map[string]float64 {
	crossFeatures := make(map[string]float64)

	// 简单的交叉特征：用户特征 × 物品特征
	// 只对关键特征做交叉，避免生成过多无意义特征
	keyUserFeatures := n.KeyUserFeatures
	if len(keyUserFeatures) == 0 {
		keyUserFeatures = []string{"age", "gender", "user_id"} // 默认值
	}
	keyItemFeatures := n.KeyItemFeatures
	if len(keyItemFeatures) == 0 {
		keyItemFeatures = []string{"ctr", "cvr", "price", "score"} // 默认值
	}

	// 用户-物品交叉特征
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

	// 用户-场景交叉特征
	if sceneID, ok := sceneFeatures["scene_id"]; ok {
		for _, uk := range keyUserFeatures {
			uv, uok := userFeatures[uk]
			if !uok {
				continue
			}
			key := fmt.Sprintf("%s_x_scene", uk)
			crossFeatures[key] = uv * sceneID
		}
	}

	// 物品-场景交叉特征
	if sceneID, ok := sceneFeatures["scene_id"]; ok {
		for _, ik := range keyItemFeatures {
			iv, iok := itemFeatures[ik]
			if !iok {
				continue
			}
			key := fmt.Sprintf("%s_x_scene", ik)
			crossFeatures[key] = iv * sceneID
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
// func (n *EnrichNode) isNumericFeature(key string) bool {
// 	// 简单判断：排除明显的分类特征
// 	nonNumeric := []string{"gender", "category", "type", "status"}
// 	for _, nn := range nonNumeric {
// 		if key == nn {
// 			return false
// 		}
// 	}
// 	return true
// }
