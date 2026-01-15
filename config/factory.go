package config

import (
	"fmt"
	"time"

	"reckit/feature"
	"reckit/filter"
	"reckit/model"
	"reckit/pipeline"
	"reckit/rank"
	"reckit/recall"
	"reckit/rerank"
)

// DefaultFactory 返回一个包含所有内置 Node 的默认工厂。
func DefaultFactory() *pipeline.NodeFactory {
	factory := pipeline.NewNodeFactory()

	// 注册 Recall Nodes
	factory.Register("recall.fanout", buildFanoutNode)
	factory.Register("recall.hot", buildHotNode)
	factory.Register("recall.ann", buildANNNode)

	// 注册 Rank Nodes
	factory.Register("rank.lr", buildLRNode)
	factory.Register("rank.rpc", buildRPCNode)

	// 注册 ReRank Nodes
	factory.Register("rerank.diversity", buildDiversityNode)

	// 注册 Filter Nodes
	factory.Register("filter", buildFilterNode)

	// 注册 Feature Nodes
	factory.Register("feature.enrich", buildFeatureEnrichNode)

	return factory
}

func buildFanoutNode(config map[string]interface{}) (pipeline.Node, error) {
	sourcesConfig, ok := config["sources"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("sources not found or invalid")
	}

	sources := make([]recall.Source, 0, len(sourcesConfig))
	for _, sc := range sourcesConfig {
		sourceMap, ok := sc.(map[string]interface{})
		if !ok {
			continue
		}
		sourceType, _ := sourceMap["type"].(string)
		switch sourceType {
		case "hot":
			ids := []string{}
			if idsRaw, ok := sourceMap["ids"].([]interface{}); ok {
				for _, id := range idsRaw {
					// 支持字符串和数字两种格式
					if idStr, ok := id.(string); ok {
						ids = append(ids, idStr)
					} else if idFloat, ok := id.(float64); ok {
						ids = append(ids, fmt.Sprintf("%.0f", idFloat))
					}
				}
			}
			sources = append(sources, &recall.Hot{IDs: ids})
		case "ann":
			// ANN 需要 VectorStore，这里简化处理
			// sources = append(sources, &recall.ANN{...})
		default:
			return nil, fmt.Errorf("unknown source type: %s", sourceType)
		}
	}

	fanout := &recall.Fanout{
		Sources: sources,
		Dedup:   getBool(config, "dedup", true),
	}

	if timeout, ok := config["timeout"].(int); ok {
		fanout.Timeout = time.Duration(timeout) * time.Second
	}
	if maxConcurrent, ok := config["max_concurrent"].(int); ok {
		fanout.MaxConcurrent = maxConcurrent
	}
	if mergeStrategy, ok := config["merge_strategy"].(string); ok {
		fanout.MergeStrategyName = mergeStrategy
	}

	return fanout, nil
}

func buildHotNode(config map[string]interface{}) (pipeline.Node, error) {
	ids := []string{}
	if idsRaw, ok := config["ids"].([]interface{}); ok {
		for _, id := range idsRaw {
			// 支持字符串和数字两种格式
			if idStr, ok := id.(string); ok {
				ids = append(ids, idStr)
			} else if idFloat, ok := id.(float64); ok {
				ids = append(ids, fmt.Sprintf("%.0f", idFloat))
			}
		}
	}
	return &recall.Hot{IDs: ids}, nil
}

func buildANNNode(config map[string]interface{}) (pipeline.Node, error) {
	// 简化实现，实际需要配置 VectorStore
	return nil, fmt.Errorf("ann node not fully implemented in factory")
}

func buildLRNode(config map[string]interface{}) (pipeline.Node, error) {
	weightsMap, ok := config["weights"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("weights not found")
	}

	weights := make(map[string]float64)
	for k, v := range weightsMap {
		if vFloat, ok := v.(float64); ok {
			weights[k] = vFloat
		}
	}

	bias := getFloat(config, "bias", 0.0)
	lr := &model.LRModel{
		Bias:    bias,
		Weights: weights,
	}

	return &rank.LRNode{Model: lr}, nil
}

func buildRPCNode(config map[string]interface{}) (pipeline.Node, error) {
	endpoint, ok := config["endpoint"].(string)
	if !ok {
		return nil, fmt.Errorf("endpoint not found")
	}

	timeout := 5 * time.Second
	if timeoutRaw, ok := config["timeout"].(int); ok {
		timeout = time.Duration(timeoutRaw) * time.Second
	}

	modelType, _ := config["model_type"].(string)
	if modelType == "" {
		modelType = "rpc"
	}
	rpcModel := model.NewRPCModel(modelType, endpoint, timeout)

	return &rank.RPCNode{Model: rpcModel}, nil
}

func buildDiversityNode(config map[string]interface{}) (pipeline.Node, error) {
	labelKey, _ := config["label_key"].(string)
	if labelKey == "" {
		labelKey = "category"
	}
	return &rerank.Diversity{LabelKey: labelKey}, nil
}

func getBool(config map[string]interface{}, key string, defaultValue bool) bool {
	if v, ok := config[key].(bool); ok {
		return v
	}
	return defaultValue
}

func buildFilterNode(config map[string]interface{}) (pipeline.Node, error) {
	filtersConfig, ok := config["filters"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("filters not found or invalid")
	}

	filters := make([]filter.Filter, 0, len(filtersConfig))
	for _, fc := range filtersConfig {
		filterMap, ok := fc.(map[string]interface{})
		if !ok {
			continue
		}
		filterType, _ := filterMap["type"].(string)

		switch filterType {
		case "blacklist":
			ids := []string{}
			if idsRaw, ok := filterMap["item_ids"].([]interface{}); ok {
				for _, id := range idsRaw {
					// 支持字符串和数字两种格式
					if idStr, ok := id.(string); ok {
						ids = append(ids, idStr)
					} else if idFloat, ok := id.(float64); ok {
						ids = append(ids, fmt.Sprintf("%.0f", idFloat))
					}
				}
			}
			key, _ := filterMap["key"].(string)
			filters = append(filters, filter.NewBlacklistFilter(ids, nil, key))

		case "user_block":
			keyPrefix, _ := filterMap["key_prefix"].(string)
			filters = append(filters, filter.NewUserBlockFilter(nil, keyPrefix))

		case "exposed":
			keyPrefix, _ := filterMap["key_prefix"].(string)
			timeWindow := int64(0)
			if tw, ok := filterMap["time_window"].(int); ok {
				timeWindow = int64(tw)
			}
			filters = append(filters, filter.NewExposedFilter(nil, keyPrefix, timeWindow))

		default:
			return nil, fmt.Errorf("unknown filter type: %s", filterType)
		}
	}

	return &filter.FilterNode{Filters: filters}, nil
}

func buildFeatureEnrichNode(config map[string]interface{}) (pipeline.Node, error) {
	userPrefix, _ := config["user_feature_prefix"].(string)
	itemPrefix, _ := config["item_feature_prefix"].(string)
	crossPrefix, _ := config["cross_feature_prefix"].(string)

	return &feature.EnrichNode{
		UserFeaturePrefix:  userPrefix,
		ItemFeaturePrefix:  itemPrefix,
		CrossFeaturePrefix: crossPrefix,
	}, nil
}

func getFloat(config map[string]interface{}, key string, defaultValue float64) float64 {
	if v, ok := config[key].(float64); ok {
		return v
	}
	return defaultValue
}
