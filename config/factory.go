package config

import (
	"fmt"
	"time"

	"github.com/rushteam/reckit/feature"
	"github.com/rushteam/reckit/filter"
	"github.com/rushteam/reckit/model"
	"github.com/rushteam/reckit/pkg/conv"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/rank"
	"github.com/rushteam/reckit/recall"
	"github.com/rushteam/reckit/rerank"
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
		sourceType := conv.ConfigGet[string](sourceMap, "type", "")
		switch sourceType {
		case "hot":
			ids := conv.SliceAnyToString(sourceMap["ids"])
			if ids == nil {
				ids = []string{}
			}
			sources = append(sources, &recall.Hot{IDs: ids})
		case "ann":
			// ANN 需要 core.VectorService，这里简化处理
			// sources = append(sources, &recall.ANN{
			//     VectorService: vectorService,
			//     Collection:    "items",
			//     TopK:          20,
			//     Metric:        "cosine",
			// })
		default:
			return nil, fmt.Errorf("unknown source type: %s", sourceType)
		}
	}

	fanout := &recall.Fanout{
		Sources: sources,
		Dedup:   conv.ConfigGet[bool](config, "dedup", true),
	}
	if sec := conv.ConfigGetInt64(config, "timeout", 0); sec > 0 {
		fanout.Timeout = time.Duration(sec) * time.Second
	}
	if n := conv.ConfigGetInt64(config, "max_concurrent", 0); n > 0 {
		fanout.MaxConcurrent = int(n)
	}
	switch conv.ConfigGet[string](config, "merge_strategy", "") {
	case "priority":
		fanout.MergeStrategy = &recall.PriorityMergeStrategy{}
	case "union":
		fanout.MergeStrategy = &recall.UnionMergeStrategy{}
	default:
		fanout.MergeStrategy = &recall.FirstMergeStrategy{}
	}

	return fanout, nil
}

func buildHotNode(config map[string]interface{}) (pipeline.Node, error) {
	ids := conv.SliceAnyToString(config["ids"])
	if ids == nil {
		ids = []string{}
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
	weights := conv.MapToFloat64(weightsMap)
	bias := conv.ConfigGet[float64](config, "bias", 0.0)
	lr := &model.LRModel{
		Bias:    bias,
		Weights: weights,
	}
	return &rank.LRNode{Model: lr}, nil
}

func buildRPCNode(config map[string]interface{}) (pipeline.Node, error) {
	endpoint := conv.ConfigGet[string](config, "endpoint", "")
	if endpoint == "" {
		return nil, fmt.Errorf("endpoint not found")
	}
	timeout := 5 * time.Second
	if sec := conv.ConfigGetInt64(config, "timeout", 5); sec > 0 {
		timeout = time.Duration(sec) * time.Second
	}
	modelType := conv.ConfigGet[string](config, "model_type", "rpc")
	if modelType == "" {
		modelType = "rpc"
	}
	rpcModel := model.NewRPCModel(modelType, endpoint, timeout)
	return &rank.RPCNode{Model: rpcModel}, nil
}

func buildDiversityNode(config map[string]interface{}) (pipeline.Node, error) {
	labelKey := conv.ConfigGet[string](config, "label_key", "category")
	if labelKey == "" {
		labelKey = "category"
	}
	return &rerank.Diversity{LabelKey: labelKey}, nil
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
		filterType := conv.ConfigGet[string](filterMap, "type", "")
		switch filterType {
		case "blacklist":
			ids := conv.SliceAnyToString(filterMap["item_ids"])
			if ids == nil {
				ids = []string{}
			}
			key := conv.ConfigGet[string](filterMap, "key", "")
			filters = append(filters, filter.NewBlacklistFilter(ids, nil, key))

		case "user_block":
			keyPrefix := conv.ConfigGet[string](filterMap, "key_prefix", "")
			filters = append(filters, filter.NewUserBlockFilter(nil, keyPrefix))

		case "exposed":
			keyPrefix := conv.ConfigGet[string](filterMap, "key_prefix", "")
			timeWindow := conv.ConfigGetInt64(filterMap, "time_window", 0)
			bloomFilterDayWindow := conv.ConfigGet[int](filterMap, "bloom_filter_day_window", 0)
			filters = append(filters, filter.NewExposedFilter(nil, keyPrefix, timeWindow, bloomFilterDayWindow))

		default:
			return nil, fmt.Errorf("unknown filter type: %s", filterType)
		}
	}

	return &filter.FilterNode{Filters: filters}, nil
}

func buildFeatureEnrichNode(config map[string]interface{}) (pipeline.Node, error) {
	return &feature.EnrichNode{
		UserFeaturePrefix:  conv.ConfigGet[string](config, "user_feature_prefix", ""),
		ItemFeaturePrefix:  conv.ConfigGet[string](config, "item_feature_prefix", ""),
		CrossFeaturePrefix: conv.ConfigGet[string](config, "cross_feature_prefix", ""),
	}, nil
}
