package builders

import (
	"fmt"
	"time"

	"github.com/rushteam/reckit/config"
	"github.com/rushteam/reckit/feature"
	"github.com/rushteam/reckit/filter"
	"github.com/rushteam/reckit/model"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/conv"
	"github.com/rushteam/reckit/rank"
	"github.com/rushteam/reckit/recall"
	"github.com/rushteam/reckit/rerank"
)

func init() {
	config.Register("recall.fanout", BuildFanoutNode)
	config.Register("recall.hot", BuildHotNode)
	config.Register("recall.ann", BuildANNNode)
	config.Register("rank.lr", BuildLRNode)
	config.Register("rank.rpc", BuildRPCNode)
	config.Register("rerank.diversity", BuildDiversityNode)
	config.Register("filter", BuildFilterNode)
	config.Register("feature.enrich", BuildFeatureEnrichNode)
}

func BuildFanoutNode(cfg map[string]interface{}) (pipeline.Node, error) {
	sourcesConfig, ok := cfg["sources"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("sources not found or invalid")
	}
	sources := make([]recall.Source, 0, len(sourcesConfig))
	for _, sc := range sourcesConfig {
		sourceMap, ok := sc.(map[string]interface{})
		if !ok {
			continue
		}
		sourceType := conv.ConfigGet(sourceMap, "type", "")
		switch sourceType {
		case "hot":
			ids := conv.SliceAnyToString(sourceMap["ids"])
			if ids == nil {
				ids = []string{}
			}
			sources = append(sources, &recall.Hot{IDs: ids})
		case "ann":
			// ANN 需 VectorService，暂未从配置构建
		default:
			return nil, fmt.Errorf("unknown source type: %s", sourceType)
		}
	}
	fanout := &recall.Fanout{
		Sources: sources,
		Dedup:   conv.ConfigGet(cfg, "dedup", true),
	}
	if sec := conv.ConfigGetInt64(cfg, "timeout", 0); sec > 0 {
		fanout.Timeout = time.Duration(sec) * time.Second
	}
	if n := conv.ConfigGetInt64(cfg, "max_concurrent", 0); n > 0 {
		fanout.MaxConcurrent = int(n)
	}
	switch conv.ConfigGet(cfg, "merge_strategy", "") {
	case "priority":
		fanout.MergeStrategy = &recall.PriorityMergeStrategy{}
	case "union":
		fanout.MergeStrategy = &recall.UnionMergeStrategy{}
	default:
		fanout.MergeStrategy = &recall.FirstMergeStrategy{}
	}
	return fanout, nil
}

func BuildHotNode(cfg map[string]interface{}) (pipeline.Node, error) {
	ids := conv.SliceAnyToString(cfg["ids"])
	if ids == nil {
		ids = []string{}
	}
	return &recall.Hot{IDs: ids}, nil
}

func BuildANNNode(cfg map[string]interface{}) (pipeline.Node, error) {
	return nil, fmt.Errorf("ann node not fully implemented (supported: recall.fanout, recall.hot, rank.lr, rank.rpc, rerank.diversity, filter, feature.enrich)")
}

func BuildLRNode(cfg map[string]interface{}) (pipeline.Node, error) {
	weightsMap, ok := cfg["weights"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("weights not found")
	}
	weights := conv.MapToFloat64(weightsMap)
	bias := conv.ConfigGet(cfg, "bias", 0.0)
	lr := &model.LRModel{Bias: bias, Weights: weights}
	return &rank.LRNode{Model: lr}, nil
}

func BuildRPCNode(cfg map[string]interface{}) (pipeline.Node, error) {
	endpoint := conv.ConfigGet(cfg, "endpoint", "")
	if endpoint == "" {
		return nil, fmt.Errorf("endpoint not found")
	}
	timeout := 5 * time.Second
	if sec := conv.ConfigGetInt64(cfg, "timeout", 5); sec > 0 {
		timeout = time.Duration(sec) * time.Second
	}
	modelType := conv.ConfigGet(cfg, "model_type", "rpc")
	if modelType == "" {
		modelType = "rpc"
	}
	rpcModel := model.NewRPCModel(modelType, endpoint, timeout)
	return &rank.RPCNode{Model: rpcModel}, nil
}

func BuildDiversityNode(cfg map[string]interface{}) (pipeline.Node, error) {
	labelKey := conv.ConfigGet(cfg, "label_key", "category")
	if labelKey == "" {
		labelKey = "category"
	}
	return &rerank.Diversity{LabelKey: labelKey}, nil
}

func BuildFilterNode(cfg map[string]interface{}) (pipeline.Node, error) {
	filtersConfig, ok := cfg["filters"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("filters not found or invalid")
	}
	filters := make([]filter.Filter, 0, len(filtersConfig))
	for _, fc := range filtersConfig {
		filterMap, ok := fc.(map[string]interface{})
		if !ok {
			continue
		}
		filterType := conv.ConfigGet(filterMap, "type", "")
		switch filterType {
		case "blacklist":
			ids := conv.SliceAnyToString(filterMap["item_ids"])
			if ids == nil {
				ids = []string{}
			}
			key := conv.ConfigGet(filterMap, "key", "")
			filters = append(filters, filter.NewBlacklistFilter(ids, nil, key))
		case "user_block":
			keyPrefix := conv.ConfigGet(filterMap, "key_prefix", "")
			filters = append(filters, filter.NewUserBlockFilter(nil, keyPrefix))
		case "exposed":
			keyPrefix := conv.ConfigGet(filterMap, "key_prefix", "")
			timeWindow := conv.ConfigGetInt64(filterMap, "time_window", 0)
			bloomFilterDayWindow := conv.ConfigGet(filterMap, "bloom_filter_day_window", 0)
			filters = append(filters, filter.NewExposedFilter(nil, keyPrefix, timeWindow, bloomFilterDayWindow))
		default:
			return nil, fmt.Errorf("unknown filter type: %s", filterType)
		}
	}
	return &filter.FilterNode{Filters: filters}, nil
}

func BuildFeatureEnrichNode(cfg map[string]interface{}) (pipeline.Node, error) {
	return &feature.EnrichNode{
		UserFeaturePrefix:  conv.ConfigGet(cfg, "user_feature_prefix", ""),
		ItemFeaturePrefix:  conv.ConfigGet(cfg, "item_feature_prefix", ""),
		CrossFeaturePrefix: conv.ConfigGet(cfg, "cross_feature_prefix", ""),
	}, nil
}
