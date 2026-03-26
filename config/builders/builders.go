package builders

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/feature"
	"github.com/rushteam/reckit/filter"
	"github.com/rushteam/reckit/model"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/conv"
	"github.com/rushteam/reckit/postprocess"
	"github.com/rushteam/reckit/rank"
	"github.com/rushteam/reckit/recall"
	"github.com/rushteam/reckit/rerank"
	"github.com/rushteam/reckit/service"
)

var (
	dependencyMu                 sync.RWMutex
	configuredFilterStore        core.Store
	configuredBloomFilterChecker filter.BloomFilterChecker
	configuredFeatureService     core.FeatureService
)

// Dependencies 是内置 builders 的实例级依赖集合。
// 推荐在创建 NodeFactory 时通过 NewFactory/ApplyBuiltins 绑定，避免使用全局可变状态。
type Dependencies struct {
	FilterStore          core.Store
	BloomFilterChecker   filter.BloomFilterChecker
	BatchExposureChecker filter.BatchExposureChecker
	FeatureService       core.FeatureService
	// TrafficPlanner 用于 rerank.traffic_plan（planner: "inject" 时必填）。
	TrafficPlanner rerank.TrafficPlanner
	// ScoreWeightProvider 用于 rerank.score_weight（必填）。
	ScoreWeightProvider rerank.ScoreWeightProvider
	// FrequencyCapStore 用于 filter.frequency_cap。
	FrequencyCapStore filter.FrequencyCapStore
	// BanditStatsProvider 用于 rerank.ucb / rerank.thompson_sampling / rerank.cold_start_boost。
	BanditStatsProvider rerank.BanditStatsProvider
	// VectorService 用于 recall.ann 等需要向量搜索的召回源。
	VectorService core.VectorService
	// PaddingFunc 用于 postprocess.padding 的动态补足策略。
	PaddingFunc postprocess.PaddingFunc
}

func (d Dependencies) filterStoreAdapter() *filter.StoreAdapter {
	if d.FilterStore == nil {
		return nil
	}
	if d.BloomFilterChecker != nil {
		return filter.NewStoreAdapterWithBloomFilter(d.FilterStore, d.BloomFilterChecker)
	}
	return filter.NewStoreAdapter(d.FilterStore)
}

// ApplyBuiltins 将内置 Node builders 注册到指定 factory（实例级依赖绑定）。
func ApplyBuiltins(factory *pipeline.NodeFactory, deps Dependencies) {
	if factory == nil {
		return
	}
	factory.Register("recall.fanout", BuildFanoutNode)
	factory.Register("recall.hot", BuildHotNode)
	factory.Register("recall.ann", BuildANNNode)
	factory.Register("rank.lr", BuildLRNode)
	factory.Register("rank.rpc", BuildRPCNode)
	factory.Register("rank.wide_deep", BuildWideDeepNode)
	factory.Register("rank.two_tower", BuildTwoTowerNode)
	factory.Register("rank.dnn", BuildDNNNode)
	factory.Register("rank.din", BuildDINNode)
	factory.Register("rerank.diversity", BuildDiversityNode)
	factory.Register("rerank.mmoe", BuildMMoENode)
	factory.Register("rerank.topn", BuildTopNNode)
	factory.Register("rerank.traffic_plan", func(cfg map[string]interface{}) (pipeline.Node, error) {
		return buildTrafficPlanNodeWithDeps(cfg, deps)
	})
	factory.Register("rerank.score_adjust", func(cfg map[string]interface{}) (pipeline.Node, error) {
		return buildScoreAdjustNodeWithDeps(cfg, deps)
	})
	factory.Register("rerank.score_weight", func(cfg map[string]interface{}) (pipeline.Node, error) {
		return buildScoreWeightBoostNodeWithDeps(cfg, deps)
	})
	factory.Register("rerank.recall_channel_mix", func(cfg map[string]interface{}) (pipeline.Node, error) {
		return buildRecallChannelMixNodeWithDeps(cfg, deps)
	})
	factory.Register("rerank.dpp_diversity", buildDPPDiversityNode)
	factory.Register("rerank.ssd_diversity", buildSSDDiversityNode)
	factory.Register("rerank.sample", buildSampleNode)
	factory.Register("rerank.fair_interleave", buildFairInterleaveNode)
	factory.Register("rerank.weighted_interleave", buildWeightedInterleaveNode)
	factory.Register("rerank.group_quota", buildGroupQuotaNode)
	factory.Register("rerank.epsilon_greedy", buildEpsilonGreedyNode)
	factory.Register("rerank.ucb", func(cfg map[string]interface{}) (pipeline.Node, error) {
		return buildUCBNodeWithDeps(cfg, deps)
	})
	factory.Register("rerank.thompson_sampling", func(cfg map[string]interface{}) (pipeline.Node, error) {
		return buildThompsonSamplingNodeWithDeps(cfg, deps)
	})
	factory.Register("rerank.cold_start_boost", func(cfg map[string]interface{}) (pipeline.Node, error) {
		return buildColdStartBoostNodeWithDeps(cfg, deps)
	})
	factory.Register("filter.conditional", func(cfg map[string]interface{}) (pipeline.Node, error) {
		return buildConditionalNodeWithDeps(cfg, deps, factory)
	})
	factory.Register("filter", func(cfg map[string]interface{}) (pipeline.Node, error) {
		return buildFilterNodeWithDeps(cfg, deps)
	})
	factory.Register("feature.enrich", func(cfg map[string]interface{}) (pipeline.Node, error) {
		return buildFeatureEnrichNodeWithDeps(cfg, deps)
	})
	factory.Register("postprocess.padding", func(cfg map[string]interface{}) (pipeline.Node, error) {
		return buildPaddingNodeWithDeps(cfg, deps)
	})
	factory.Register("postprocess.truncate_fields", buildTruncateFieldsNode)

	// recall sources
	factory.Register("recall.rpc", buildRPCRecallNode)
	factory.Register("recall.graph", buildGraphRecallNode)
}

// NewFactory 创建只包含内置 Node builders 的工厂（推荐）。
func NewFactory(deps Dependencies) *pipeline.NodeFactory {
	factory := pipeline.NewNodeFactory()
	ApplyBuiltins(factory, deps)
	return factory
}

func legacyDependencies() Dependencies {
	dependencyMu.RLock()
	defer dependencyMu.RUnlock()
	return Dependencies{
		FilterStore:        configuredFilterStore,
		BloomFilterChecker: configuredBloomFilterChecker,
		FeatureService:     configuredFeatureService,
	}
}

// SetFilterStore 配置 filter 节点使用的 Store 依赖。
// 若未设置，配置化 Filter 仅使用内存 item_ids，不会访问外部存储。
// Deprecated: 推荐使用 NewFactory/ApplyBuiltins 进行实例级依赖绑定。
func SetFilterStore(store core.Store) {
	dependencyMu.Lock()
	defer dependencyMu.Unlock()
	configuredFilterStore = store
}

// SetFilterBloomFilterChecker 配置 ExposedFilter 的布隆过滤器检查器（可选）。
// Deprecated: 推荐使用 NewFactory/ApplyBuiltins 进行实例级依赖绑定。
func SetFilterBloomFilterChecker(checker filter.BloomFilterChecker) {
	dependencyMu.Lock()
	defer dependencyMu.Unlock()
	configuredBloomFilterChecker = checker
}

// SetFeatureService 配置 feature.enrich 节点使用的 FeatureService（可选）。
// Deprecated: 推荐使用 NewFactory/ApplyBuiltins 进行实例级依赖绑定。
func SetFeatureService(svc core.FeatureService) {
	dependencyMu.Lock()
	defer dependencyMu.Unlock()
	configuredFeatureService = svc
}

// ResetDependencies 重置 builders 依赖（主要用于测试）。
// Deprecated: 推荐通过实例级 factory 隔离测试依赖。
func ResetDependencies() {
	dependencyMu.Lock()
	defer dependencyMu.Unlock()
	configuredFilterStore = nil
	configuredBloomFilterChecker = nil
	configuredFeatureService = nil
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
	ms, err := buildMergeStrategy(cfg)
	if err != nil {
		return nil, fmt.Errorf("merge_strategy: %w", err)
	}
	fanout.MergeStrategy = ms
	return fanout, nil
}

func buildMergeStrategy(cfg map[string]interface{}) (recall.MergeStrategy, error) {
	switch conv.ConfigGet(cfg, "merge_strategy", "") {
	case "priority":
		return &recall.PriorityMergeStrategy{}, nil
	case "union":
		return &recall.UnionMergeStrategy{}, nil
	case "weighted":
		s := &recall.WeightedScoreMergeStrategy{
			TopN:          int(conv.ConfigGetInt64(cfg, "top_n", 0)),
			DefaultWeight: conv.ConfigGet(cfg, "default_weight", 1.0),
		}
		if sw, ok := cfg["source_weights"].(map[string]interface{}); ok {
			s.SourceWeights = conv.MapToFloat64(sw)
		}
		return s, nil
	case "quota":
		s := &recall.QuotaMergeStrategy{
			DefaultQuota: int(conv.ConfigGetInt64(cfg, "default_quota", 0)),
		}
		if sq, ok := cfg["source_quotas"].(map[string]interface{}); ok {
			s.SourceQuotas = make(map[string]int, len(sq))
			for k := range sq {
				s.SourceQuotas[k] = int(conv.ConfigGetInt64(sq, k, 0))
			}
		}
		return s, nil
	case "ratio":
		s := &recall.RatioMergeStrategy{
			TotalLimit: int(conv.ConfigGetInt64(cfg, "total_limit", 0)),
		}
		if sr, ok := cfg["source_ratios"].(map[string]interface{}); ok {
			s.SourceRatios = conv.MapToFloat64(sr)
		}
		return s, nil
	case "hybrid_ratio":
		s := &recall.HybridRatioMergeStrategy{
			TotalLimit:                int(conv.ConfigGetInt64(cfg, "total_limit", 0)),
			DropUnconfiguredSources:   conv.ConfigGet(cfg, "drop_unconfigured_sources", false),
			SortByPriorityBeforeDedup: conv.ConfigGet(cfg, "sort_by_priority_before_dedup", false),
		}
		if sr, ok := cfg["source_ratios"].(map[string]interface{}); ok {
			s.SourceRatios = conv.MapToFloat64(sr)
		}
		return s, nil
	case "round_robin":
		s := &recall.RoundRobinMergeStrategy{
			TopN: int(conv.ConfigGetInt64(cfg, "top_n", 0)),
		}
		if order, ok := cfg["source_order"].([]interface{}); ok {
			s.SourceOrder = conv.SliceAnyToString(order)
		}
		return s, nil
	case "waterfall":
		s := &recall.WaterfallMergeStrategy{
			TotalLimit: int(conv.ConfigGetInt64(cfg, "total_limit", 0)),
		}
		if sp, ok := cfg["source_priority"].([]interface{}); ok {
			s.SourcePriority = conv.SliceAnyToString(sp)
		}
		if sl, ok := cfg["source_limits"].(map[string]interface{}); ok {
			s.SourceLimits = make(map[string]int, len(sl))
			for k := range sl {
				s.SourceLimits[k] = int(conv.ConfigGetInt64(sl, k, 0))
			}
		}
		return s, nil
	case "chain":
		steps, ok := cfg["strategies"].([]interface{})
		if !ok || len(steps) == 0 {
			return nil, fmt.Errorf("chain merge_strategy requires non-empty 'strategies' list")
		}
		chain := &recall.ChainMergeStrategy{
			Strategies: make([]recall.MergeStrategy, 0, len(steps)),
		}
		for i, step := range steps {
			stepCfg, ok := step.(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf("chain strategies[%d]: invalid config", i)
			}
			ms, err := buildMergeStrategy(stepCfg)
			if err != nil {
				return nil, fmt.Errorf("chain strategies[%d]: %w", i, err)
			}
			chain.Strategies = append(chain.Strategies, ms)
		}
		return chain, nil
	default:
		return &recall.FirstMergeStrategy{}, nil
	}
}

func BuildHotNode(cfg map[string]interface{}) (pipeline.Node, error) {
	ids := conv.SliceAnyToString(cfg["ids"])
	if ids == nil {
		ids = []string{}
	}
	return &recall.Hot{IDs: ids}, nil
}

func BuildANNNode(cfg map[string]interface{}) (pipeline.Node, error) {
	return nil, fmt.Errorf("ann node not fully implemented: requires VectorService injection")
}

func BuildLRNode(cfg map[string]interface{}) (pipeline.Node, error) {
	weightsMap, ok := cfg["weights"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("weights not found")
	}
	weights := conv.MapToFloat64(weightsMap)
	bias := conv.ConfigGet(cfg, "bias", 0.0)
	lr := &model.LRModel{Bias: bias, Weights: weights}
	node := &rank.LRNode{Model: lr}
	if explain, ok := cfg["explain"].(map[string]interface{}); ok {
		node.Explain = &rank.LRExplainConfig{
			EmitRawScore:        conv.ConfigGet(explain, "emit_raw_score", false),
			EmitMissingFlag:     conv.ConfigGet(explain, "emit_missing_flag", false),
			EmitFeatureCoverage: conv.ConfigGet(explain, "emit_feature_coverage", false),
		}
	}
	return node, nil
}

// buildKServeRPCModel 从配置中构建基于 KServe V2 协议的 RPCModel。
// 默认走 KServe V2（Open Inference Protocol），配置 protocol: "legacy" 可回退到旧直连协议。
func buildKServeRPCModel(cfg map[string]interface{}, defaultModelType string) (*model.RPCModel, error) {
	endpoint := conv.ConfigGet(cfg, "endpoint", "")
	if endpoint == "" {
		return nil, fmt.Errorf("endpoint not found")
	}
	timeout := 5 * time.Second
	if sec := conv.ConfigGetInt64(cfg, "timeout", 5); sec > 0 {
		timeout = time.Duration(sec) * time.Second
	}
	modelType := conv.ConfigGet(cfg, "model_type", defaultModelType)
	if modelType == "" {
		modelType = defaultModelType
	}

	// protocol: "legacy" 回退到旧的直连 HTTP 协议（{"data": [...]}, {"predictions": [...]}）
	protocol := conv.ConfigGet(cfg, "protocol", "")
	if protocol == "legacy" {
		return model.NewRPCModel(modelType, endpoint, timeout), nil
	}

	// 默认走 KServe V2（Open Inference Protocol）
	modelName := conv.ConfigGet(cfg, "model_name", modelType)
	opts := []service.KServeOption{
		service.WithKServeTimeout(timeout),
	}
	if v := conv.ConfigGet(cfg, "model_version", ""); v != "" {
		opts = append(opts, service.WithKServeVersion(v))
	}
	if v := conv.ConfigGet(cfg, "v2_input_name", ""); v != "" {
		opts = append(opts, service.WithKServeV2InputName(v))
	}
	if v := conv.ConfigGet(cfg, "v2_output_name", ""); v != "" {
		opts = append(opts, service.WithKServeV2OutputName(v))
	}
	kserveClient := service.NewKServeClient(endpoint, modelName, opts...)
	return model.NewRPCModelFromService(modelType, kserveClient), nil
}

func BuildRPCNode(cfg map[string]interface{}) (pipeline.Node, error) {
	rpcModel, err := buildKServeRPCModel(cfg, "rpc")
	if err != nil {
		return nil, err
	}
	return &rank.RPCNode{Model: rpcModel}, nil
}

func BuildWideDeepNode(cfg map[string]interface{}) (pipeline.Node, error) {
	rpcModel, err := buildKServeRPCModel(cfg, "wide_deep")
	if err != nil {
		return nil, err
	}
	return &rank.WideDeepNode{Model: rpcModel}, nil
}

func BuildTwoTowerNode(cfg map[string]interface{}) (pipeline.Node, error) {
	rpcModel, err := buildKServeRPCModel(cfg, "two_tower")
	if err != nil {
		return nil, err
	}
	return &rank.TwoTowerNode{Model: rpcModel}, nil
}

func BuildDNNNode(cfg map[string]interface{}) (pipeline.Node, error) {
	rpcModel, err := buildKServeRPCModel(cfg, "dnn")
	if err != nil {
		return nil, err
	}
	return &rank.DNNNode{Model: rpcModel}, nil
}

func BuildDINNode(cfg map[string]interface{}) (pipeline.Node, error) {
	rpcModel, err := buildKServeRPCModel(cfg, "din")
	if err != nil {
		return nil, err
	}
	maxSeq := int(conv.ConfigGetInt64(cfg, "max_behavior_seq_len", 10))
	if maxSeq <= 0 {
		maxSeq = 10
	}
	return &rank.DINNode{Model: rpcModel, MaxBehaviorSeqLen: maxSeq}, nil
}

func BuildDiversityNode(cfg map[string]interface{}) (pipeline.Node, error) {
	node := &rerank.Diversity{
		LabelKey:        conv.ConfigGet(cfg, "label_key", ""),
		DiversityKeys:   conv.SliceAnyToString(cfg["diversity_keys"]),
		MaxConsecutive:  int(conv.ConfigGetInt64(cfg, "max_consecutive", 0)),
		WindowSize:      int(conv.ConfigGetInt64(cfg, "window_size", 0)),
		ChannelLabelKey: conv.ConfigGet(cfg, "channel_label_key", ""),
		Limit:           int(conv.ConfigGetInt64(cfg, "limit", 0)),
		ExploreLimit:    int(conv.ConfigGetInt64(cfg, "explore_limit", 0)),
	}
	if ch, ok := cfg["exclude_channels"].([]interface{}); ok {
		node.ExcludeChannels = conv.SliceAnyToString(ch)
	}
	if rawConstraints, ok := cfg["constraints"].([]interface{}); ok {
		for _, rc := range rawConstraints {
			cm, ok := rc.(map[string]interface{})
			if !ok {
				continue
			}
			c := rerank.DiversityConstraint{
				Dimensions:          conv.SliceAnyToString(cm["dimensions"]),
				MaxConsecutive:      int(conv.ConfigGetInt64(cm, "max_consecutive", 0)),
				WindowSize:          int(conv.ConfigGetInt64(cm, "window_size", 0)),
				MaxPerWindow:        int(conv.ConfigGetInt64(cm, "max_per_window", 0)),
				Weight:              conv.ConfigGet(cm, "weight", 0.0),
				MultiValueDelimiter: conv.ConfigGet(cm, "multi_value_delimiter", ""),
			}
			node.Constraints = append(node.Constraints, c)
		}
	}
	return node, nil
}

func BuildMMoENode(cfg map[string]interface{}) (pipeline.Node, error) {
	endpoint := conv.ConfigGet(cfg, "endpoint", "")
	if endpoint == "" {
		return nil, fmt.Errorf("endpoint not found")
	}
	timeout := 5 * time.Second
	if sec := conv.ConfigGetInt64(cfg, "timeout", 5); sec > 0 {
		timeout = time.Duration(sec) * time.Second
	}
	node := &rerank.MMoENode{
		Endpoint:           endpoint,
		Timeout:            timeout,
		WeightCTR:          conv.ConfigGet(cfg, "weight_ctr", 1.0),
		WeightWatchTime:    conv.ConfigGet(cfg, "weight_watch_time", 0.01),
		WeightGMV:          conv.ConfigGet(cfg, "weight_gmv", 1e-6),
		StripFeaturePrefix: conv.ConfigGet(cfg, "strip_feature_prefix", false),
	}
	return node, nil
}

func BuildTopNNode(cfg map[string]interface{}) (pipeline.Node, error) {
	n := int(conv.ConfigGetInt64(cfg, "top_n", 0))
	if n <= 0 {
		n = int(conv.ConfigGetInt64(cfg, "n", 0))
	}
	return &rerank.TopNNode{N: n}, nil
}

func BuildFilterNode(cfg map[string]interface{}) (pipeline.Node, error) {
	return buildFilterNodeWithDeps(cfg, legacyDependencies())
}

// BuildFilterNodeWithDependencies 使用指定依赖构建 filter 节点（实例级依赖绑定）。
func BuildFilterNodeWithDependencies(cfg map[string]interface{}, deps Dependencies) (pipeline.Node, error) {
	return buildFilterNodeWithDeps(cfg, deps)
}

func parseFilterFromMap(filterMap map[string]interface{}, deps Dependencies) (filter.Filter, error) {
	filterType := conv.ConfigGet(filterMap, "type", "")
	storeAdapter := deps.filterStoreAdapter()
	switch filterType {
	case "blacklist":
		ids := conv.SliceAnyToString(filterMap["item_ids"])
		if ids == nil {
			ids = []string{}
		}
		key := conv.ConfigGet(filterMap, "key", "")
		return filter.NewBlacklistFilter(ids, storeAdapter, key), nil
	case "user_block":
		keyPrefix := conv.ConfigGet(filterMap, "key_prefix", "")
		return filter.NewUserBlockFilter(storeAdapter, keyPrefix), nil
	case "exposed":
		keyPrefix := conv.ConfigGet(filterMap, "key_prefix", "")
		timeWindow := conv.ConfigGetInt64(filterMap, "time_window", 0)
		bloomFilterDayWindow := conv.ConfigGet(filterMap, "bloom_filter_day_window", 0)
		return filter.NewExposedFilter(storeAdapter, keyPrefix, timeWindow, bloomFilterDayWindow), nil
	case "exposed_batch":
		keyPrefix := conv.ConfigGet(filterMap, "key_prefix", "")
		timeWindow := conv.ConfigGetInt64(filterMap, "time_window", 0)
		bloomFilterDayWindow := conv.ConfigGet(filterMap, "bloom_filter_day_window", 0)
		return filter.NewBatchExposedFilter(storeAdapter, deps.BatchExposureChecker, keyPrefix, timeWindow, bloomFilterDayWindow), nil
	case "expr":
		return &filter.ExprFilter{
			Expr:   conv.ConfigGet(filterMap, "expr", ""),
			Invert: conv.ConfigGet(filterMap, "invert", false),
		}, nil
	case "quality_gate":
		return &filter.QualityGateFilter{
			MinScore: conv.ConfigGet(filterMap, "min_score", 0.0),
		}, nil
	case "dedup_field":
		return &filter.DedupByFieldFilter{
			FieldKey: conv.ConfigGet(filterMap, "field_key", ""),
		}, nil
	case "time_decay":
		maxAgeSec := conv.ConfigGetInt64(filterMap, "max_age_seconds", 0)
		return &filter.TimeDecayFilter{
			TimeField: conv.ConfigGet(filterMap, "time_field", ""),
			MaxAge:    time.Duration(maxAgeSec) * time.Second,
		}, nil
	case "frequency_cap":
		if deps.FrequencyCapStore == nil {
			return nil, fmt.Errorf("filter.frequency_cap requires Dependencies.FrequencyCapStore")
		}
		windowSec := conv.ConfigGetInt64(filterMap, "window_seconds", 0)
		return &filter.FrequencyCapFilter{
			Store:    deps.FrequencyCapStore,
			MaxCount: int(conv.ConfigGetInt64(filterMap, "max_count", 3)),
			Window:   time.Duration(windowSec) * time.Second,
		}, nil
	default:
		return nil, fmt.Errorf("unknown filter type: %s", filterType)
	}
}

func buildFilterNodeWithDeps(cfg map[string]interface{}, deps Dependencies) (pipeline.Node, error) {
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
		f, err := parseFilterFromMap(filterMap, deps)
		if err != nil {
			return nil, err
		}
		filters = append(filters, f)
	}
	return &filter.FilterNode{Filters: filters}, nil
}

func BuildFeatureEnrichNode(cfg map[string]interface{}) (pipeline.Node, error) {
	return buildFeatureEnrichNodeWithDeps(cfg, legacyDependencies())
}

// BuildFeatureEnrichNodeWithDependencies 使用指定依赖构建 feature.enrich 节点（实例级依赖绑定）。
func BuildFeatureEnrichNodeWithDependencies(cfg map[string]interface{}, deps Dependencies) (pipeline.Node, error) {
	return buildFeatureEnrichNodeWithDeps(cfg, deps)
}

func buildFeatureEnrichNodeWithDeps(cfg map[string]interface{}, deps Dependencies) (pipeline.Node, error) {
	return &feature.EnrichNode{
		FeatureService:     deps.FeatureService,
		UserFeaturePrefix:  conv.ConfigGet(cfg, "user_feature_prefix", ""),
		ItemFeaturePrefix:  conv.ConfigGet(cfg, "item_feature_prefix", ""),
		CrossFeaturePrefix: conv.ConfigGet(cfg, "cross_feature_prefix", ""),
	}, nil
}

func buildTrafficPlanNodeWithDeps(cfg map[string]interface{}, deps Dependencies) (pipeline.Node, error) {
	kind := conv.ConfigGet(cfg, "planner", "noop")
	var planner rerank.TrafficPlanner
	switch kind {
	case "noop", "":
		planner = rerank.NoOpTrafficPlanner{}
	case "static":
		planner = &rerank.StaticTrafficPlanner{
			ControlID: conv.ConfigGet(cfg, "control_id", ""),
			Slot:      conv.ConfigGet(cfg, "slot", ""),
		}
	case "inject":
		if deps.TrafficPlanner == nil {
			return nil, fmt.Errorf(`rerank.traffic_plan: planner "inject" requires Dependencies.TrafficPlanner`)
		}
		planner = deps.TrafficPlanner
	default:
		return nil, fmt.Errorf("rerank.traffic_plan: unknown planner %q", kind)
	}
	return &rerank.TrafficPlanNode{
		Planner:     planner,
		LabelSource: conv.ConfigGet(cfg, "label_source", ""),
	}, nil
}

func buildScoreAdjustNodeWithDeps(cfg map[string]interface{}, deps Dependencies) (pipeline.Node, error) {
	rawRules, ok := cfg["rules"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("rerank.score_adjust: rules not found or invalid")
	}
	rules := make([]rerank.ScoreAdjustRule, 0, len(rawRules))
	for i, rr := range rawRules {
		rm, ok := rr.(map[string]interface{})
		if !ok {
			continue
		}
		rule := rerank.ScoreAdjustRule{
			Expr:  conv.ConfigGet(rm, "expr", ""),
			Mode:  rerank.ScoreAdjustMode(conv.ConfigGet(rm, "mode", "add")),
			Value: conv.ConfigGet(rm, "value", 0.0),
		}
		if fm, ok := rm["filter"].(map[string]interface{}); ok {
			f, err := parseFilterFromMap(fm, deps)
			if err != nil {
				return nil, fmt.Errorf("rerank.score_adjust rules[%d]: %w", i, err)
			}
			rule.Filter = f
		}
		if rule.Filter == nil && rule.Expr == "" {
			return nil, fmt.Errorf("rerank.score_adjust: rules[%d] needs expr and/or filter", i)
		}
		rules = append(rules, rule)
	}
	if len(rules) == 0 {
		return nil, fmt.Errorf("rerank.score_adjust: no valid rules")
	}
	return &rerank.ScoreAdjust{
		Rules:         rules,
		MatchAllRules: conv.ConfigGet(cfg, "match_all_rules", false),
	}, nil
}

func buildScoreWeightBoostNodeWithDeps(cfg map[string]interface{}, deps Dependencies) (pipeline.Node, error) {
	if deps.ScoreWeightProvider == nil {
		return nil, fmt.Errorf("rerank.score_weight requires Dependencies.ScoreWeightProvider")
	}
	mode := conv.ConfigGet(cfg, "mode", "mul")
	if mode == "" {
		mode = "mul"
	}
	return &rerank.ScoreWeightBoost{
		Provider: deps.ScoreWeightProvider,
		Mode:     rerank.ScoreWeightApplyMode(mode),
	}, nil
}

func buildRecallChannelMixNodeWithDeps(cfg map[string]interface{}, deps Dependencies) (pipeline.Node, error) {
	rawRules, ok := cfg["rules"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("rerank.recall_channel_mix: rules not found or invalid")
	}
	rules := make([]rerank.ChannelRule, 0, len(rawRules))
	for i, rr := range rawRules {
		rm, ok := rr.(map[string]interface{})
		if !ok {
			continue
		}
		rule := rerank.ChannelRule{
			Kind:            rerank.ChannelSlotKind(conv.ConfigGet(rm, "kind", "")),
			RandomSlotStart: int(conv.ConfigGetInt64(rm, "random_slot_start", 0)),
			RandomSlotEnd:   int(conv.ConfigGetInt64(rm, "random_slot_end", 0)),
			RandomCount:     int(conv.ConfigGetInt64(rm, "random_count", 0)),
			Expr:            conv.ConfigGet(rm, "expr", ""),
		}
		if ch, ok := rm["channels"].([]interface{}); ok {
			rule.Channels = conv.SliceAnyToString(ch)
		}
		if fs, ok := rm["fixed_slots"].([]interface{}); ok {
			rule.FixedSlots = sliceAnyToIntSlice(fs)
		}
		if fm, ok := rm["filter"].(map[string]interface{}); ok {
			f, err := parseFilterFromMap(fm, deps)
			if err != nil {
				return nil, fmt.Errorf("rerank.recall_channel_mix rules[%d]: %w", i, err)
			}
			rule.Filter = f
		}
		rules = append(rules, rule)
	}
	if len(rules) == 0 {
		return nil, fmt.Errorf("rerank.recall_channel_mix: no valid rules")
	}
	return &rerank.RecallChannelMix{
		LabelKey:        conv.ConfigGet(cfg, "label_key", ""),
		OutputSize:      int(conv.ConfigGetInt64(cfg, "output_size", 0)),
		RemainderPolicy: rerank.RemainderPolicy(conv.ConfigGet(cfg, "remainder_policy", "")),
		Rules:           rules,
	}, nil
}

func sliceAnyToIntSlice(raw []interface{}) []int {
	out := make([]int, 0, len(raw))
	for _, x := range raw {
		if n, ok := conv.ToInt(x); ok {
			out = append(out, n)
		}
	}
	return out
}

func buildSampleNode(cfg map[string]interface{}) (pipeline.Node, error) {
	n := int(conv.ConfigGetInt64(cfg, "n", 0))
	shuffle := conv.ConfigGet(cfg, "shuffle", false)
	return &rerank.SampleNode{N: n, Shuffle: shuffle}, nil
}

func buildFairInterleaveNode(cfg map[string]interface{}) (pipeline.Node, error) {
	return &rerank.FairInterleaveNode{
		N:        int(conv.ConfigGetInt64(cfg, "n", 0)),
		LabelKey: conv.ConfigGet(cfg, "label_key", ""),
	}, nil
}

func buildWeightedInterleaveNode(cfg map[string]interface{}) (pipeline.Node, error) {
	node := &rerank.WeightedInterleaveNode{
		N:        int(conv.ConfigGetInt64(cfg, "n", 0)),
		LabelKey: conv.ConfigGet(cfg, "label_key", ""),
	}
	if wm, ok := cfg["weights"].(map[string]interface{}); ok {
		node.Weights = conv.MapToFloat64(wm)
	}
	return node, nil
}

func buildGroupQuotaNode(cfg map[string]interface{}) (pipeline.Node, error) {
	node := &rerank.GroupQuotaNode{
		N:        int(conv.ConfigGetInt64(cfg, "n", 0)),
		FieldKey: conv.ConfigGet(cfg, "field_key", ""),
		Strategy: rerank.GroupQuotaStrategy(conv.ConfigGet(cfg, "strategy", "")),
		GroupMin: int(conv.ConfigGetInt64(cfg, "group_min", 0)),
		GroupMax: int(conv.ConfigGetInt64(cfg, "group_max", 0)),
	}
	if caps, ok := cfg["group_caps"].(map[string]interface{}); ok {
		node.GroupCaps = make(map[string]int, len(caps))
		for k := range caps {
			node.GroupCaps[k] = int(conv.ConfigGetInt64(caps, k, 0))
		}
	}
	if rawGroups, ok := cfg["expr_groups"].([]interface{}); ok {
		for _, rg := range rawGroups {
			gm, ok := rg.(map[string]interface{})
			if !ok {
				continue
			}
			node.ExprGroups = append(node.ExprGroups, rerank.ExprGroup{
				Name:  conv.ConfigGet(gm, "name", ""),
				Expr:  conv.ConfigGet(gm, "expr", ""),
				Quota: int(conv.ConfigGetInt64(gm, "quota", 0)),
			})
		}
	}
	return node, nil
}

func buildDPPDiversityNode(cfg map[string]interface{}) (pipeline.Node, error) {
	return &rerank.DPPDiversityNode{
		N:            int(conv.ConfigGetInt64(cfg, "n", 0)),
		Alpha:        conv.ConfigGet(cfg, "alpha", 1.0),
		WindowSize:   int(conv.ConfigGetInt64(cfg, "window_size", 0)),
		EmbeddingKey: conv.ConfigGet(cfg, "embedding_key", ""),
		NormalizeEmb: conv.ConfigGet(cfg, "normalize_emb", true),
		ScoreNorm:    rerank.ScoreNormMode(conv.ConfigGetInt64(cfg, "score_norm", 0)),
	}, nil
}

func buildSSDDiversityNode(cfg map[string]interface{}) (pipeline.Node, error) {
	return &rerank.SSDDiversityNode{
		N:            int(conv.ConfigGetInt64(cfg, "n", 0)),
		Gamma:        conv.ConfigGet(cfg, "gamma", 0.25),
		WindowSize:   int(conv.ConfigGetInt64(cfg, "window_size", 5)),
		EmbeddingKey: conv.ConfigGet(cfg, "embedding_key", ""),
		NormalizeEmb: conv.ConfigGet(cfg, "normalize_emb", true),
		ScoreNorm:    rerank.ScoreNormMode(conv.ConfigGetInt64(cfg, "score_norm", 0)),
	}, nil
}

func buildConditionalNodeWithDeps(cfg map[string]interface{}, deps Dependencies, factory *pipeline.NodeFactory) (pipeline.Node, error) {
	nodeCfg, ok := cfg["node"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("filter.conditional: node config required")
	}
	nodeType := conv.ConfigGet(nodeCfg, "type", "")
	if nodeType == "" {
		return nil, fmt.Errorf("filter.conditional: node.type required")
	}
	innerCfg, _ := nodeCfg["config"].(map[string]interface{})
	if innerCfg == nil {
		innerCfg = make(map[string]interface{})
	}
	inner, err := factory.Build(nodeType, innerCfg)
	if err != nil {
		return nil, fmt.Errorf("filter.conditional: build inner node %q: %w", nodeType, err)
	}
	return &filter.ConditionalNode{
		Node: inner,
	}, nil
}

// ---------------------------------------------------------------------------
// Explore / Exploit builders
// ---------------------------------------------------------------------------

func buildEpsilonGreedyNode(cfg map[string]interface{}) (pipeline.Node, error) {
	return &rerank.EpsilonGreedyNode{
		Epsilon:     conv.ConfigGet(cfg, "epsilon", 0.1),
		ExploitSize: int(conv.ConfigGetInt64(cfg, "exploit_size", 0)),
	}, nil
}

func buildUCBNodeWithDeps(cfg map[string]interface{}, deps Dependencies) (pipeline.Node, error) {
	if deps.BanditStatsProvider == nil {
		return nil, fmt.Errorf("rerank.ucb requires Dependencies.BanditStatsProvider")
	}
	return &rerank.UCBNode{
		Provider: deps.BanditStatsProvider,
		C:        conv.ConfigGet(cfg, "c", 1.0),
		N:        int(conv.ConfigGetInt64(cfg, "n", 0)),
	}, nil
}

func buildThompsonSamplingNodeWithDeps(cfg map[string]interface{}, deps Dependencies) (pipeline.Node, error) {
	if deps.BanditStatsProvider == nil {
		return nil, fmt.Errorf("rerank.thompson_sampling requires Dependencies.BanditStatsProvider")
	}
	return &rerank.ThompsonSamplingNode{
		Provider:    deps.BanditStatsProvider,
		N:           int(conv.ConfigGetInt64(cfg, "n", 0)),
		PureExplore: conv.ConfigGet(cfg, "pure_explore", false),
	}, nil
}

func buildColdStartBoostNodeWithDeps(cfg map[string]interface{}, deps Dependencies) (pipeline.Node, error) {
	return &rerank.ColdStartBoostNode{
		Provider:      deps.BanditStatsProvider,
		ImpressionKey: conv.ConfigGet(cfg, "impression_key", ""),
		Threshold:     conv.ConfigGetInt64(cfg, "threshold", 100),
		BoostValue:    conv.ConfigGet(cfg, "boost_value", 1.0),
	}, nil
}

// ---------------------------------------------------------------------------
// PostProcess builders
// ---------------------------------------------------------------------------

func buildPaddingNodeWithDeps(cfg map[string]interface{}, deps Dependencies) (pipeline.Node, error) {
	return &postprocess.PaddingNode{
		N:            int(conv.ConfigGetInt64(cfg, "n", 0)),
		FallbackFunc: deps.PaddingFunc,
	}, nil
}

func buildTruncateFieldsNode(cfg map[string]interface{}) (pipeline.Node, error) {
	return &postprocess.TruncateFieldsNode{
		ClearFeatures: conv.ConfigGet(cfg, "clear_features", false),
		ClearMeta:     conv.ConfigGet(cfg, "clear_meta", false),
		ClearLabels:   conv.ConfigGet(cfg, "clear_labels", false),
		KeepMetaKeys:  conv.SliceAnyToString(cfg["keep_meta_keys"]),
	}, nil
}

// ---------------------------------------------------------------------------
// Recall source builders
// ---------------------------------------------------------------------------

func buildRPCRecallNode(cfg map[string]interface{}) (pipeline.Node, error) {
	endpoint := conv.ConfigGet(cfg, "endpoint", "")
	if endpoint == "" {
		return nil, fmt.Errorf("recall.rpc: endpoint required")
	}
	timeout := 5 * time.Second
	if sec := conv.ConfigGetInt64(cfg, "timeout", 5); sec > 0 {
		timeout = time.Duration(sec) * time.Second
	}
	r := recall.NewRPCRecall(endpoint, timeout)
	if topK := int(conv.ConfigGetInt64(cfg, "top_k", 0)); topK > 0 {
		r.TopK = topK
	}
	return wrapSourceAsNode(r), nil
}

func buildGraphRecallNode(cfg map[string]interface{}) (pipeline.Node, error) {
	endpoint := conv.ConfigGet(cfg, "endpoint", "")
	if endpoint == "" {
		return nil, fmt.Errorf("recall.graph: endpoint required")
	}
	timeout := 5 * time.Second
	if sec := conv.ConfigGetInt64(cfg, "timeout", 5); sec > 0 {
		timeout = time.Duration(sec) * time.Second
	}
	return wrapSourceAsNode(&recall.GraphRecall{
		Endpoint: endpoint,
		Timeout:  timeout,
		TopK:     int(conv.ConfigGetInt64(cfg, "top_k", 20)),
	}), nil
}

// sourceNodeAdapter 将 recall.Source 适配为 pipeline.Node。
type sourceNodeAdapter struct {
	source recall.Source
}

func wrapSourceAsNode(s recall.Source) pipeline.Node {
	return &sourceNodeAdapter{source: s}
}

func (a *sourceNodeAdapter) Name() string        { return a.source.Name() }
func (a *sourceNodeAdapter) Kind() pipeline.Kind { return pipeline.KindRecall }
func (a *sourceNodeAdapter) Process(ctx context.Context, rctx *core.RecommendContext, _ []*core.Item) ([]*core.Item, error) {
	return a.source.Recall(ctx, rctx)
}
