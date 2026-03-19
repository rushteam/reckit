package builders

import (
	"fmt"
	"sync"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/feature"
	"github.com/rushteam/reckit/filter"
	"github.com/rushteam/reckit/model"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/conv"
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
	FilterStore        core.Store
	BloomFilterChecker filter.BloomFilterChecker
	FeatureService     core.FeatureService
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
	factory.Register("filter", func(cfg map[string]interface{}) (pipeline.Node, error) {
		return buildFilterNodeWithDeps(cfg, deps)
	})
	factory.Register("feature.enrich", func(cfg map[string]interface{}) (pipeline.Node, error) {
		return buildFeatureEnrichNodeWithDeps(cfg, deps)
	})
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
	return &rank.LRNode{Model: lr}, nil
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
	labelKey := conv.ConfigGet(cfg, "label_key", "category")
	if labelKey == "" {
		labelKey = "category"
	}
	diversityKeys := conv.SliceAnyToString(cfg["diversity_keys"])
	maxConsecutive := int(conv.ConfigGetInt64(cfg, "max_consecutive", 0))
	windowSize := int(conv.ConfigGetInt64(cfg, "window_size", 0))
	return &rerank.Diversity{
		LabelKey:       labelKey,
		DiversityKeys:  diversityKeys,
		MaxConsecutive: maxConsecutive,
		WindowSize:     windowSize,
	}, nil
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

func buildFilterNodeWithDeps(cfg map[string]interface{}, deps Dependencies) (pipeline.Node, error) {
	filtersConfig, ok := cfg["filters"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("filters not found or invalid")
	}
	storeAdapter := deps.filterStoreAdapter()
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
			filters = append(filters, filter.NewBlacklistFilter(ids, storeAdapter, key))
		case "user_block":
			keyPrefix := conv.ConfigGet(filterMap, "key_prefix", "")
			filters = append(filters, filter.NewUserBlockFilter(storeAdapter, keyPrefix))
		case "exposed":
			keyPrefix := conv.ConfigGet(filterMap, "key_prefix", "")
			timeWindow := conv.ConfigGetInt64(filterMap, "time_window", 0)
			bloomFilterDayWindow := conv.ConfigGet(filterMap, "bloom_filter_day_window", 0)
			filters = append(filters, filter.NewExposedFilter(storeAdapter, keyPrefix, timeWindow, bloomFilterDayWindow))
		default:
			return nil, fmt.Errorf("unknown filter type: %s", filterType)
		}
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
