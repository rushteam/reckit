package feature

import (
	"context"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/conv"
)

// FeatureExtractor 是特征抽取器的统一接口，采用策略模式。
//
// 作为推荐脚手架，不同模型可能需要不同的特征抽取逻辑：
//   - 双塔模型：需要用户特征（age, gender, interests）
//   - YouTube DNN：需要用户特征 + 历史行为（user_age, user_gender, history_item_ids）
//   - DSSM：需要 Query 特征（query_features）
//   - 其他模型：可能有自定义需求
//
// 通过实现此接口，用户可以完全自定义特征抽取逻辑，无需修改库代码。
//
// 使用示例：
//
//	// 自定义抽取器
//	customExtractor := feature.NewCustomExtractor(func(ctx context.Context, rctx *core.RecommendContext) (map[string]float64, error) {
//	    // 自定义逻辑：从多个源组合特征
//	    features := make(map[string]float64)
//	    // ... 自定义抽取逻辑
//	    return features, nil
//	})
//
//	// 在召回源中使用
//	twoTowerRecall := recall.NewTwoTowerRecall(
//	    featureService,
//	    userTowerService,
//	    vectorService,
//	    recall.WithTwoTowerUserFeatureExtractor(customExtractor.Extract),
//	)
type FeatureExtractor interface {
	// Extract 从 RecommendContext 中提取特征
	// 返回特征字典，key 为特征名，value 为特征值
	Extract(ctx context.Context, rctx *core.RecommendContext) (map[string]float64, error)

	// Name 返回抽取器名称（用于日志/监控）
	Name() string
}

// DefaultFeatureExtractor 是默认的特征抽取器实现。
//
// 抽取策略（优先级顺序）：
//   1. 从 UserProfile（强类型）提取：age, gender, interests
//   2. 从 UserProfileMap 提取：所有可转换为 float64 的值
//   3. 从 Realtime 提取：所有可转换为 float64 的值（添加 "realtime_" 前缀）
//
// 字段命名：
//   - UserProfile 字段：age, gender, interest_<tag>
//   - UserProfileMap 字段：保持原 key
//   - Realtime 字段：realtime_<key>
type DefaultFeatureExtractor struct {
	// FeatureService 特征服务（可选，如果设置则优先使用）
	FeatureService FeatureService

	// FieldPrefix 字段前缀（可选，如 "user_"）
	FieldPrefix string

	// IncludeRealtime 是否包含实时特征
	IncludeRealtime bool
}

// NewDefaultFeatureExtractor 创建默认特征抽取器
func NewDefaultFeatureExtractor(opts ...DefaultFeatureExtractorOption) *DefaultFeatureExtractor {
	extractor := &DefaultFeatureExtractor{
		IncludeRealtime: true,
	}
	for _, opt := range opts {
		opt(extractor)
	}
	return extractor
}

// DefaultFeatureExtractorOption 默认抽取器配置选项
type DefaultFeatureExtractorOption func(*DefaultFeatureExtractor)

// WithFeatureService 设置特征服务（优先使用）
func WithFeatureService(service FeatureService) DefaultFeatureExtractorOption {
	return func(e *DefaultFeatureExtractor) {
		e.FeatureService = service
	}
}

// WithFieldPrefix 设置字段前缀（如 "user_"）
func WithFieldPrefix(prefix string) DefaultFeatureExtractorOption {
	return func(e *DefaultFeatureExtractor) {
		e.FieldPrefix = prefix
	}
}

// WithIncludeRealtime 设置是否包含实时特征
func WithIncludeRealtime(include bool) DefaultFeatureExtractorOption {
	return func(e *DefaultFeatureExtractor) {
		e.IncludeRealtime = include
	}
}

func (e *DefaultFeatureExtractor) Name() string {
	return "default"
}

func (e *DefaultFeatureExtractor) Extract(ctx context.Context, rctx *core.RecommendContext) (map[string]float64, error) {
	// 优先使用 FeatureService
	if e.FeatureService != nil && rctx != nil && rctx.UserID != "" {
		features, err := e.FeatureService.GetUserFeatures(ctx, rctx.UserID)
		if err == nil && len(features) > 0 {
			return e.applyPrefix(features), nil
		}
	}

	// 从 Context 提取
	features := e.extractFromContext(rctx)
	return e.applyPrefix(features), nil
}

func (e *DefaultFeatureExtractor) extractFromContext(rctx *core.RecommendContext) map[string]float64 {
	features := make(map[string]float64)
	if rctx == nil {
		return features
	}

	// 从 UserProfile（强类型）提取
	if rctx.User != nil {
		features["age"] = float64(rctx.User.Age)
		if rctx.User.Gender == "male" {
			features["gender"] = 1.0
		} else if rctx.User.Gender == "female" {
			features["gender"] = 2.0
		} else {
			features["gender"] = 0.0
		}
		for tag, score := range rctx.User.Interests {
			features["interest_"+tag] = score
		}
	}

	// 从 UserProfileMap 提取
	if rctx.UserProfile != nil {
		for k, v := range rctx.UserProfile {
			if fv, ok := conv.ToFloat64(v); ok {
				features[k] = fv
			}
		}
	}

	// 从 Realtime 提取
	if e.IncludeRealtime && rctx.Realtime != nil {
		for k, v := range rctx.Realtime {
			if fv, ok := conv.ToFloat64(v); ok {
				features["realtime_"+k] = fv
			}
		}
	}

	return features
}

func (e *DefaultFeatureExtractor) applyPrefix(features map[string]float64) map[string]float64 {
	if e.FieldPrefix == "" {
		return features
	}
	result := make(map[string]float64, len(features))
	for k, v := range features {
		result[e.FieldPrefix+k] = v
	}
	return result
}

// CustomFeatureExtractor 是自定义特征抽取器，允许用户完全自定义抽取逻辑。
type CustomFeatureExtractor struct {
	name    string
	extract func(ctx context.Context, rctx *core.RecommendContext) (map[string]float64, error)
}

// NewCustomFeatureExtractor 创建自定义特征抽取器
func NewCustomFeatureExtractor(name string, extract func(ctx context.Context, rctx *core.RecommendContext) (map[string]float64, error)) *CustomFeatureExtractor {
	return &CustomFeatureExtractor{
		name:    name,
		extract: extract,
	}
}

func (e *CustomFeatureExtractor) Name() string {
	return e.name
}

func (e *CustomFeatureExtractor) Extract(ctx context.Context, rctx *core.RecommendContext) (map[string]float64, error) {
	if e.extract == nil {
		return nil, nil
	}
	return e.extract(ctx, rctx)
}

// CompositeFeatureExtractor 是组合特征抽取器，支持从多个源组合特征。
//
// 使用场景：
//   - 从 FeatureService 获取基础特征
//   - 从 Context 获取实时特征
//   - 从外部服务获取补充特征
//   - 组合并返回
type CompositeFeatureExtractor struct {
	name      string
	extractors []FeatureExtractor
	mergeFunc func(featuresList []map[string]float64) map[string]float64
}

// NewCompositeFeatureExtractor 创建组合特征抽取器
func NewCompositeFeatureExtractor(name string, extractors ...FeatureExtractor) *CompositeFeatureExtractor {
	return &CompositeFeatureExtractor{
		name:       name,
		extractors: extractors,
		mergeFunc:  defaultMergeFeatures,
	}
}

// WithMergeFunc 设置合并函数（默认：后覆盖前）
func (e *CompositeFeatureExtractor) WithMergeFunc(merge func(featuresList []map[string]float64) map[string]float64) *CompositeFeatureExtractor {
	e.mergeFunc = merge
	return e
}

func (e *CompositeFeatureExtractor) Name() string {
	return e.name
}

func (e *CompositeFeatureExtractor) Extract(ctx context.Context, rctx *core.RecommendContext) (map[string]float64, error) {
	featuresList := make([]map[string]float64, 0, len(e.extractors))
	for _, extractor := range e.extractors {
		features, err := extractor.Extract(ctx, rctx)
		if err != nil {
			continue // 忽略单个抽取器错误，继续其他抽取器
		}
		if len(features) > 0 {
			featuresList = append(featuresList, features)
		}
	}
	return e.mergeFunc(featuresList), nil
}

// defaultMergeFeatures 默认合并策略：后覆盖前
func defaultMergeFeatures(featuresList []map[string]float64) map[string]float64 {
	result := make(map[string]float64)
	for _, features := range featuresList {
		for k, v := range features {
			result[k] = v
		}
	}
	return result
}

// QueryFeatureExtractor 是 Query 特征抽取器（用于 DSSM 等场景）。
//
// 从 RecommendContext 中提取 Query 相关特征，支持：
//   - 从 Params["query_features"] 获取
//   - 从 Params["query"] 文本构建特征
//   - 自定义抽取逻辑
type QueryFeatureExtractor struct {
	// QueryFeaturesKey Params 中的 query_features key
	QueryFeaturesKey string

	// QueryTextKey Params 中的 query 文本 key（用于文本特征化）
	QueryTextKey string

	// TextFeatureBuilder 文本特征构建器（可选）
	TextFeatureBuilder func(queryText string) map[string]float64
}

// NewQueryFeatureExtractor 创建 Query 特征抽取器
func NewQueryFeatureExtractor(opts ...QueryFeatureExtractorOption) *QueryFeatureExtractor {
	extractor := &QueryFeatureExtractor{
		QueryFeaturesKey: "query_features",
		QueryTextKey:     "query",
	}
	for _, opt := range opts {
		opt(extractor)
	}
	return extractor
}

// QueryFeatureExtractorOption Query 抽取器配置选项
type QueryFeatureExtractorOption func(*QueryFeatureExtractor)

// WithQueryFeaturesKey 设置 query_features key
func WithQueryFeaturesKey(key string) QueryFeatureExtractorOption {
	return func(e *QueryFeatureExtractor) {
		e.QueryFeaturesKey = key
	}
}

// WithQueryTextKey 设置 query 文本 key
func WithQueryTextKey(key string) QueryFeatureExtractorOption {
	return func(e *QueryFeatureExtractor) {
		e.QueryTextKey = key
	}
}

// WithTextFeatureBuilder 设置文本特征构建器
func WithTextFeatureBuilder(builder func(queryText string) map[string]float64) QueryFeatureExtractorOption {
	return func(e *QueryFeatureExtractor) {
		e.TextFeatureBuilder = builder
	}
}

func (e *QueryFeatureExtractor) Name() string {
	return "query"
}

func (e *QueryFeatureExtractor) Extract(ctx context.Context, rctx *core.RecommendContext) (map[string]float64, error) {
	if rctx == nil || rctx.Params == nil {
		return nil, nil
	}

	// 优先从 query_features 获取
	if e.QueryFeaturesKey != "" {
		if qf, ok := rctx.Params[e.QueryFeaturesKey].(map[string]any); ok {
			return conv.MapToFloat64(qf), nil
		}
	}

	// 从 query 文本构建特征
	if e.QueryTextKey != "" && e.TextFeatureBuilder != nil {
		if queryText, ok := rctx.Params[e.QueryTextKey].(string); ok && queryText != "" {
			return e.TextFeatureBuilder(queryText), nil
		}
	}

	return nil, nil
}

// HistoryExtractor 是历史行为抽取器（用于 YouTube DNN 等场景）。
//
// 从 RecommendContext 中提取用户历史行为序列（如最近点击的物品 ID 列表）。
type HistoryExtractor struct {
	// HistoryKey UserProfile 或 Params 中的历史 key
	HistoryKey string

	// MaxLength 最大历史长度
	MaxLength int

	// CustomExtractor 自定义抽取函数（可选）
	CustomExtractor func(rctx *core.RecommendContext) []string
}

// NewHistoryExtractor 创建历史行为抽取器
func NewHistoryExtractor(opts ...HistoryExtractorOption) *HistoryExtractor {
	extractor := &HistoryExtractor{
		HistoryKey: "recent_clicks",
		MaxLength:  50,
	}
	for _, opt := range opts {
		opt(extractor)
	}
	return extractor
}

// HistoryExtractorOption 历史抽取器配置选项
type HistoryExtractorOption func(*HistoryExtractor)

// WithHistoryKey 设置历史 key
func WithHistoryKey(key string) HistoryExtractorOption {
	return func(e *HistoryExtractor) {
		e.HistoryKey = key
	}
}

// WithMaxLength 设置最大历史长度
func WithMaxLength(maxLen int) HistoryExtractorOption {
	return func(e *HistoryExtractor) {
		e.MaxLength = maxLen
	}
}

// WithCustomHistoryExtractor 设置自定义抽取函数
func WithCustomHistoryExtractor(extract func(rctx *core.RecommendContext) []string) HistoryExtractorOption {
	return func(e *HistoryExtractor) {
		e.CustomExtractor = extract
	}
}

// Extract 提取历史行为序列
func (e *HistoryExtractor) Extract(rctx *core.RecommendContext) []string {
	if e.CustomExtractor != nil {
		return e.CustomExtractor(rctx)
	}

	if rctx == nil {
		return nil
	}

	// 从 User.RecentClicks 获取
	if rctx.User != nil && len(rctx.User.RecentClicks) > 0 {
		hist := rctx.User.RecentClicks
		if e.MaxLength > 0 && len(hist) > e.MaxLength {
			return hist[len(hist)-e.MaxLength:]
		}
		return hist
	}

	// 从 UserProfile 或 Params 获取
	if rctx.UserProfile != nil {
		if hist, ok := rctx.UserProfile[e.HistoryKey].([]string); ok {
			if e.MaxLength > 0 && len(hist) > e.MaxLength {
				return hist[len(hist)-e.MaxLength:]
			}
			return hist
		}
	}

	if rctx.Params != nil {
		if hist, ok := rctx.Params[e.HistoryKey].([]string); ok {
			if e.MaxLength > 0 && len(hist) > e.MaxLength {
				return hist[len(hist)-e.MaxLength:]
			}
			return hist
		}
	}

	return nil
}

// AdaptFeatureExtractor 适配函数类型或接口为 FeatureExtractor 接口。
//
// 支持类型：
//   - feature.FeatureExtractor 接口：直接返回
//   - func(ctx context.Context, rctx *core.RecommendContext) (map[string]float64, error)：包装为 CustomFeatureExtractor
//
// 用于向后兼容：允许传入函数类型，自动转换为接口。
func AdaptFeatureExtractor(extractor interface{}, name string) FeatureExtractor {
	if extractor == nil {
		return nil
	}

	// 如果已经是 FeatureExtractor 接口，直接返回
	if fe, ok := extractor.(FeatureExtractor); ok {
		return fe
	}

	// 如果是函数类型，包装为 CustomFeatureExtractor
	if fn, ok := extractor.(func(ctx context.Context, rctx *core.RecommendContext) (map[string]float64, error)); ok {
		return NewCustomFeatureExtractor(name, fn)
	}

	// 不支持的类型，返回 nil
	return nil
}
