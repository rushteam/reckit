package core

import "context"

// ABDecision 是通用 AB 决策结果结构。
// 具体平台（GrowthBook/自研）可映射到该结构后在链路中透传。
type ABDecision struct {
	FeatureKey   string
	Value        any
	On           bool
	Source       string
	InExperiment bool
}

// ABRuntime 是 AB 运行时扩展接口模板。
// 建议由业务侧实现并通过 RecommendContext.SetExtension 注入。
type ABRuntime interface {
	Extension
	Decide(ctx context.Context, featureKey string) (ABDecision, error)
}

// ABRuntimeFromContext 从 RecommendContext 提取 ABRuntime。
func ABRuntimeFromContext(rctx *RecommendContext, extensionName string) (ABRuntime, bool) {
	if rctx == nil || extensionName == "" {
		return nil, false
	}
	e, ok := rctx.GetExtension(extensionName)
	if !ok || e == nil {
		return nil, false
	}
	rt, ok := e.(ABRuntime)
	return rt, ok
}

// GetABDecision 是 AB 决策快捷函数。
// 未注入运行时时返回零值决策且不报错，便于链路降级。
func GetABDecision(
	ctx context.Context,
	rctx *RecommendContext,
	extensionName string,
	featureKey string,
) (ABDecision, error) {
	rt, ok := ABRuntimeFromContext(rctx, extensionName)
	if !ok || rt == nil {
		return ABDecision{FeatureKey: featureKey}, nil
	}
	return rt.Decide(ctx, featureKey)
}
