package pipeline

import (
	"context"

	"github.com/rushteam/reckit/core"
)

const (
	ParamPipelineInputCount  = "__pipeline_input_count__"
	ParamPipelineOutputCount = "__pipeline_output_count__"
	ParamNodeInputCountMap   = "__pipeline_node_input_count__"
	ParamNodeOutputCountMap  = "__pipeline_node_output_count__"
)

// StatsHook 在 pipeline 执行阶段记录节点输入/输出条数到 rctx.Params。
// 适合作为通用观测模板，业务可基于这些统计打日志或指标。
type StatsHook struct{}

var _ PipelineHook = (*StatsHook)(nil)

func (h *StatsHook) BeforeNode(_ context.Context, rctx *core.RecommendContext, node Node, items []*core.Item) ([]*core.Item, error) {
	if rctx == nil || node == nil {
		return items, nil
	}
	inMap, _ := ensureStatsMaps(rctx)
	if _, ok := rctx.Params[ParamPipelineInputCount]; !ok {
		rctx.Params[ParamPipelineInputCount] = len(items)
	}
	inMap[node.Name()] = len(items)
	return items, nil
}

func (h *StatsHook) AfterNode(_ context.Context, rctx *core.RecommendContext, node Node, items []*core.Item, err error) ([]*core.Item, error) {
	if rctx == nil || node == nil {
		return items, err
	}
	_, outMap := ensureStatsMaps(rctx)
	outMap[node.Name()] = len(items)
	rctx.Params[ParamPipelineOutputCount] = len(items)
	return items, err
}

// ensureStatsMaps 保证 Params 中的统计 map 存在且类型正确，
// 返回 (inputCountMap, outputCountMap)，调用方无需再做类型断言。
func ensureStatsMaps(rctx *core.RecommendContext) (map[string]int, map[string]int) {
	if rctx.Params == nil {
		rctx.Params = make(map[string]any)
	}
	inMap, ok := rctx.Params[ParamNodeInputCountMap].(map[string]int)
	if !ok {
		inMap = make(map[string]int)
		rctx.Params[ParamNodeInputCountMap] = inMap
	}
	outMap, ok := rctx.Params[ParamNodeOutputCountMap].(map[string]int)
	if !ok {
		outMap = make(map[string]int)
		rctx.Params[ParamNodeOutputCountMap] = outMap
	}
	return inMap, outMap
}
