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
	ensureStatsMaps(rctx)
	if _, ok := rctx.Params[ParamPipelineInputCount]; !ok {
		rctx.Params[ParamPipelineInputCount] = len(items)
	}
	in := rctx.Params[ParamNodeInputCountMap].(map[string]int)
	in[node.Name()] = len(items)
	return items, nil
}

func (h *StatsHook) AfterNode(_ context.Context, rctx *core.RecommendContext, node Node, items []*core.Item, err error) ([]*core.Item, error) {
	if rctx == nil || node == nil {
		return items, err
	}
	ensureStatsMaps(rctx)
	out := rctx.Params[ParamNodeOutputCountMap].(map[string]int)
	out[node.Name()] = len(items)
	rctx.Params[ParamPipelineOutputCount] = len(items)
	return items, err
}

func ensureStatsMaps(rctx *core.RecommendContext) {
	if rctx.Params == nil {
		rctx.Params = make(map[string]any)
	}
	if _, ok := rctx.Params[ParamNodeInputCountMap]; !ok {
		rctx.Params[ParamNodeInputCountMap] = make(map[string]int)
	}
	if _, ok := rctx.Params[ParamNodeOutputCountMap]; !ok {
		rctx.Params[ParamNodeOutputCountMap] = make(map[string]int)
	}
}
