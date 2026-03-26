package rerank

import (
	"context"
	"errors"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/utils"
)

var (
	errTrafficReorderLen      = errors.New("rerank: traffic_plan reorder requires same length as input")
	errTrafficReorderNil      = errors.New("rerank: traffic_plan reorder output contains nil item")
	errTrafficReorderMismatch = errors.New("rerank: traffic_plan reorder must be a permutation of input items")
)

// 流量调控写入 Item.Labels 的 key，便于与业务埋点/实验侧对齐。
const (
	LabelKeyTrafficControlID = "__traffic_control_id__"
	LabelKeyTrafficSlot      = "__traffic_slot__"
)

// TrafficPlanEntry 描述与单条候选对齐的调控信息（按输入 items 下标对齐）。
type TrafficPlanEntry struct {
	// ControlID 非空时写入 LabelKeyTrafficControlID
	ControlID string
	// Slot 非空时写入 LabelKeyTrafficSlot（字符串位次/槽位 id，由业务约定格式）
	Slot string
}

// TrafficPlanResult 是 TrafficPlanner.Plan 的返回值。
// PerItem 与输入 items 按下标对齐；某项不写标签则对应 ControlID、Slot 均为空字符串。
// Items 非 nil 时表示输出顺序改为 Items（须与输入为同一组 *core.Item 指针的重排）；
// 此时 PerItem 仍与**输入** items 按下标对齐，标签在重排前写入各 items[i]。
type TrafficPlanResult struct {
	PerItem []TrafficPlanEntry
	Items   []*core.Item
}

// TrafficPlanner 流量/投放调控策略：由业务实现（PID、实验、宏控等），节点只负责调用并落标签。
type TrafficPlanner interface {
	Plan(ctx context.Context, rctx *core.RecommendContext, items []*core.Item) (*TrafficPlanResult, error)
}

// TrafficPlanNode 调用 TrafficPlanner，将调控 id/位次写入 Label，可选应用重排后的顺序。
type TrafficPlanNode struct {
	Planner TrafficPlanner
	// LabelSource 写入 utils.Label.Source，默认 "rerank.traffic_plan"
	LabelSource string
}

func (n *TrafficPlanNode) Name() string {
	return "rerank.traffic_plan"
}

func (n *TrafficPlanNode) Kind() pipeline.Kind {
	return pipeline.KindReRank
}

func (n *TrafficPlanNode) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if len(items) == 0 {
		return items, nil
	}
	planner := n.Planner
	if planner == nil {
		planner = NoOpTrafficPlanner{}
	}
	res, err := planner.Plan(ctx, rctx, items)
	if err != nil {
		return nil, err
	}
	if res == nil {
		return items, nil
	}

	src := n.LabelSource
	if src == "" {
		src = "rerank.traffic_plan"
	}

	if len(res.PerItem) > 0 {
		for i, it := range items {
			if it == nil {
				continue
			}
			if i >= len(res.PerItem) {
				break
			}
			e := res.PerItem[i]
			if e.ControlID != "" {
				it.PutLabel(LabelKeyTrafficControlID, utils.Label{Value: e.ControlID, Source: src})
			}
			if e.Slot != "" {
				it.PutLabel(LabelKeyTrafficSlot, utils.Label{Value: e.Slot, Source: src})
			}
		}
	}

	if len(res.Items) == 0 {
		return items, nil
	}
	if err := validateTrafficReorder(items, res.Items); err != nil {
		return nil, err
	}
	return res.Items, nil
}

// validateTrafficReorder 校验 TrafficPlanner 返回的重排结果合法性。
//
// 约定：
//   - out 长度必须等于 input 长度
//   - out 中不允许 nil（planner 重排结果应全部指向有效 item）
//   - out 必须是 input 中非 nil 指针的置换（同一组指针，数量相同）
//   - input 中若含 nil，planner 应在对应位置放入非 nil item（相当于"补位"），
//     无法补位时应返回空 Items，由节点按原序返回。
func validateTrafficReorder(input, out []*core.Item) error {
	if len(input) != len(out) {
		return errTrafficReorderLen
	}
	count := make(map[*core.Item]int, len(input))
	for _, it := range input {
		if it == nil {
			continue
		}
		count[it]++
	}
	for _, it := range out {
		if it == nil {
			return errTrafficReorderNil
		}
		count[it]--
		if count[it] < 0 {
			return errTrafficReorderMismatch
		}
	}
	for _, c := range count {
		if c != 0 {
			return errTrafficReorderMismatch
		}
	}
	return nil
}

// NoOpTrafficPlanner 不写标签、不重排。
type NoOpTrafficPlanner struct{}

func (NoOpTrafficPlanner) Plan(context.Context, *core.RecommendContext, []*core.Item) (*TrafficPlanResult, error) {
	return &TrafficPlanResult{}, nil
}

// StaticTrafficPlanner 测试或简单场景：为每条候选写入相同的调控 id 与位次。
type StaticTrafficPlanner struct {
	ControlID string
	Slot      string
}

func (p *StaticTrafficPlanner) Plan(_ context.Context, _ *core.RecommendContext, items []*core.Item) (*TrafficPlanResult, error) {
	per := make([]TrafficPlanEntry, len(items))
	for i := range items {
		per[i] = TrafficPlanEntry{ControlID: p.ControlID, Slot: p.Slot}
	}
	return &TrafficPlanResult{PerItem: per}, nil
}
