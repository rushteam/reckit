package rerank

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"sort"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/trafficctrl"
	"github.com/rushteam/reckit/pkg/utils"
)

// TrafficControlNode 运营流量调控节点：在 Rank 之后执行。
//
// 闭环流程：
//  1. 加载当前场景活跃的调控任务
//  2. 按人群圈选过滤任务，按优先级计算 boost factor（Weight / PID / Pin）
//  3. 对每个 item 匹配物品池规则，调整 score（score × factor）
//  4. 应用多样性约束，防止被调控 item 主导列表
//  5. 执行 Pin 强插
//  6. 异步上报调控事件
type TrafficControlNode struct {
	TaskRepo     trafficctrl.TaskRepository
	Metrics      trafficctrl.MetricsReader  // PID 模式必需
	PIDStore     trafficctrl.PIDStateStore  // PID 模式必需
	EventEmitter trafficctrl.EventEmitter   // 可选

	// UserAttrsFunc 从 RecommendContext 提取用户属性用于 Audience 匹配。
	// nil 时默认使用 rctx.Attributes。
	UserAttrsFunc func(rctx *core.RecommendContext) map[string]any

	// ItemAttrsFunc 从 Item 提取属性用于 ItemPool 匹配。
	// nil 时默认合并 Item.Meta + Item.Labels (Value) + Item.Features。
	ItemAttrsFunc func(item *core.Item) map[string]any

	Scene string // 可选固定场景；空则取 rctx.Scene
}

func (n *TrafficControlNode) Name() string        { return "rerank.traffic_control" }
func (n *TrafficControlNode) Kind() pipeline.Kind { return pipeline.KindReRank }

func (n *TrafficControlNode) resolveScene(rctx *core.RecommendContext) string {
	if n.Scene != "" {
		return n.Scene
	}
	if rctx != nil {
		return rctx.Scene
	}
	return ""
}

type boostedItem struct {
	item      *core.Item
	origScore float64
	factor    float64
	taskID    string
	isBoosted bool
}

func (n *TrafficControlNode) Process(ctx context.Context, rctx *core.RecommendContext, items []*core.Item) ([]*core.Item, error) {
	if n == nil || n.TaskRepo == nil || len(items) == 0 {
		return items, nil
	}

	scene := n.resolveScene(rctx)
	now := time.Now()

	tasks, err := n.TaskRepo.GetActiveTasks(ctx, scene)
	if err != nil {
		slog.WarnContext(ctx, "traffic_control: load tasks failed, skip",
			slog.String("error", err.Error()))
		return items, nil
	}

	activeTasks := filterActiveTasks(tasks, now)
	if len(activeTasks) == 0 {
		return items, nil
	}

	userAttrs := n.userAttrs(rctx)
	sort.Slice(activeTasks, func(i, j int) bool {
		return activeTasks[i].Priority > activeTasks[j].Priority
	})

	taskFactors := make(map[string]float64, len(activeTasks))
	for _, task := range activeTasks {
		if !task.Audience.MatchAll(userAttrs) {
			continue
		}
		factor := n.computeTaskFactor(ctx, task)
		if factor != 1.0 {
			taskFactors[task.ID] = factor
		}
	}
	if len(taskFactors) == 0 {
		return items, nil
	}

	boosted := make([]boostedItem, 0, len(items))
	for _, it := range items {
		if it == nil {
			continue
		}
		itemAttrs := n.itemAttrs(it)
		bi := boostedItem{item: it, origScore: it.Score}

		for _, task := range activeTasks {
			factor, hasFactor := taskFactors[task.ID]
			if !hasFactor {
				continue
			}
			if !task.ItemPool.MatchItem(it.ID, itemAttrs) {
				continue
			}
			bi.factor = factor
			bi.taskID = task.ID
			bi.isBoosted = true
			it.Score *= factor

			it.PutLabel(LabelKeyTrafficControlID, utils.Label{Value: task.ID, Source: "traffic_control"})
			it.PutLabel("__traffic_control_factor__", utils.Label{Value: fmt.Sprintf("%.4f", factor), Source: "traffic_control"})
			break
		}
		boosted = append(boosted, bi)
	}

	// Pin 任务候选
	var pinItems []*pinCandidate
	for _, task := range activeTasks {
		if task.Strategy.BoostType != trafficctrl.BoostTypePin || len(task.Strategy.PinPositions) == 0 {
			continue
		}
		for idx := range boosted {
			if boosted[idx].isBoosted && boosted[idx].taskID == task.ID {
				pinItems = append(pinItems, &pinCandidate{
					item:      boosted[idx].item,
					positions: task.Strategy.PinPositions,
				})
			}
		}
	}

	sort.SliceStable(boosted, func(i, j int) bool {
		return boosted[i].item.Score > boosted[j].item.Score
	})

	result := n.applyDiversityConstraint(activeTasks, boosted)
	result = applyPinPositions(result, pinItems)

	if n.EventEmitter != nil {
		for i, bi := range boosted {
			if bi.isBoosted {
				n.EventEmitter.EmitBoostEvent(ctx, trafficctrl.BoostEvent{
					TaskID:      bi.taskID,
					ItemID:      bi.item.ID,
					UserID:      rctx.UserID,
					Scene:       scene,
					BoostFactor: bi.factor,
					OrigScore:   bi.origScore,
					NewScore:    bi.item.Score,
					Position:    i,
				})
			}
		}
	}

	return result, nil
}

func (n *TrafficControlNode) userAttrs(rctx *core.RecommendContext) map[string]any {
	if n.UserAttrsFunc != nil {
		return n.UserAttrsFunc(rctx)
	}
	if rctx != nil && rctx.Attributes != nil {
		return rctx.Attributes
	}
	return nil
}

func (n *TrafficControlNode) itemAttrs(it *core.Item) map[string]any {
	if n.ItemAttrsFunc != nil {
		return n.ItemAttrsFunc(it)
	}
	attrs := make(map[string]any)
	for k, v := range it.Meta {
		attrs[k] = v
	}
	for k, v := range it.Labels {
		attrs[k] = v.Value
	}
	for k, v := range it.Features {
		attrs[k] = v
	}
	return attrs
}

func (n *TrafficControlNode) computeTaskFactor(ctx context.Context, task *trafficctrl.Task) float64 {
	strategy := &task.Strategy
	switch strategy.BoostType {
	case trafficctrl.BoostTypeWeight:
		factor := strategy.BoostFactor
		if factor <= 0 {
			factor = 1.0
		}
		return math.Min(factor, strategy.EffectiveMaxBoost())
	case trafficctrl.BoostTypePID:
		return n.computePIDFactor(ctx, task)
	case trafficctrl.BoostTypePin:
		return strategy.EffectiveMaxBoost()
	default:
		return 1.0
	}
}

func (n *TrafficControlNode) computePIDFactor(ctx context.Context, task *trafficctrl.Task) float64 {
	if len(task.Targets) == 0 || n.Metrics == nil {
		fb := task.Strategy.FallbackBoost
		if fb <= 0 {
			return 1.0
		}
		return fb
	}

	target := &task.Targets[0]
	currentVal, err := n.Metrics.GetMetric(ctx, task.ID, target.Metric, target.Window)
	if err != nil {
		slog.WarnContext(ctx, "traffic_control: read metric failed, use fallback",
			slog.String("task_id", task.ID),
			slog.String("error", err.Error()))
		if task.Strategy.FallbackBoost > 0 {
			return task.Strategy.FallbackBoost
		}
		return 1.0
	}

	if target.IsSatisfied(currentVal) {
		return 1.0
	}

	pid := trafficctrl.NewPIDController(task.Strategy.PIDParams)
	if n.PIDStore != nil {
		state, loadErr := n.PIDStore.Load(ctx, task.ID)
		if loadErr == nil && state.Iterations > 0 {
			pid.FromState(state)
		}
	}

	pidErr := target.Error(currentVal)
	signal := pid.Update(pidErr)
	factor := trafficctrl.SignalToBoostFactor(signal, task.Strategy.EffectiveMaxBoost())

	if n.PIDStore != nil {
		state := pid.ToState(task.ID, signal)
		if saveErr := n.PIDStore.Save(ctx, state); saveErr != nil {
			slog.WarnContext(ctx, "traffic_control: save PID state failed",
				slog.String("task_id", task.ID),
				slog.String("error", saveErr.Error()))
		}
	}

	return factor
}

// applyDiversityConstraint 限制被调控 item 在输出列表中的密度。
func (n *TrafficControlNode) applyDiversityConstraint(tasks []*trafficctrl.Task, boosted []boostedItem) []*core.Item {
	var divCfg *trafficctrl.DiversityConfig
	for _, task := range tasks {
		if task.Strategy.Diversity.HasDiversity() {
			divCfg = &task.Strategy.Diversity
			break
		}
	}

	result := make([]*core.Item, 0, len(boosted))
	if divCfg == nil {
		for _, bi := range boosted {
			result = append(result, bi.item)
		}
		return result
	}

	boostedCount := 0
	lastBoostedPos := -divCfg.MinGap - 1
	for _, bi := range boosted {
		if bi.isBoosted {
			pos := len(result)
			windowStart := pos - divCfg.WindowSize
			if windowStart < 0 {
				windowStart = 0
			}
			boostedInWindow := 0
			for j := windowStart; j < pos; j++ {
				if isBoostedItem(result[j]) {
					boostedInWindow++
				}
			}
			if boostedInWindow >= divCfg.MaxBoosted {
				result = append(result, bi.item)
				continue
			}
			if divCfg.MinGap > 0 && pos-lastBoostedPos <= divCfg.MinGap {
				result = append(result, bi.item)
				continue
			}
			lastBoostedPos = pos
			boostedCount++
		}
		result = append(result, bi.item)
	}
	return result
}

func isBoostedItem(item *core.Item) bool {
	if item == nil || item.Labels == nil {
		return false
	}
	_, ok := item.Labels[LabelKeyTrafficControlID]
	return ok
}

type pinCandidate struct {
	item      *core.Item
	positions []int
}

func applyPinPositions(items []*core.Item, pins []*pinCandidate) []*core.Item {
	if len(pins) == 0 {
		return items
	}
	for _, pin := range pins {
		for _, pos := range pin.positions {
			if pos < 0 || pos > len(items) {
				continue
			}
			// Remove item from current position
			idx := -1
			for i, it := range items {
				if it == pin.item {
					idx = i
					break
				}
			}
			if idx < 0 {
				continue
			}
			items = append(items[:idx], items[idx+1:]...)
			if pos > len(items) {
				pos = len(items)
			}
			items = append(items[:pos], append([]*core.Item{pin.item}, items[pos:]...)...)
			break
		}
	}
	return items
}

func filterActiveTasks(tasks []*trafficctrl.Task, now time.Time) []*trafficctrl.Task {
	active := make([]*trafficctrl.Task, 0, len(tasks))
	for _, t := range tasks {
		if t.IsActive(now) {
			active = append(active, t)
		}
	}
	return active
}
