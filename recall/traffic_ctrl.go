package recall

import (
	"context"
	"log/slog"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/trafficctrl"
	"github.com/rushteam/reckit/pkg/utils"
)

// TrafficCtrlRecallSource 从活跃的流量调控任务中提取 static_ids，
// 确保 TC 指定的物品一定出现在候选池中，使后续 TrafficControlNode 加权有靶可打。
type TrafficCtrlRecallSource struct {
	TaskRepo trafficctrl.TaskRepository
	Scene    string // 可选固定场景；空则取 rctx.Scene
	MaxItems int    // 最大召回条数，默认 50

	// UserAttrsFunc 从 RecommendContext 提取用户属性用于 Audience 过滤。
	// nil 时默认使用 rctx.Attributes。
	UserAttrsFunc func(rctx *core.RecommendContext) map[string]any
}

func (s *TrafficCtrlRecallSource) Name() string { return "recall.traffic_ctrl" }

func (s *TrafficCtrlRecallSource) Recall(ctx context.Context, rctx *core.RecommendContext) ([]*core.Item, error) {
	if s.TaskRepo == nil {
		return nil, nil
	}

	scene := s.Scene
	if scene == "" && rctx != nil {
		scene = rctx.Scene
	}

	tasks, err := s.TaskRepo.GetActiveTasks(ctx, scene)
	if err != nil {
		slog.WarnContext(ctx, "traffic_ctrl_recall: load tasks failed, skip",
			slog.String("error", err.Error()))
		return nil, nil
	}
	if len(tasks) == 0 {
		return nil, nil
	}

	maxItems := s.MaxItems
	if maxItems <= 0 {
		maxItems = 50
	}

	var userAttrs map[string]any
	if s.UserAttrsFunc != nil {
		userAttrs = s.UserAttrsFunc(rctx)
	} else if rctx != nil {
		userAttrs = rctx.Attributes
	}

	seen := make(map[string]struct{}, maxItems)
	items := make([]*core.Item, 0, maxItems)

	for _, task := range tasks {
		if !task.ItemPool.HasStaticIDs() {
			continue
		}
		if !task.Audience.MatchAll(userAttrs) {
			continue
		}
		for _, id := range task.ItemPool.StaticIDs {
			if _, dup := seen[id]; dup {
				continue
			}
			seen[id] = struct{}{}
			items = append(items, &core.Item{
				ID: id,
				Labels: map[string]utils.Label{
					"recall_source":   {Value: s.Name(), Source: s.Name()},
					"recall_priority": {Value: "50", Source: s.Name()},
				},
			})
			if len(items) >= maxItems {
				break
			}
		}
		if len(items) >= maxItems {
			break
		}
	}

	return items, nil
}
