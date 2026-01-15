package recall

import (
	"context"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"

	"reckit/core"
	"reckit/pipeline"
	"reckit/pkg/utils"
)

// Fanout 是一个 Recall Node：并发执行多个召回源，并合并结果。
// 支持超时、限流、优先级合并策略。
type Fanout struct {
	Sources       []Source
	Dedup         bool
	Timeout       time.Duration // 每个召回源的超时时间
	MaxConcurrent int           // 最大并发数（0 表示无限制）
	MergeStrategy string        // 合并策略：first / union / priority（优先级按 Sources 顺序）
}

func (n *Fanout) Name() string        { return "recall.fanout" }
func (n *Fanout) Kind() pipeline.Kind { return pipeline.KindRecall }

func (n *Fanout) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	_ []*core.Item,
) ([]*core.Item, error) {
	if len(n.Sources) == 0 {
		return nil, nil
	}

	var (
		mu    sync.Mutex
		all   []*core.Item
		eg, _ = errgroup.WithContext(ctx)
	)

	// 限流：使用 semaphore 控制并发数
	sem := make(chan struct{}, n.MaxConcurrent)
	if n.MaxConcurrent <= 0 {
		close(sem) // 无限制时直接关闭，避免阻塞
	}

	for i, src := range n.Sources {
		s := src
		priority := i // 优先级（索引越小优先级越高）

		eg.Go(func() error {
			// 限流
			if n.MaxConcurrent > 0 {
				sem <- struct{}{}
				defer func() { <-sem }()
			}

			// 超时控制
			recallCtx := ctx
			if n.Timeout > 0 {
				var cancel context.CancelFunc
				recallCtx, cancel = context.WithTimeout(ctx, n.Timeout)
				defer cancel()
			}

			items, err := s.Recall(recallCtx, rctx)
			if err != nil {
				// 超时或错误时返回空结果，不中断其他召回源
				return nil
			}

			// 记录召回来源 label，方便 explain / 观测
			for _, it := range items {
				it.PutLabel("recall_source", utils.Label{Value: s.Name(), Source: "recall"})
				it.PutLabel("recall_priority", utils.Label{Value: string(rune('0' + priority)), Source: "recall"})
			}

			mu.Lock()
			all = append(all, items...)
			mu.Unlock()
			return nil
		})
	}

	if err := eg.Wait(); err != nil {
		return nil, err
	}

	// 合并策略
	switch n.MergeStrategy {
	case "priority":
		return n.mergeByPriority(all), nil
	case "union":
		return n.mergeUnion(all), nil
	default: // "first" 或默认
		return n.mergeFirst(all), nil
	}
}

// mergeFirst 按 ID 去重，保留第一个出现的（默认策略）。
func (n *Fanout) mergeFirst(all []*core.Item) []*core.Item {
	if !n.Dedup {
		return all
	}
	seen := make(map[string]*core.Item, len(all))
	out := make([]*core.Item, 0, len(all))
	for _, it := range all {
		if it == nil {
			continue
		}
		if old, ok := seen[it.ID]; ok {
			for k, v := range it.Labels {
				old.PutLabel(k, v)
			}
			continue
		}
		seen[it.ID] = it
		out = append(out, it)
	}
	return out
}

// mergeUnion 合并所有结果，不去重（用于需要保留所有来源的场景）。
func (n *Fanout) mergeUnion(all []*core.Item) []*core.Item {
	return all
}

// mergeByPriority 按优先级合并：相同 ID 时保留优先级更高的（索引更小）。
func (n *Fanout) mergeByPriority(all []*core.Item) []*core.Item {
	if !n.Dedup {
		return all
	}
	seen := make(map[string]*core.Item, len(all))
	for _, it := range all {
		if it == nil {
			continue
		}
		old, exists := seen[it.ID]
		if !exists {
			seen[it.ID] = it
			continue
		}
		// 比较优先级（priority label 的值）
		oldPriority := 999
		newPriority := 999
		if oldLbl, ok := old.Labels["recall_priority"]; ok {
			if len(oldLbl.Value) > 0 {
				oldPriority = int(oldLbl.Value[0] - '0')
			}
		}
		if newLbl, ok := it.Labels["recall_priority"]; ok {
			if len(newLbl.Value) > 0 {
				newPriority = int(newLbl.Value[0] - '0')
			}
		}
		// 保留优先级更高的（值更小）
		if newPriority < oldPriority {
			seen[it.ID] = it
		} else {
			// 合并 labels
			for k, v := range it.Labels {
				old.PutLabel(k, v)
			}
		}
	}
	out := make([]*core.Item, 0, len(seen))
	for _, it := range seen {
		out = append(out, it)
	}
	return out
}
