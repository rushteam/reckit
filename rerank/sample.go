package rerank

import (
	"context"
	"math/rand"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

// SampleNode 从候选中采样 N 条。
// Shuffle=false 时等价于 TopNNode（前缀截断）；
// Shuffle=true 时先 Fisher-Yates 洗牌再取前 N，适用于召回/粗排后降量、增加探索性。
type SampleNode struct {
	N       int
	Shuffle bool
	Rand    *rand.Rand
}

func (n *SampleNode) Name() string {
	return "rerank.sample"
}

func (n *SampleNode) Kind() pipeline.Kind {
	return pipeline.KindReRank
}

func (n *SampleNode) Process(
	_ context.Context,
	_ *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if n.N <= 0 || len(items) <= n.N {
		return items, nil
	}
	if !n.Shuffle {
		return items[:n.N], nil
	}
	rng := n.Rand
	if rng == nil {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}
	cp := make([]*core.Item, len(items))
	copy(cp, items)
	rng.Shuffle(len(cp), func(i, j int) {
		cp[i], cp[j] = cp[j], cp[i]
	})
	return cp[:n.N], nil
}
