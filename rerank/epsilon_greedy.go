package rerank

import (
	"context"
	"math/rand"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

// EpsilonGreedyNode 以概率 Epsilon 将排序靠后的物品提升到前列，增加探索性。
//
// 工作方式：
//  1. 将 items 分为 exploit 区（前 ExploitSize 条）和 explore 池（其余）
//  2. 对 exploit 区中每个位置，以 Epsilon 概率替换为 explore 池中随机一条
//  3. 被替换出的 exploit 物品回到 explore 池末尾
type EpsilonGreedyNode struct {
	Epsilon     float64
	ExploitSize int
	Rand        *rand.Rand
}

func (n *EpsilonGreedyNode) Name() string        { return "rerank.epsilon_greedy" }
func (n *EpsilonGreedyNode) Kind() pipeline.Kind { return pipeline.KindReRank }

func (n *EpsilonGreedyNode) Process(
	_ context.Context,
	_ *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	eps := n.Epsilon
	if eps <= 0 || eps > 1 {
		return items, nil
	}
	exploitN := n.ExploitSize
	if exploitN <= 0 {
		exploitN = len(items)
	}
	if exploitN > len(items) {
		exploitN = len(items)
	}
	if exploitN >= len(items) {
		return items, nil
	}

	rng := n.Rand
	if rng == nil {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	out := make([]*core.Item, len(items))
	copy(out, items)

	pool := out[exploitN:]
	for i := 0; i < exploitN; i++ {
		if len(pool) == 0 {
			break
		}
		if rng.Float64() < eps {
			j := rng.Intn(len(pool))
			out[i], pool[j] = pool[j], out[i]
		}
	}
	return out, nil
}
