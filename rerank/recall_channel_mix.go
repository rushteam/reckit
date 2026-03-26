package rerank

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/filter"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/conv"
	"github.com/rushteam/reckit/pkg/dsl"
)

// DefaultRecallSourceLabel 与 recall 合并策略中 recall_source 约定一致。
const DefaultRecallSourceLabel = "recall_source"

// RemainderPolicy 未进入任何规则槽位的候选如何处理。
type RemainderPolicy string

const (
	// RemainderAppend 将剩余候选按原顺序接到输出尾部（总长度可能大于 OutputSize）。
	RemainderAppend RemainderPolicy = "append"
	// RemainderDiscard 丢弃未进槽位的候选。
	RemainderDiscard RemainderPolicy = "discard"
)

// ChannelSlotKind 槽位策略：固定序号或区间内随机占位。
type ChannelSlotKind string

const (
	ChannelSlotKindFixed  ChannelSlotKind = "fixed_slot"
	ChannelSlotKindRandom ChannelSlotKind = "random_slot"
)

// ChannelRule 描述一路召回通道在输出序列中的占位方式。
// 通道匹配：默认取 LabelKey 对应 Label 的 Value 在按 "|" 合并前的**首段**（与 Diversity 等模块对合并标签的用法一致）。
// Channels 非空时须与首段精确匹配；为空表示不限制通道，仅 Expr/Filter 生效。
type ChannelRule struct {
	Kind ChannelSlotKind

	Channels []string

	// FixedSlots：0-based 输出下标，按列表顺序依次为匹配到的候选占位（该下标须仍为空）。
	FixedSlots []int

	// RandomSlotStart（含）到 RandomSlotEnd（不含）之间，随机选取空位放入匹配候选。
	RandomSlotStart int
	RandomSlotEnd   int
	RandomCount     int

	Expr   string
	Filter filter.Filter

	compiledExpr *dsl.CompiledExpr // 懒编译缓存
}

// RecallChannelMix 在精排之后按召回通道将候选填入固定或随机槽位，用于运营位次/通道曝光。
// 与 recall.MergeStrategy 分工：MergeStrategy 负责多路召回**合并去重与配额**；本节点负责**精排后**的槽位编排。
type RecallChannelMix struct {
	LabelKey string
	// OutputSize 输出槽位长度；为 0 时尝试 rctx.Params["size"]（正整数），再否则为 len(items)。
	OutputSize int
	// RemainderPolicy 默认 RemainderAppend。
	RemainderPolicy RemainderPolicy
	Rules           []ChannelRule
	// Rand 可选；为 nil 时使用全局随机源（测试可注入固定种子）。
	Rand *rand.Rand
}

func (n *RecallChannelMix) Name() string {
	return "rerank.recall_channel_mix"
}

func (n *RecallChannelMix) Kind() pipeline.Kind {
	return pipeline.KindReRank
}

func (r *ChannelRule) getCompiledExpr() (*dsl.CompiledExpr, error) {
	if r.compiledExpr != nil {
		return r.compiledExpr, nil
	}
	c, err := dsl.Compile(r.Expr)
	if err != nil {
		return nil, err
	}
	r.compiledExpr = c
	return c, nil
}

// PrimaryRecallChannel 返回 labelKey 对应召回通道的「主通道名」：取合并标签 Value 按 "|" 分割后的首段非空片段。
func PrimaryRecallChannel(it *core.Item, labelKey string) string {
	if it == nil || labelKey == "" {
		return ""
	}
	v, ok := it.GetValue(labelKey)
	if !ok {
		return ""
	}
	if idx := strings.IndexByte(v, '|'); idx >= 0 {
		v = v[:idx]
	}
	return strings.TrimSpace(v)
}

func (n *RecallChannelMix) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if len(items) == 0 {
		return items, nil
	}

	labelKey := n.LabelKey
	if labelKey == "" {
		labelKey = DefaultRecallSourceLabel
	}

	outLen := n.outputLen(rctx, len(items))
	if outLen <= 0 {
		outLen = len(items)
	}

	slots := make([]*core.Item, outLen)

	// pool 是非 nil 的候选，taken 标记已占位的下标——避免每次 O(n) 切片拷贝
	pool := make([]*core.Item, 0, len(items))
	for _, it := range items {
		if it != nil {
			pool = append(pool, it)
		}
	}
	taken := make([]bool, len(pool))

	rng := n.Rand
	if rng == nil {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	for i := range n.Rules {
		rule := &n.Rules[i]
		switch rule.Kind {
		case ChannelSlotKindFixed:
			if err := n.applyFixed(ctx, rctx, labelKey, rule, slots, pool, taken); err != nil {
				return nil, err
			}
		case ChannelSlotKindRandom:
			if err := n.applyRandom(ctx, rctx, labelKey, rule, slots, pool, taken, rng); err != nil {
				return nil, err
			}
		default:
			return nil, fmt.Errorf("rerank.recall_channel_mix: unknown rule kind %q", rule.Kind)
		}
	}

	// 顺序填满空槽
	pi := 0
	for si := range slots {
		if slots[si] != nil {
			continue
		}
		for pi < len(pool) && taken[pi] {
			pi++
		}
		if pi < len(pool) {
			slots[si] = pool[pi]
			taken[pi] = true
			pi++
		}
	}

	out := make([]*core.Item, 0, outLen+len(pool))
	for _, it := range slots {
		if it != nil {
			out = append(out, it)
		}
	}

	policy := n.RemainderPolicy
	if policy == "" {
		policy = RemainderAppend
	}
	if policy == RemainderAppend {
		for i, it := range pool {
			if !taken[i] {
				out = append(out, it)
			}
		}
	}
	return out, nil
}

func (n *RecallChannelMix) outputLen(rctx *core.RecommendContext, nItems int) int {
	if n.OutputSize > 0 {
		return n.OutputSize
	}
	if rctx != nil && rctx.Params != nil {
		if v, ok := rctx.Params["size"]; ok {
			if s, ok2 := conv.ToInt(v); ok2 && s > 0 {
				return s
			}
		}
	}
	return nItems
}

func (n *RecallChannelMix) applyFixed(
	ctx context.Context,
	rctx *core.RecommendContext,
	labelKey string,
	rule *ChannelRule,
	slots []*core.Item,
	pool []*core.Item,
	taken []bool,
) error {
	for _, pos := range rule.FixedSlots {
		if pos < 0 || pos >= len(slots) || slots[pos] != nil {
			continue
		}
		idx, it, err := takeFirstMatching(ctx, rctx, labelKey, rule, pool, taken)
		if err != nil {
			return err
		}
		if it == nil {
			continue
		}
		slots[pos] = it
		taken[idx] = true
	}
	return nil
}

func (n *RecallChannelMix) applyRandom(
	ctx context.Context,
	rctx *core.RecommendContext,
	labelKey string,
	rule *ChannelRule,
	slots []*core.Item,
	pool []*core.Item,
	taken []bool,
	rng *rand.Rand,
) error {
	if rule.RandomSlotEnd <= rule.RandomSlotStart {
		return fmt.Errorf("rerank.recall_channel_mix: random_slot requires RandomSlotEnd > RandomSlotStart")
	}
	if rule.RandomCount <= 0 {
		return nil
	}

	availPos := make([]int, 0, rule.RandomSlotEnd-rule.RandomSlotStart)
	for p := rule.RandomSlotStart; p < rule.RandomSlotEnd && p < len(slots); p++ {
		if p >= 0 && slots[p] == nil {
			availPos = append(availPos, p)
		}
	}
	if len(availPos) == 0 {
		return nil
	}

	placed := 0
	for placed < rule.RandomCount && len(availPos) > 0 {
		idx, it, err := takeFirstMatching(ctx, rctx, labelKey, rule, pool, taken)
		if err != nil {
			return err
		}
		if it == nil {
			break
		}
		j := rng.Intn(len(availPos))
		pos := availPos[j]
		availPos[j] = availPos[len(availPos)-1]
		availPos = availPos[:len(availPos)-1]

		slots[pos] = it
		taken[idx] = true
		placed++
	}
	return nil
}

func takeFirstMatching(
	ctx context.Context,
	rctx *core.RecommendContext,
	labelKey string,
	rule *ChannelRule,
	pool []*core.Item,
	taken []bool,
) (idx int, it *core.Item, err error) {
	for i, cand := range pool {
		if taken[i] {
			continue
		}
		ok, err := itemMatchesRule(ctx, rctx, labelKey, rule, cand)
		if err != nil {
			return 0, nil, err
		}
		if ok {
			return i, cand, nil
		}
	}
	return 0, nil, nil
}

// itemMatchesRule 若 rule 中 Filter/Expr 均未设置，则仅按通道匹配。
func itemMatchesRule(
	ctx context.Context,
	rctx *core.RecommendContext,
	labelKey string,
	rule *ChannelRule,
	item *core.Item,
) (bool, error) {
	if item == nil {
		return false, nil
	}
	if len(rule.Channels) > 0 {
		prim := PrimaryRecallChannel(item, labelKey)
		found := false
		for _, ch := range rule.Channels {
			if ch == prim {
				found = true
				break
			}
		}
		if !found {
			return false, nil
		}
	}
	if rule.Filter != nil {
		skip, err := rule.Filter.ShouldFilter(ctx, rctx, item)
		if err != nil {
			return false, err
		}
		if skip {
			return false, nil
		}
	}
	if rule.Expr != "" {
		c, err := rule.getCompiledExpr()
		if err != nil {
			return false, err
		}
		ok, err := c.Eval(item, rctx)
		if err != nil {
			return false, err
		}
		if !ok {
			return false, nil
		}
	}
	return true, nil
}
