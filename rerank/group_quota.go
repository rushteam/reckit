package rerank

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/dsl"
)

// GroupQuotaStrategy 组间配额分配策略。
type GroupQuotaStrategy string

const (
	// GroupQuotaSoftmax 按组分数总和的 softmax 分配比例。
	GroupQuotaSoftmax GroupQuotaStrategy = "softmax"
	// GroupQuotaAvg 各组等权分配。
	GroupQuotaAvg GroupQuotaStrategy = "avg"
)

// ExprGroup 用布尔表达式定义虚拟分组（一条 item 归入第一个命中的 ExprGroup）。
type ExprGroup struct {
	Name  string
	Expr  string
	Quota int // > 0 时为固定配额，覆盖按比例分配

	compiledExpr *dsl.CompiledExpr
}

func (g *ExprGroup) getCompiledExpr() (*dsl.CompiledExpr, error) {
	if g.compiledExpr != nil {
		return g.compiledExpr, nil
	}
	c, err := dsl.Compile(g.Expr)
	if err != nil {
		return nil, err
	}
	g.compiledExpr = c
	return c, nil
}

// GroupQuotaNode 按维度字段分组后，根据策略做加权配额分配。
//
// 两种分组方式（互斥，ExprGroups 优先）：
//   - FieldKey：按 Labels > Meta > Features 的字段值自动分组
//   - ExprGroups：用 CEL 表达式定义命名分组
type GroupQuotaNode struct {
	N          int                // 输出总量；<= 0 取 len(items)
	FieldKey   string             // 按字段值分组
	ExprGroups []ExprGroup        // 按表达式分组（优先于 FieldKey）
	Strategy   GroupQuotaStrategy // 默认 softmax
	GroupMin   int                // 每组最少保留
	GroupMax   int                // 每组最多保留（0 = 不限）
	GroupCaps  map[string]int     // 可选：按维度值/组名单独设上限
}

func (n *GroupQuotaNode) Name() string {
	return "rerank.group_quota"
}

func (n *GroupQuotaNode) Kind() pipeline.Kind {
	return pipeline.KindReRank
}

func (n *GroupQuotaNode) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if len(items) == 0 {
		return items, nil
	}

	total := n.N
	if total <= 0 {
		total = len(items)
	}

	var groups map[string][]*core.Item
	var order []string
	var err error

	if len(n.ExprGroups) > 0 {
		groups, order, err = n.groupByExpr(ctx, rctx, items)
		if err != nil {
			return nil, err
		}
	} else {
		groups, order = n.groupByField(items)
	}

	if len(groups) == 0 {
		if total < len(items) {
			return items[:total], nil
		}
		return items, nil
	}

	for _, g := range groups {
		sort.SliceStable(g, func(i, j int) bool {
			return g[i].Score > g[j].Score
		})
	}

	quotas := n.allocateQuotas(groups, order, total)

	out := make([]*core.Item, 0, total)
	for _, key := range order {
		g := groups[key]
		q := quotas[key]
		if q > len(g) {
			q = len(g)
		}
		out = append(out, g[:q]...)
	}

	if len(out) < total {
		used := make(map[string]bool, len(out))
		for _, it := range out {
			used[it.ID] = true
		}
		groupCount := make(map[string]int, len(order))
		for _, key := range order {
			groupCount[key] = quotas[key]
			if quotas[key] > len(groups[key]) {
				groupCount[key] = len(groups[key])
			}
		}
		for _, key := range order {
			cap := n.groupCap(key)
			maxCap := n.GroupMax
			for _, it := range groups[key] {
				if len(out) >= total {
					break
				}
				if used[it.ID] {
					continue
				}
				if maxCap > 0 && groupCount[key] >= maxCap {
					break
				}
				if cap > 0 && groupCount[key] >= cap {
					break
				}
				out = append(out, it)
				used[it.ID] = true
				groupCount[key]++
			}
		}
	}

	sort.SliceStable(out, func(i, j int) bool {
		return out[i].Score > out[j].Score
	})

	if len(out) > total {
		out = out[:total]
	}
	return out, nil
}

func (n *GroupQuotaNode) groupByField(items []*core.Item) (map[string][]*core.Item, []string) {
	key := n.FieldKey
	if key == "" {
		key = "category"
	}
	groups := make(map[string][]*core.Item)
	var order []string
	seen := make(map[string]bool)
	for _, it := range items {
		if it == nil {
			continue
		}
		v, _ := it.GetValue(key)
		groups[v] = append(groups[v], it)
		if !seen[v] {
			order = append(order, v)
			seen[v] = true
		}
	}
	return groups, order
}

func (n *GroupQuotaNode) groupByExpr(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) (map[string][]*core.Item, []string, error) {
	groups := make(map[string][]*core.Item)
	order := make([]string, 0, len(n.ExprGroups))
	for i := range n.ExprGroups {
		order = append(order, n.ExprGroups[i].Name)
	}

	for _, it := range items {
		if it == nil {
			continue
		}
		matched := false
		for i := range n.ExprGroups {
			eg := &n.ExprGroups[i]
			c, err := eg.getCompiledExpr()
			if err != nil {
				return nil, nil, fmt.Errorf("rerank.group_quota ExprGroup %q: %w", eg.Name, err)
			}
			ok, err := c.Eval(it, rctx)
			if err != nil {
				return nil, nil, fmt.Errorf("rerank.group_quota ExprGroup %q eval: %w", eg.Name, err)
			}
			if ok {
				groups[eg.Name] = append(groups[eg.Name], it)
				matched = true
				break
			}
		}
		if !matched {
			groups[""] = append(groups[""], it)
		}
	}
	if _, ok := groups[""]; ok {
		hasEmpty := false
		for _, o := range order {
			if o == "" {
				hasEmpty = true
				break
			}
		}
		if !hasEmpty {
			order = append(order, "")
		}
	}
	return groups, order, nil
}

func (n *GroupQuotaNode) allocateQuotas(
	groups map[string][]*core.Item,
	order []string,
	total int,
) map[string]int {
	quotas := make(map[string]int, len(order))

	// ExprGroups 固定 Quota 优先：有 Quota > 0 的组直接采用，剩余量分配给其他组
	exprQuotaMap := n.exprGroupQuotaMap()
	fixedTotal := 0
	var dynamicOrder []string
	for _, key := range order {
		if q, ok := exprQuotaMap[key]; ok && q > 0 {
			quotas[key] = q
			fixedTotal += q
		} else {
			dynamicOrder = append(dynamicOrder, key)
		}
	}

	remaining := total - fixedTotal
	if remaining < 0 {
		remaining = 0
	}

	if len(dynamicOrder) > 0 && remaining > 0 {
		strategy := n.Strategy
		if strategy == "" {
			strategy = GroupQuotaSoftmax
		}
		switch strategy {
		case GroupQuotaSoftmax:
			scores := make(map[string]float64, len(dynamicOrder))
			for _, key := range dynamicOrder {
				var s float64
				for _, it := range groups[key] {
					s += it.Score
				}
				scores[key] = s
			}
			maxScore := math.Inf(-1)
			for _, s := range scores {
				if s > maxScore {
					maxScore = s
				}
			}
			var sumExp float64
			exps := make(map[string]float64, len(dynamicOrder))
			for _, key := range dynamicOrder {
				e := math.Exp(scores[key] - maxScore)
				exps[key] = e
				sumExp += e
			}
			if sumExp == 0 {
				sumExp = 1
			}
			for _, key := range dynamicOrder {
				quotas[key] = int(math.Round(float64(remaining) * exps[key] / sumExp))
			}

		case GroupQuotaAvg:
			base := remaining / len(dynamicOrder)
			rem := remaining % len(dynamicOrder)
			for i, key := range dynamicOrder {
				quotas[key] = base
				if i < rem {
					quotas[key]++
				}
			}
		}
	}

	for key := range quotas {
		cap := n.groupCap(key)
		if cap > 0 && quotas[key] > cap {
			quotas[key] = cap
		}
		if n.GroupMin > 0 && quotas[key] < n.GroupMin {
			quotas[key] = n.GroupMin
		}
		if n.GroupMax > 0 && quotas[key] > n.GroupMax {
			quotas[key] = n.GroupMax
		}
	}
	return quotas
}

func (n *GroupQuotaNode) exprGroupQuotaMap() map[string]int {
	m := make(map[string]int, len(n.ExprGroups))
	for i := range n.ExprGroups {
		if n.ExprGroups[i].Quota > 0 {
			m[n.ExprGroups[i].Name] = n.ExprGroups[i].Quota
		}
	}
	return m
}

func (n *GroupQuotaNode) groupCap(key string) int {
	if n.GroupCaps == nil {
		return 0
	}
	return n.GroupCaps[key]
}
