package recall

import (
	"sort"

	"github.com/rushteam/reckit/core"
)

// MergeStrategy 是合并策略接口，用于自定义多路召回结果的合并逻辑。
type MergeStrategy interface {
	Merge(items []*core.Item, dedup bool) []*core.Item
}

// FirstMergeStrategy 是默认的合并策略：按 ID 去重，保留第一个出现的。
type FirstMergeStrategy struct{}

func (s *FirstMergeStrategy) Merge(items []*core.Item, dedup bool) []*core.Item {
	if !dedup {
		return items
	}
	return dedupItems(items)
}

// UnionMergeStrategy 是并集策略：不去重，保留所有结果。
type UnionMergeStrategy struct{}

func (s *UnionMergeStrategy) Merge(items []*core.Item, dedup bool) []*core.Item {
	return items
}

// PriorityMergeStrategy 是按优先级合并的策略。
// 优先级由 Source 的索引决定（索引越小优先级越高），或通过 PriorityWeights 自定义。
type PriorityMergeStrategy struct {
	PriorityWeights map[string]int
}

func (s *PriorityMergeStrategy) Merge(items []*core.Item, dedup bool) []*core.Item {
	if !dedup {
		return items
	}
	seen := make(map[string]*core.Item, len(items))
	order := make([]string, 0, len(items))
	for _, it := range items {
		if it == nil {
			continue
		}
		old, exists := seen[it.ID]
		if !exists {
			seen[it.ID] = it
			order = append(order, it.ID)
			continue
		}
		oldPriority := s.getPriority(old)
		newPriority := s.getPriority(it)
		if newPriority < oldPriority {
			seen[it.ID] = it
		} else {
			for k, v := range it.Labels {
				old.PutLabel(k, v)
			}
		}
	}
	out := make([]*core.Item, 0, len(order))
	for _, id := range order {
		out = append(out, seen[id])
	}
	sort.SliceStable(out, func(i, j int) bool {
		pi, pj := s.getPriority(out[i]), s.getPriority(out[j])
		if pi != pj {
			return pi < pj
		}
		return out[i].ID < out[j].ID
	})
	return out
}

func (s *PriorityMergeStrategy) getPriority(item *core.Item) int {
	if s.PriorityWeights != nil {
		if sourceLbl, ok := item.Labels["recall_source"]; ok {
			if weight, ok := s.PriorityWeights[sourceLbl.Value]; ok {
				return weight
			}
		}
	}
	if lbl, ok := item.Labels["recall_priority"]; ok {
		if len(lbl.Value) > 0 {
			return int(lbl.Value[0] - '0')
		}
	}
	return 999
}

// ---------------------------------------------------------------------------
// 辅助函数
// ---------------------------------------------------------------------------

// groupBySource 按 recall_source label 将 items 分组，保持组内原始顺序。
func groupBySource(items []*core.Item) map[string][]*core.Item {
	groups := make(map[string][]*core.Item)
	for _, it := range items {
		if it == nil {
			continue
		}
		source := ""
		if lbl, ok := it.Labels["recall_source"]; ok {
			source = lbl.Value
		}
		groups[source] = append(groups[source], it)
	}
	return groups
}

// sourceOrder 返回 items 中各 recall_source 按首次出现顺序排列的列表（确定性）。
func sourceOrder(items []*core.Item) []string {
	seen := make(map[string]bool)
	var order []string
	for _, it := range items {
		if it == nil {
			continue
		}
		source := ""
		if lbl, ok := it.Labels["recall_source"]; ok {
			source = lbl.Value
		}
		if !seen[source] {
			order = append(order, source)
			seen[source] = true
		}
	}
	return order
}

// dedupItems 按 ID 去重，保留第一个出现的 item，合并后续重复 item 的 labels。
func dedupItems(items []*core.Item) []*core.Item {
	seen := make(map[string]*core.Item, len(items))
	out := make([]*core.Item, 0, len(items))
	for _, it := range items {
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

// sortByScoreDesc 按 Score 降序排列（稳定排序，相同分数保持原始顺序）。
func sortByScoreDesc(items []*core.Item) {
	sort.SliceStable(items, func(i, j int) bool {
		return items[i].Score > items[j].Score
	})
}

// ChainMergeStrategy 将多个 MergeStrategy 串联执行，前一个策略的输出作为下一个的输入。
// 适用于需要组合多种策略的场景，例如"先加权调分，再按配额截取"。
//
// dedup 仅传递给第一个策略，后续策略收到 dedup=false（避免重复去重）。
type ChainMergeStrategy struct {
	Strategies []MergeStrategy
}

func (s *ChainMergeStrategy) Merge(items []*core.Item, dedup bool) []*core.Item {
	for i, strategy := range s.Strategies {
		if i == 0 {
			items = strategy.Merge(items, dedup)
		} else {
			items = strategy.Merge(items, false)
		}
	}
	return items
}

// WeightedScoreMergeStrategy 按召回源权重调整 item 分数，然后按分数降序排列。
// 适用于通过权重控制各源贡献度的场景。
type WeightedScoreMergeStrategy struct {
	// SourceWeights 各召回源的权重乘数（source name -> weight multiplier）。
	// 未配置的源使用 DefaultWeight。
	SourceWeights map[string]float64

	// DefaultWeight 未在 SourceWeights 中配置的源的默认权重，默认 1.0。
	DefaultWeight float64

	// TopN 合并后取前 N 个，0 表示不限制。
	TopN int
}

func (s *WeightedScoreMergeStrategy) Merge(items []*core.Item, dedup bool) []*core.Item {
	if dedup {
		items = dedupItems(items)
	}

	defaultW := s.DefaultWeight
	if defaultW == 0 {
		defaultW = 1.0
	}

	for _, it := range items {
		w := defaultW
		if s.SourceWeights != nil {
			if lbl, ok := it.Labels["recall_source"]; ok {
				if sw, ok := s.SourceWeights[lbl.Value]; ok {
					w = sw
				}
			}
		}
		it.Score *= w
	}

	sortByScoreDesc(items)

	if s.TopN > 0 && len(items) > s.TopN {
		items = items[:s.TopN]
	}
	return items
}

// QuotaMergeStrategy 每个召回源取固定数量的 item。
// 适用于明确知道各源应出多少条的场景（如运营配置）。
type QuotaMergeStrategy struct {
	// SourceQuotas 各源配额（source name -> 数量）。
	SourceQuotas map[string]int

	// DefaultQuota 未在 SourceQuotas 中配置的源的默认配额，0 表示不取。
	DefaultQuota int
}

func (s *QuotaMergeStrategy) Merge(items []*core.Item, dedup bool) []*core.Item {
	if dedup {
		items = dedupItems(items)
	}

	order := sourceOrder(items)
	groups := groupBySource(items)

	var out []*core.Item
	for _, source := range order {
		group := groups[source]
		quota := s.DefaultQuota
		if s.SourceQuotas != nil {
			if q, ok := s.SourceQuotas[source]; ok {
				quota = q
			}
		}
		if quota <= 0 {
			continue
		}

		sortByScoreDesc(group)
		if len(group) > quota {
			group = group[:quota]
		}
		out = append(out, group...)
	}

	return out
}

// RatioMergeStrategy 按比例从各召回源取 item，总量由 TotalLimit 控制。
// 适用于"hot 占 20%、cf 占 30%、ann 占 50%"的场景。
type RatioMergeStrategy struct {
	// SourceRatios 各源比例（source name -> 比例 0.0~1.0）。
	// 比例之和不必为 1.0，内部会自动归一化。
	SourceRatios map[string]float64

	// TotalLimit 总数量限制，必须 > 0。
	TotalLimit int
}

func (s *RatioMergeStrategy) Merge(items []*core.Item, dedup bool) []*core.Item {
	if s.TotalLimit <= 0 {
		return nil
	}
	if dedup {
		items = dedupItems(items)
	}

	order := sourceOrder(items)
	groups := groupBySource(items)
	for _, group := range groups {
		sortByScoreDesc(group)
	}

	// 归一化比例
	var totalRatio float64
	for _, r := range s.SourceRatios {
		if r > 0 {
			totalRatio += r
		}
	}
	if totalRatio == 0 {
		totalRatio = 1.0
	}

	// 按 order 顺序分配配额（floor），收集余量
	type sourceAlloc struct {
		name     string
		quota    int
		fraction float64
	}
	allocs := make([]sourceAlloc, 0, len(s.SourceRatios))
	allocated := 0
	for _, source := range order {
		ratio := s.SourceRatios[source]
		if ratio <= 0 {
			continue
		}
		normalized := ratio / totalRatio
		exact := normalized * float64(s.TotalLimit)
		quota := int(exact)
		allocs = append(allocs, sourceAlloc{
			name:     source,
			quota:    quota,
			fraction: exact - float64(quota),
		})
		allocated += quota
	}

	// 余量分配给小数部分最大的源（fraction 相同时保持 allocs 中的原始顺序）
	remainder := s.TotalLimit - allocated
	if remainder > 0 {
		sort.SliceStable(allocs, func(i, j int) bool {
			return allocs[i].fraction > allocs[j].fraction
		})
		for i := range allocs {
			if remainder <= 0 {
				break
			}
			allocs[i].quota++
			remainder--
		}
	}

	quotaMap := make(map[string]int, len(allocs))
	for _, a := range allocs {
		quotaMap[a.name] = a.quota
	}

	var out []*core.Item
	totalTaken := 0
	shortfall := 0

	// 第一轮：按 order 顺序取配额
	for _, source := range order {
		quota := quotaMap[source]
		group := groups[source]
		take := quota
		if take > len(group) {
			shortfall += take - len(group)
			take = len(group)
		}
		if take > 0 {
			out = append(out, group[:take]...)
			totalTaken += take
		}
	}

	// 第二轮：按 order 顺序余量重分配
	if shortfall > 0 && totalTaken < s.TotalLimit {
		need := s.TotalLimit - totalTaken
		for _, source := range order {
			if need <= 0 {
				break
			}
			group := groups[source]
			taken := quotaMap[source]
			if taken >= len(group) {
				continue
			}
			extra := len(group) - taken
			if extra > need {
				extra = need
			}
			out = append(out, group[taken:taken+extra]...)
			need -= extra
		}
	}

	return out
}

// HybridRatioMergeStrategy 混合比例合并：
// 1) 可选按 recall_priority 预排序并去重（保留优先级高者）
// 2) 未配置在 SourceRatios 的源整路保留（按组内分数降序）
// 3) SourceRatios 中配置的源在剩余槽位内按比例分配
//
// 适用于“核心召回路固定保留 + 实验召回按比例占坑”的在线场景。
type HybridRatioMergeStrategy struct {
	// SourceRatios 显式参与比例分配的源（source -> ratio）
	SourceRatios map[string]float64
	// TotalLimit 合并总上限，<=0 时视为不截断（使用去重后总长度）
	TotalLimit int
	// DropUnconfiguredSources 是否丢弃未配置 ratio 的源，默认 false（即保留）
	DropUnconfiguredSources bool
	// SortByPriorityBeforeDedup 去重前是否按 recall_priority 排序，默认 false
	SortByPriorityBeforeDedup bool
}

func (s *HybridRatioMergeStrategy) Merge(items []*core.Item, dedup bool) []*core.Item {
	if len(items) == 0 {
		return items
	}

	if s.SortByPriorityBeforeDedup {
		sort.SliceStable(items, func(i, j int) bool {
			pi, pj := priorityFromLabel(items[i]), priorityFromLabel(items[j])
			if pi != pj {
				return pi < pj
			}
			if items[i] == nil || items[j] == nil {
				return false
			}
			return items[i].ID < items[j].ID
		})
	}

	if dedup {
		items = dedupItems(items)
	}

	limit := s.TotalLimit
	if limit <= 0 {
		limit = len(items)
	}

	keepUnconfigured := !s.DropUnconfiguredSources

	order := sourceOrder(items)
	groups := groupBySource(items)
	for _, g := range groups {
		sortByScoreDesc(g)
	}

	out := make([]*core.Item, 0, len(items))
	explicitItems := make([]*core.Item, 0, len(items))
	explicitRatios := make(map[string]float64)

	for _, src := range order {
		group := groups[src]
		if len(group) == 0 {
			continue
		}
		ratio, configured := s.SourceRatios[src]
		if configured && ratio > 0 {
			explicitRatios[src] = ratio
			explicitItems = append(explicitItems, group...)
			continue
		}
		if keepUnconfigured {
			out = append(out, group...)
		}
	}

	remain := limit - len(out)
	if remain > 0 && len(explicitItems) > 0 {
		merged := (&RatioMergeStrategy{
			SourceRatios: explicitRatios,
			TotalLimit:   remain,
		}).Merge(explicitItems, false)
		out = append(out, merged...)
	}

	if len(out) > limit {
		out = out[:limit]
	}
	return out
}

func priorityFromLabel(it *core.Item) int {
	if it == nil {
		return 999
	}
	lbl, ok := it.Labels["recall_priority"]
	if !ok || len(lbl.Value) == 0 {
		return 999
	}
	c := lbl.Value[0]
	if c >= '0' && c <= '9' {
		return int(c - '0')
	}
	return 999
}

// RoundRobinMergeStrategy 从各召回源轮流取 item，实现均匀交叉排列。
// 适用于信息流等需要内容多样性的场景。
type RoundRobinMergeStrategy struct {
	// SourceOrder 轮询顺序（可选），未指定时按源首次出现的顺序。
	SourceOrder []string

	// TopN 合并后取前 N 个，0 表示不限制。
	TopN int
}

func (s *RoundRobinMergeStrategy) Merge(items []*core.Item, dedup bool) []*core.Item {
	if dedup {
		items = dedupItems(items)
	}

	groups := groupBySource(items)
	for _, group := range groups {
		sortByScoreDesc(group)
	}

	// 确定轮询顺序
	order := s.SourceOrder
	if len(order) == 0 {
		seen := make(map[string]bool)
		for _, it := range items {
			source := ""
			if lbl, ok := it.Labels["recall_source"]; ok {
				source = lbl.Value
			}
			if !seen[source] {
				order = append(order, source)
				seen[source] = true
			}
		}
	}

	// 各源的取值游标
	cursors := make(map[string]int, len(order))
	var out []*core.Item

	for {
		added := false
		for _, source := range order {
			if s.TopN > 0 && len(out) >= s.TopN {
				return out
			}
			group := groups[source]
			idx := cursors[source]
			if idx < len(group) {
				out = append(out, group[idx])
				cursors[source] = idx + 1
				added = true
			}
		}
		if !added {
			break
		}
	}

	if s.TopN > 0 && len(out) > s.TopN {
		out = out[:s.TopN]
	}
	return out
}

// WaterfallMergeStrategy 高优先级源优先填满，不足时由低优先级源补充。
// 适用于"优先用个性化结果，不够再用热门兜底"的场景。
type WaterfallMergeStrategy struct {
	// SourcePriority 源优先级顺序（高 -> 低），不在列表中的源排在最后。
	SourcePriority []string

	// TotalLimit 总数量限制，必须 > 0。
	TotalLimit int

	// SourceLimits 每源最大数量（可选），防止单源占满全部配额。
	// 未配置的源无单独限制，受 TotalLimit 约束。
	SourceLimits map[string]int
}

func (s *WaterfallMergeStrategy) Merge(items []*core.Item, dedup bool) []*core.Item {
	if s.TotalLimit <= 0 {
		return nil
	}
	if dedup {
		items = dedupItems(items)
	}

	groups := groupBySource(items)
	for _, group := range groups {
		sortByScoreDesc(group)
	}

	// 构建完整的优先级列表：显式指定的 + 未指定的（按首次出现顺序）
	prioritySet := make(map[string]bool, len(s.SourcePriority))
	order := make([]string, 0, len(groups))
	for _, src := range s.SourcePriority {
		if _, exists := groups[src]; exists {
			order = append(order, src)
			prioritySet[src] = true
		}
	}
	for _, it := range items {
		source := ""
		if lbl, ok := it.Labels["recall_source"]; ok {
			source = lbl.Value
		}
		if !prioritySet[source] {
			order = append(order, source)
			prioritySet[source] = true
		}
	}

	var out []*core.Item
	for _, source := range order {
		if len(out) >= s.TotalLimit {
			break
		}
		group := groups[source]
		remaining := s.TotalLimit - len(out)

		limit := remaining
		if s.SourceLimits != nil {
			if sl, ok := s.SourceLimits[source]; ok && sl < limit {
				limit = sl
			}
		}
		if limit > len(group) {
			limit = len(group)
		}

		out = append(out, group[:limit]...)
	}

	return out
}
