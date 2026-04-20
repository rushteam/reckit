package rerank

import (
	"context"
	"strings"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

// DiversityConstraint 单条多样性约束规则。
// 多条 Constraint 之间为 AND 语义——候选物品需同时满足所有约束才算"完美匹配"。
// 当无完美匹配时，选 Weight 总和最高的候选作为回退。
type DiversityConstraint struct {
	// Dimensions 组合维度；多个维度的值用 "\x00" 拼接成复合 key。
	// 值从 Labels > Meta > Features 读取。
	Dimensions []string

	// MaxConsecutive 最大连续同组数（0 = 不检查连续）。
	// 例如 2 表示同组最多连续出现 2 次。
	MaxConsecutive int

	// WindowSize 滑动窗口大小（与 MaxPerWindow 搭配）。
	WindowSize int

	// MaxPerWindow 窗口内同组最大出现次数（0 = 不检查窗口频率）。
	// 需要 WindowSize > MaxPerWindow 才有意义。
	MaxPerWindow int

	// Weight 当前约束的权重，用于无完美匹配时的加权回退选择。
	Weight float64

	// MultiValueDelimiter 多值分隔符。非空时维度值按此分隔，
	// 两个物品在某维度上存在任意交集即视为"同组"。
	MultiValueDelimiter string
}

// Diversity 多样性 ReRank 节点，支持三种模式：
//
// 模式 1（简单）：设置 LabelKey → 按类别去重（保留首个出现的类别）
// 模式 2（简单）：设置 DiversityKeys → 多 key 滑动窗口打散
// 模式 3（高级）：设置 Constraints → 多规则独立约束 + 权重回退 + 多值维度
//
// Constraints 非空时优先使用模式 3；否则按 LabelKey / DiversityKeys 走旧逻辑。
type Diversity struct {
	// --- 简单模式字段（模式 1 / 2，向后兼容）---

	// LabelKey 类别去重的字段 key，默认 "category"
	LabelKey string
	// DiversityKeys 多 key 打散
	DiversityKeys []string
	// MaxConsecutive 允许同一 key 值连续出现的最大次数，默认 1
	MaxConsecutive int
	// WindowSize 滑动窗口大小，默认 MaxConsecutive+1
	WindowSize int

	// --- 高级模式字段（模式 3）---

	// Constraints 多规则约束；非空时启用高级模式。
	Constraints []DiversityConstraint
	// ExcludeChannels 召回通道名列表；匹配的物品不参与多样性重排，最后追加。
	ExcludeChannels []string
	// ChannelLabelKey 用于识别召回通道的标签 key，默认 "recall_source"。
	ChannelLabelKey string
	// Limit 只对前 N 个位置做多样性；0 = 全部。
	Limit int
	// ExploreLimit 每个位置最多扫描候选数；0 = 不限。
	ExploreLimit int
}

func (n *Diversity) Name() string {
	return "rerank.diversity"
}

func (n *Diversity) Kind() pipeline.Kind {
	return pipeline.KindReRank
}

func (n *Diversity) getValue(item *core.Item, key string) string {
	if item == nil {
		return ""
	}
	v, _ := item.GetValue(key)
	return v
}

func (n *Diversity) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if len(items) == 0 {
		return items, nil
	}

	if len(n.Constraints) > 0 {
		return n.processConstraintDiversity(items)
	}

	result := items

	if n.LabelKey != "" {
		var err error
		result, err = n.processCategoryDeduplication(result)
		if err != nil {
			return nil, err
		}
	}

	diversityKeys := n.getDiversityKeys()
	if len(diversityKeys) > 0 {
		var err error
		result, err = n.processMultiKeyDiversity(result, diversityKeys)
		if err != nil {
			return nil, err
		}
	}

	return result, nil
}

// ---------------------------------------------------------------------------
// 高级模式：多规则约束 + 权重回退
// ---------------------------------------------------------------------------

// dimVals 存储物品在某条约束下的维度值集合。
// 单值时只有一个元素；多值时拆分成多个。
type dimVals struct {
	parts [][]string // 每个 Dimension 对应一个 string slice
}

// overlaps 判断两个 dimVals 是否"同组"：每个维度都需有交集（AND）。
func (a dimVals) overlaps(b dimVals) bool {
	for i := range a.parts {
		if !sliceHasOverlap(a.parts[i], b.parts[i]) {
			return false
		}
	}
	return true
}

func sliceHasOverlap(a, b []string) bool {
	for _, va := range a {
		for _, vb := range b {
			if va == vb {
				return true
			}
		}
	}
	return false
}

func (c *DiversityConstraint) extractDimVals(item *core.Item) dimVals {
	dv := dimVals{parts: make([][]string, len(c.Dimensions))}
	for i, dim := range c.Dimensions {
		raw, _ := item.GetValue(dim)
		if c.MultiValueDelimiter != "" && strings.Contains(raw, c.MultiValueDelimiter) {
			segs := strings.Split(raw, c.MultiValueDelimiter)
			trimmed := make([]string, 0, len(segs))
			for _, s := range segs {
				s = strings.TrimSpace(s)
				if s != "" {
					trimmed = append(trimmed, s)
				}
			}
			dv.parts[i] = trimmed
		} else {
			dv.parts[i] = []string{raw}
		}
	}
	return dv
}

// constraintState 跟踪一条约束的已选历史。
type constraintState struct {
	history []dimVals
}

func (s *constraintState) checkConsecutive(cand dimVals, maxCons int) bool {
	if maxCons <= 0 {
		return true
	}
	run := 0
	for i := len(s.history) - 1; i >= 0; i-- {
		if s.history[i].overlaps(cand) {
			run++
		} else {
			break
		}
	}
	return run < maxCons
}

func (s *constraintState) checkWindow(cand dimVals, windowSize, maxPer int) bool {
	if maxPer <= 0 || windowSize <= 0 {
		return true
	}
	start := len(s.history) - windowSize
	if start < 0 {
		start = 0
	}
	count := 0
	for i := start; i < len(s.history); i++ {
		if s.history[i].overlaps(cand) {
			count++
		}
	}
	return count < maxPer
}

func (s *constraintState) append(dv dimVals) {
	s.history = append(s.history, dv)
}

func (n *Diversity) processConstraintDiversity(items []*core.Item) ([]*core.Item, error) {
	// 按 ExcludeChannels 拆分
	var candidates, excluded []*core.Item
	if len(n.ExcludeChannels) > 0 {
		chKey := n.ChannelLabelKey
		if chKey == "" {
			chKey = DefaultRecallSourceLabel
		}
		excludeSet := make(map[string]bool, len(n.ExcludeChannels))
		for _, ch := range n.ExcludeChannels {
			excludeSet[ch] = true
		}
		for _, it := range items {
			if it == nil {
				continue
			}
			ch := PrimaryRecallChannel(it, chKey)
			if excludeSet[ch] {
				excluded = append(excluded, it)
			} else {
				candidates = append(candidates, it)
			}
		}
	} else {
		for _, it := range items {
			if it != nil {
				candidates = append(candidates, it)
			}
		}
	}

	if len(candidates) == 0 {
		return append(candidates, excluded...), nil
	}

	limit := n.Limit
	if limit <= 0 || limit > len(candidates) {
		limit = len(candidates)
	}

	hasWeight := false
	for i := range n.Constraints {
		if n.Constraints[i].Weight > 0 {
			hasWeight = true
			break
		}
	}

	states := make([]constraintState, len(n.Constraints))
	used := make([]bool, len(candidates))
	result := make([]*core.Item, 0, limit)

	// 种子：第一个候选
	result = append(result, candidates[0])
	used[0] = true
	for ci := range n.Constraints {
		states[ci].append(n.Constraints[ci].extractDimVals(candidates[0]))
	}

	for len(result) < limit {
		found := false
		bestIdx := -1
		bestWeight := -1.0
		firstEligible := -1
		scanned := 0

		for i, it := range candidates {
			if used[i] {
				continue
			}
			if firstEligible < 0 {
				firstEligible = i
			}
			if n.ExploreLimit > 0 {
				scanned++
				if scanned > n.ExploreLimit {
					break
				}
			}

			allMatch := true
			totalWeight := 0.0
			for ci := range n.Constraints {
				c := &n.Constraints[ci]
				dv := c.extractDimVals(it)
				consOK := states[ci].checkConsecutive(dv, c.MaxConsecutive)
				winOK := states[ci].checkWindow(dv, c.WindowSize, c.MaxPerWindow)
				if consOK && winOK {
					totalWeight += c.Weight
				} else {
					allMatch = false
				}
			}

			if allMatch {
				result = append(result, it)
				used[i] = true
				for ci := range n.Constraints {
					states[ci].append(n.Constraints[ci].extractDimVals(it))
				}
				found = true
				break
			}
			if hasWeight && totalWeight > bestWeight {
				bestWeight = totalWeight
				bestIdx = i
			}
		}

		if !found {
			pick := bestIdx
			if pick < 0 {
				pick = firstEligible
			}
			if pick < 0 {
				break
			}
			result = append(result, candidates[pick])
			used[pick] = true
			for ci := range n.Constraints {
				states[ci].append(n.Constraints[ci].extractDimVals(candidates[pick]))
			}
		}
	}

	// 追加未选中的候选
	for i, it := range candidates {
		if !used[i] {
			result = append(result, it)
		}
	}
	// 追加被隔离的通道物品
	result = append(result, excluded...)
	return result, nil
}

// ---------------------------------------------------------------------------
// 简单模式：向后兼容的去重 / 打散逻辑
// ---------------------------------------------------------------------------

func (n *Diversity) processCategoryDeduplication(items []*core.Item) ([]*core.Item, error) {
	key := n.LabelKey
	if key == "" {
		key = "category"
	}

	seen := make(map[string]bool, 32)
	out := make([]*core.Item, 0, len(items))

	for _, it := range items {
		if it == nil {
			continue
		}
		cate := n.getValue(it, key)
		if cate == "" {
			out = append(out, it)
			continue
		}
		if seen[cate] {
			continue
		}
		seen[cate] = true
		out = append(out, it)
	}

	return out, nil
}

func (n *Diversity) getDiversityKeys() []string {
	keys := make([]string, 0, len(n.DiversityKeys))
	seen := make(map[string]struct{}, len(n.DiversityKeys))
	for _, k := range n.DiversityKeys {
		if k == "" {
			continue
		}
		if _, ok := seen[k]; ok {
			continue
		}
		seen[k] = struct{}{}
		keys = append(keys, k)
	}
	return keys
}

func (n *Diversity) processMultiKeyDiversity(items []*core.Item, diversityKeys []string) ([]*core.Item, error) {
	maxConsecutive := n.MaxConsecutive
	if maxConsecutive <= 0 {
		maxConsecutive = 1
	}

	windowSize := n.WindowSize
	if windowSize <= 0 {
		windowSize = maxConsecutive + 1
	}

	windows := make(map[string][]string, len(diversityKeys))
	valueCount := make(map[string]map[string]int, len(diversityKeys))
	for _, key := range diversityKeys {
		windows[key] = make([]string, 0, windowSize)
		valueCount[key] = make(map[string]int, 32)
	}

	pending := make([]*core.Item, 0)
	out := make([]*core.Item, 0, len(items))

	canInsert := func(it *core.Item) bool {
		for _, key := range diversityKeys {
			v := n.getValue(it, key)
			if v == "" {
				continue
			}
			if valueCount[key][v] >= maxConsecutive {
				return false
			}
		}
		return true
	}

	applyInsert := func(it *core.Item) {
		out = append(out, it)
		for _, key := range diversityKeys {
			v := n.getValue(it, key)
			if v == "" {
				continue
			}
			windows[key] = append(windows[key], v)
			valueCount[key][v]++
			if len(windows[key]) > windowSize {
				oldest := windows[key][0]
				windows[key] = windows[key][1:]
				valueCount[key][oldest]--
				if valueCount[key][oldest] == 0 {
					delete(valueCount[key], oldest)
				}
			}
		}
	}

	for _, it := range items {
		if it == nil {
			continue
		}
		if !canInsert(it) {
			pending = append(pending, it)
			continue
		}
		applyInsert(it)
	}

	for _, it := range pending {
		if canInsert(it) {
			applyInsert(it)
		} else {
			out = append(out, it)
		}
	}

	return out, nil
}
