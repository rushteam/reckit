package rerank

import (
	"context"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

// Diversity 是一个多样性 ReRank 节点，支持两种规则，可以单独或组合使用：
// 1. 按类别去重（保留首个出现的类别）
// 2. 按一个或多个 key 打散（避免连续出现相同属性）
//
// 使用方式：
// - 只设置 LabelKey：仅按类别去重
// - 只设置 DiversityKeys：按一个或多个 key 打散
// - 设置 DiversityKeys：按多个 key 同时打散（例如 author + category）
// - 同时设置 LabelKey 和 DiversityKeys：先按类别去重，再按多 key 打散
//
// 类别/作者来源优先级：
// - label[key].Value
// - meta[key] (string)
type Diversity struct {
	// LabelKey 用于类别去重模式的字段 key，默认 "category"
	// 如果设置了 LabelKey，会先执行类别去重
	LabelKey string

	// DiversityKeys 用于多 key 打散，支持同时按多个字段约束连续出现次数。
	// 例如：[]string{"author", "category"} 表示同时约束作者与分类。
	DiversityKeys []string

	// MaxConsecutive 允许同一 key 值连续出现的最大次数，默认 1（不能连续出现）
	// 例如：MaxConsecutive=1 表示不能连续出现，MaxConsecutive=2 表示最多连续出现2次
	MaxConsecutive int

	// WindowSize 滑动窗口大小，用于检查最近出现的值，默认等于 MaxConsecutive+1
	// 如果为 0，则自动设置为 MaxConsecutive+1
	WindowSize int
}

func (n *Diversity) Name() string {
	return "rerank.diversity"
}

func (n *Diversity) Kind() pipeline.Kind {
	return pipeline.KindReRank
}

// getValue 委托给 Item.GetValue，统一按 Labels > Meta > Features 查找。
func (n *Diversity) getValue(item *core.Item, key string) string {
	if item == nil {
		return ""
	}
	v, _ := item.GetValue(key)
	return v
}

// Process 处理 items，根据配置执行类别去重和/或多 key 打散。
func (n *Diversity) Process(
	_ context.Context,
	_ *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if len(items) == 0 {
		return items, nil
	}

	result := items

	// 如果设置了 LabelKey，先执行类别去重
	if n.LabelKey != "" {
		var err error
		result, err = n.processCategoryDeduplication(result)
		if err != nil {
			return nil, err
		}
	}

	// 如果设置了打散 key，再执行多 key 打散
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

// processCategoryDeduplication 按类别去重（保留首个出现的类别）
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

// processMultiKeyDiversity 按多个 key 打散，避免连续出现同值。
func (n *Diversity) processMultiKeyDiversity(items []*core.Item, diversityKeys []string) ([]*core.Item, error) {
	maxConsecutive := n.MaxConsecutive
	if maxConsecutive <= 0 {
		maxConsecutive = 1 // 默认不能连续出现
	}

	windowSize := n.WindowSize
	if windowSize <= 0 {
		windowSize = maxConsecutive + 1 // 默认窗口大小为 maxConsecutive+1
	}

	// 每个 key 各维护一套滑动窗口与计数器。
	windows := make(map[string][]string, len(diversityKeys))
	valueCount := make(map[string]map[string]int, len(diversityKeys))
	for _, key := range diversityKeys {
		windows[key] = make([]string, 0, windowSize)
		valueCount[key] = make(map[string]int, 32)
	}

	// 待处理的 items（因为冲突而延迟插入的）
	pending := make([]*core.Item, 0)
	// 最终结果
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

	// 处理所有 items
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

	// 处理待处理的 items（尝试插入剩余位置）
	for _, it := range pending {
		if canInsert(it) {
			applyInsert(it)
		}
		// 如果仍然无法插入，则丢弃该 item（避免无限循环）
	}

	return out, nil
}
