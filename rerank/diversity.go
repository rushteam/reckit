package rerank

import (
	"context"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

// Diversity 是一个多样性 ReRank 节点，支持两种规则，可以单独或组合使用：
// 1. 按类别去重（保留首个出现的类别）
// 2. 按作者（用户uid）打散（避免连续出现同一作者）
//
// 使用方式：
// - 只设置 LabelKey：仅按类别去重
// - 只设置 AuthorKey：仅按作者打散
// - 同时设置 LabelKey 和 AuthorKey：先按类别去重，再按作者打散
//
// 类别/作者来源优先级：
// - label[key].Value
// - meta[key] (string)
type Diversity struct {
	// LabelKey 用于类别去重模式的字段 key，默认 "category"
	// 如果设置了 LabelKey，会先执行类别去重
	LabelKey string

	// AuthorKey 用于作者打散模式的字段 key，默认 "author"
	// 如果设置了 AuthorKey，会在类别去重后（如果有）执行作者打散
	AuthorKey string

	// MaxConsecutive 允许同一作者连续出现的最大次数，默认 1（不能连续出现）
	// 例如：MaxConsecutive=1 表示不能连续出现，MaxConsecutive=2 表示最多连续出现2次
	MaxConsecutive int

	// WindowSize 滑动窗口大小，用于检查最近出现的作者，默认等于 MaxConsecutive+1
	// 如果为 0，则自动设置为 MaxConsecutive+1
	WindowSize int
}

func (n *Diversity) Name() string {
	return "rerank.diversity"
}

func (n *Diversity) Kind() pipeline.Kind {
	return pipeline.KindReRank
}

// getValue 从 Item 中获取指定 key 的值，优先级：Labels > Meta
func (n *Diversity) getValue(item *core.Item, key string) string {
	if item == nil {
		return ""
	}

	// 优先从 Labels 获取
	if item.Labels != nil {
		if lbl, ok := item.Labels[key]; ok {
			return lbl.Value
		}
	}

	// 从 Meta 获取
	if item.Meta != nil {
		if v, ok := item.Meta[key]; ok {
			if s, ok := v.(string); ok {
				return s
			}
		}
	}

	return ""
}

// Process 处理 items，根据配置执行类别去重和/或作者打散
// 如果同时设置了 LabelKey 和 AuthorKey，先执行类别去重，再执行作者打散
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

	// 如果设置了 AuthorKey，再执行作者打散
	if n.AuthorKey != "" {
		var err error
		result, err = n.processAuthorDiversity(result)
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

// processAuthorDiversity 按作者打散，避免连续出现同一作者
func (n *Diversity) processAuthorDiversity(items []*core.Item) ([]*core.Item, error) {
	authorKey := n.AuthorKey
	if authorKey == "" {
		authorKey = "author"
	}

	maxConsecutive := n.MaxConsecutive
	if maxConsecutive <= 0 {
		maxConsecutive = 1 // 默认不能连续出现
	}

	windowSize := n.WindowSize
	if windowSize <= 0 {
		windowSize = maxConsecutive + 1 // 默认窗口大小为 maxConsecutive+1
	}

	// 滑动窗口：记录最近出现的作者
	window := make([]string, 0, windowSize)
	// 当前窗口中每个作者出现的次数
	authorCount := make(map[string]int, 32)
	// 待处理的 items（因为作者冲突而延迟插入的）
	pending := make([]*core.Item, 0)
	// 最终结果
	out := make([]*core.Item, 0, len(items))

	// 处理所有 items
	for _, it := range items {
		if it == nil {
			continue
		}

		author := n.getValue(it, authorKey)
		
		// 如果没有作者信息，直接通过
		if author == "" {
			out = append(out, it)
			// 清空窗口，因为插入了无作者信息的 item
			window = window[:0]
			for k := range authorCount {
				delete(authorCount, k)
			}
			continue
		}

		// 检查当前作者在窗口中的出现次数
		count := authorCount[author]
		
		// 如果当前作者在窗口中的出现次数已达到上限，需要延迟插入
		if count >= maxConsecutive {
			pending = append(pending, it)
			continue
		}

		// 可以插入，添加到结果中
		out = append(out, it)
		
		// 更新窗口
		window = append(window, author)
		authorCount[author]++
		
		// 如果窗口超过大小，移除最旧的元素
		if len(window) > windowSize {
			oldest := window[0]
			window = window[1:]
			authorCount[oldest]--
			if authorCount[oldest] == 0 {
				delete(authorCount, oldest)
			}
		}
	}

	// 处理待处理的 items（尝试插入剩余位置）
	for _, it := range pending {
		author := n.getValue(it, authorKey)
		if author == "" {
			out = append(out, it)
			continue
		}

		// 检查是否可以插入
		count := authorCount[author]
		if count < maxConsecutive {
			out = append(out, it)
			window = append(window, author)
			authorCount[author]++
			
			// 更新窗口
			if len(window) > windowSize {
				oldest := window[0]
				window = window[1:]
				authorCount[oldest]--
				if authorCount[oldest] == 0 {
					delete(authorCount, oldest)
				}
			}
		}
		// 如果仍然无法插入，则丢弃该 item（避免无限循环）
	}

	return out, nil
}
