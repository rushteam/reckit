package rerank

import (
	"context"
	"sort"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/wdrr"
)

// ChannelQuotaSlot 定义单个渠道的配额。
type ChannelQuotaSlot struct {
	Key   string // 渠道标识（recall_source / pool_key / 自定义）
	Quota int    // 目标配额
}

// ChannelQuotaMixNode 基于 WDRR（Weighted Deficit Round Robin）的渠道配额混排节点。
//
// 将输入 items 按 ChannelKeyFunc 分到各渠道，使用 WDRR 算法按配额比例交替选取。
// 空渠道自动退出，剩余活跃渠道按比例吸收其份额。
//
// 可选 DiversityConstraints：在 WDRR 逐 slot 放置时检查多样性约束，
// 不满足的候选跳过（cursor 前移），该渠道试下一个候选。
type ChannelQuotaMixNode struct {
	// Channels 渠道配额配置（必需）。
	Channels []ChannelQuotaSlot

	// TopN 输出总数（默认 10）。
	TopN int

	// CandidateMultiplier 每个渠道候选数 = Quota × CandidateMultiplier，
	// 用于给 Diversity 留出跳过余地。默认 5。
	CandidateMultiplier int

	// DiversityConstraints 可选多样性约束。
	DiversityConstraints []DiversityConstraint

	// ChannelKeyFunc 从 item 提取渠道 key。
	// nil 时默认读取 Labels["recall_source"].Value。
	ChannelKeyFunc func(item *core.Item) string

	// PassthroughUnmatched 未匹配任何渠道的 items 追加到结果末尾。
	// 默认 false（丢弃未匹配的 items）。
	PassthroughUnmatched bool
}

func (n *ChannelQuotaMixNode) Name() string        { return "rerank.channel_quota_mix" }
func (n *ChannelQuotaMixNode) Kind() pipeline.Kind { return pipeline.KindReRank }

func (n *ChannelQuotaMixNode) Process(ctx context.Context, rctx *core.RecommendContext, items []*core.Item) ([]*core.Item, error) {
	if n == nil || len(items) == 0 || len(n.Channels) == 0 {
		return items, nil
	}

	topN := n.TopN
	if topN <= 0 {
		topN = 10
	}
	mult := n.CandidateMultiplier
	if mult <= 0 {
		mult = 5
	}

	channelKeyFn := n.ChannelKeyFunc
	if channelKeyFn == nil {
		channelKeyFn = defaultChannelKey
	}

	groups := make(map[string][]*core.Item)
	var unmatched []*core.Item
	quotaSet := make(map[string]struct{}, len(n.Channels))
	for _, ch := range n.Channels {
		quotaSet[ch.Key] = struct{}{}
	}

	for _, it := range items {
		if it == nil {
			continue
		}
		key := channelKeyFn(it)
		if _, ok := quotaSet[key]; ok {
			groups[key] = append(groups[key], it)
		} else {
			unmatched = append(unmatched, it)
		}
	}

	// 各组按 Score 降序
	for _, g := range groups {
		sort.SliceStable(g, func(i, j int) bool {
			return g[i].Score > g[j].Score
		})
	}

	channels := make([]*wdrr.Channel[*core.Item], 0, len(n.Channels))
	for _, slot := range n.Channels {
		g := groups[slot.Key]
		cap := slot.Quota * mult
		if cap > len(g) {
			cap = len(g)
		}
		channels = append(channels, &wdrr.Channel[*core.Item]{
			Key:   slot.Key,
			Quota: slot.Quota,
			Items: g[:cap],
		})
	}

	var filter wdrr.PlacementFilter[*core.Item]
	if len(n.DiversityConstraints) > 0 {
		filter = NewDiversityPlacementFilter(n.DiversityConstraints)
	} else {
		filter = &noopFilter{}
	}

	result := wdrr.Schedule(channels, filter, topN)

	if n.PassthroughUnmatched && len(unmatched) > 0 {
		result = append(result, unmatched...)
	}

	return result, nil
}

func defaultChannelKey(item *core.Item) string {
	if item.Labels == nil {
		return ""
	}
	if lbl, ok := item.Labels["recall_source"]; ok {
		return lbl.Value
	}
	return ""
}

type noopFilter struct{}

func (f *noopFilter) CanPlace(_ *core.Item) bool { return true }
func (f *noopFilter) Place(_ *core.Item)          {}
