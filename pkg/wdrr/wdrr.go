package wdrr

import "math"

// Channel 表示 WDRR 调度器中的一个加权来源。
type Channel[T any] struct {
	Key       string
	Quota     int
	Items     []T
	cursor    int
	Picked    int
	exhausted bool
}

// PlacementFilter 决定一个候选是否可以放置在当前位置。
type PlacementFilter[T any] interface {
	CanPlace(item T) bool
	Place(item T)
}

// Schedule 使用 Weighted Deficit Round Robin 在多个 channel 间按配额比例分配。
// 当某 channel 候选耗尽时自动退出，其份额按比例流给剩余活跃 channel。
func Schedule[T any](channels []*Channel[T], filter PlacementFilter[T], topN int) []T {
	totalQuota := 0
	for _, ch := range channels {
		totalQuota += ch.Quota
	}
	if totalQuota == 0 {
		return nil
	}

	result := make([]T, 0, topN)
	for len(result) < topN {
		bestIdx := -1
		bestDeficit := math.Inf(-1)
		slot := float64(len(result) + 1)

		for i, ch := range channels {
			if ch.exhausted {
				continue
			}
			target := float64(ch.Quota) / float64(totalQuota) * slot
			deficit := target - float64(ch.Picked)
			if deficit > bestDeficit {
				bestDeficit = deficit
				bestIdx = i
			}
		}

		if bestIdx < 0 {
			break
		}

		item, ok := channels[bestIdx].tryPick(filter)
		if !ok {
			channels[bestIdx].exhausted = true
			continue
		}

		result = append(result, item)
		filter.Place(item)
		channels[bestIdx].Picked++
	}
	return result
}

func (ch *Channel[T]) tryPick(filter PlacementFilter[T]) (T, bool) {
	for ch.cursor < len(ch.Items) {
		item := ch.Items[ch.cursor]
		ch.cursor++
		if filter.CanPlace(item) {
			return item, true
		}
	}
	var zero T
	return zero, false
}
