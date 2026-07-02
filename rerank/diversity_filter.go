package rerank

import (
	"strings"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/wdrr"
)

// compile-time check
var _ wdrr.PlacementFilter[*core.Item] = (*DiversityPlacementFilter)(nil)

// DiversityPlacementFilter 实现 wdrr.PlacementFilter[*core.Item]，
// 基于 DiversityConstraint 判断候选 item 是否可放置在当前位置。
//
// 与 Diversity Node 使用相同的 DiversityConstraint 定义，
// 但工作在 WDRR 调度器的逐 slot 放置阶段（CanPlace/Place），
// 而非 Diversity Node 的全量重排。
type DiversityPlacementFilter struct {
	constraints []DiversityConstraint
	histories   [][]dimKey
}

type dimKey struct {
	parts [][]string
}

// NewDiversityPlacementFilter 创建多样性放置过滤器。
// constraints 为空时 CanPlace 始终返回 true。
func NewDiversityPlacementFilter(constraints []DiversityConstraint) *DiversityPlacementFilter {
	return &DiversityPlacementFilter{
		constraints: constraints,
		histories:   make([][]dimKey, len(constraints)),
	}
}

func (f *DiversityPlacementFilter) CanPlace(item *core.Item) bool {
	if len(f.constraints) == 0 {
		return true
	}
	for ci, c := range f.constraints {
		dk := extractDimKey(item, c)
		if c.MaxConsecutive > 0 {
			run := 0
			for i := len(f.histories[ci]) - 1; i >= 0; i-- {
				if f.histories[ci][i].overlaps(dk) {
					run++
				} else {
					break
				}
			}
			if run >= c.MaxConsecutive {
				return false
			}
		}
		if c.MaxPerWindow > 0 && c.WindowSize > 0 {
			start := len(f.histories[ci]) - c.WindowSize
			if start < 0 {
				start = 0
			}
			count := 0
			for i := start; i < len(f.histories[ci]); i++ {
				if f.histories[ci][i].overlaps(dk) {
					count++
				}
			}
			if count >= c.MaxPerWindow {
				return false
			}
		}
	}
	return true
}

func (f *DiversityPlacementFilter) Place(item *core.Item) {
	for ci, c := range f.constraints {
		f.histories[ci] = append(f.histories[ci], extractDimKey(item, c))
	}
}

func extractDimKey(item *core.Item, c DiversityConstraint) dimKey {
	dk := dimKey{parts: make([][]string, len(c.Dimensions))}
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
			dk.parts[i] = trimmed
		} else {
			dk.parts[i] = []string{raw}
		}
	}
	return dk
}

func (a dimKey) overlaps(b dimKey) bool {
	for i := range a.parts {
		if !sliceOverlap(a.parts[i], b.parts[i]) {
			return false
		}
	}
	return true
}

func sliceOverlap(a, b []string) bool {
	for _, va := range a {
		for _, vb := range b {
			if va == vb {
				return true
			}
		}
	}
	return false
}
