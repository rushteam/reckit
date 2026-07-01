package core

import (
	"strconv"

	"github.com/rushteam/reckit/pkg/utils"
)

// Item 是推荐链路中的统一承载结构：特征、分数、元信息、标签、内部状态。
//
// 四个 map 职责分明：
//   - Features: ML 数值特征，供排序模型消费
//   - Meta:     业务属性（title, category 等），供展示与过滤
//   - Labels:   策略标签（带 Source 追踪），对外可解释
//   - State:    pipeline 内部状态，节点间传递，不对外暴露
//
// ID 类型设计：
//   - 使用 string 类型（通用，支持所有 ID 格式）
type Item struct {
	ID       string // 使用 string 类型（通用，支持所有 ID 格式）
	Score    float64
	Features map[string]float64
	Meta     map[string]any
	Labels   map[string]utils.Label
	State    map[string]any // pipeline 内部状态（节点间传递，不参与 Enrich / 序列化）

	// LabelMergeStrategy 自定义 Label 合并策略（可选）
	// 如果为 nil，则使用默认策略
	LabelMergeStrategy utils.LabelMergeStrategy
}

// NewItem 创建一个新的 Item
func NewItem(id string) *Item {
	return &Item{
		ID:       id,
		Score:    0,
		Features: make(map[string]float64),
		Meta:     make(map[string]any),
		Labels:   make(map[string]utils.Label),
	}
}

// GetValue 从 Item 中按优先级查找 key 对应的字符串值。
// 查找顺序：Labels > Meta > Features。
// 用于 Diversity、Filter 等需要统一读取字段的场景，
// 避免各模块重复实现多字段查找逻辑。
func (it *Item) GetValue(key string) (string, bool) {
	if it.Labels != nil {
		if lbl, ok := it.Labels[key]; ok {
			return lbl.Value, true
		}
	}
	if it.Meta != nil {
		if v, ok := it.Meta[key]; ok {
			if s, ok := v.(string); ok {
				return s, true
			}
		}
	}
	if it.Features != nil {
		if v, ok := it.Features[key]; ok {
			if v == float64(int64(v)) {
				return strconv.FormatInt(int64(v), 10), true
			}
			return strconv.FormatFloat(v, 'f', -1, 64), true
		}
	}
	return "", false
}

func (it *Item) GetValueString(key string) string {
	v, _ := it.GetValue(key)
	return v
}

// SetState 写入 pipeline 内部状态（lazy init）。
func (it *Item) SetState(key string, value any) {
	if it.State == nil {
		it.State = make(map[string]any)
	}
	it.State[key] = value
}

// GetState 读取 pipeline 内部状态。
func (it *Item) GetState(key string) (any, bool) {
	if it.State == nil {
		return nil, false
	}
	v, ok := it.State[key]
	return v, ok
}

// PutLabel 写入 Label；若已存在同名 key，则按合并策略合并。
// 如果 Item 设置了 LabelMergeStrategy，则使用自定义策略；否则使用默认策略。
func (it *Item) PutLabel(key string, lbl utils.Label) {
	if it.Labels == nil {
		it.Labels = make(map[string]utils.Label)
	}
	if old, ok := it.Labels[key]; ok {
		var merged utils.Label
		if it.LabelMergeStrategy != nil {
			merged = it.LabelMergeStrategy.Merge(old, lbl)
		} else {
			merged = utils.MergeLabel(old, lbl)
		}
		it.Labels[key] = merged
		return
	}
	it.Labels[key] = lbl
}
