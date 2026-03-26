package postprocess

import (
	"context"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/utils"
)

var labelPadding = utils.Label{Value: "true", Source: "postprocess.padding"}

// TruncateFieldsNode 响应前裁剪 Item 内部字段，减少序列化体积。
// 可选择清空 Features / Meta / Labels，或仅保留指定的 Meta key。
type TruncateFieldsNode struct {
	ClearFeatures bool
	ClearMeta     bool
	ClearLabels   bool
	// KeepMetaKeys 非空时仅保留指定 Meta key（ClearMeta 为 true 时忽略）。
	KeepMetaKeys []string
}

func (n *TruncateFieldsNode) Name() string        { return "postprocess.truncate_fields" }
func (n *TruncateFieldsNode) Kind() pipeline.Kind { return pipeline.KindPostProcess }

func (n *TruncateFieldsNode) Process(
	_ context.Context,
	_ *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	for _, it := range items {
		if it == nil {
			continue
		}
		if n.ClearFeatures {
			it.Features = nil
		}
		if n.ClearMeta {
			it.Meta = nil
		} else if len(n.KeepMetaKeys) > 0 && len(it.Meta) > 0 {
			keep := make(map[string]any, len(n.KeepMetaKeys))
			for _, k := range n.KeepMetaKeys {
				if v, ok := it.Meta[k]; ok {
					keep[k] = v
				}
			}
			it.Meta = keep
		}
		if n.ClearLabels {
			it.Labels = nil
		}
	}
	return items, nil
}
