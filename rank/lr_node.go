package rank

import (
	"context"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/model"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/utils"
)

// SortStrategy 是排序策略接口，用于自定义物品排序逻辑。
type SortStrategy interface {
	// Sort 对物品列表进行排序（原地排序）
	Sort(items []*core.Item)
}

// ScoreDescSortStrategy 是按分数降序排序（默认策略）。
type ScoreDescSortStrategy struct{}

func (s *ScoreDescSortStrategy) Sort(items []*core.Item) {
	sort.SliceStable(items, func(i, j int) bool {
		if items[i] == nil {
			return false
		}
		if items[j] == nil {
			return true
		}
		return items[i].Score > items[j].Score
	})
}

// ScoreAscSortStrategy 是按分数升序排序。
type ScoreAscSortStrategy struct{}

func (s *ScoreAscSortStrategy) Sort(items []*core.Item) {
	sort.SliceStable(items, func(i, j int) bool {
		if items[i] == nil {
			return false
		}
		if items[j] == nil {
			return true
		}
		return items[i].Score < items[j].Score
	})
}

// MultiFieldSortStrategy 是多字段排序策略。
type MultiFieldSortStrategy struct {
	Fields []SortField
}

type SortField struct {
	Key       string // 特征名或 Label key
	IsLabel   bool   // true 表示从 Label 读取，false 表示从 Features 读取
	Ascending bool   // true 表示升序，false 表示降序
}

func (s *MultiFieldSortStrategy) Sort(items []*core.Item) {
	sort.SliceStable(items, func(i, j int) bool {
		if items[i] == nil {
			return false
		}
		if items[j] == nil {
			return true
		}

		for _, field := range s.Fields {
			var valI, valJ float64
			var okI, okJ bool

			if field.IsLabel {
				if lbl, ok := items[i].Labels[field.Key]; ok {
					valI, okI = parseLabelNumeric(lbl.Value)
				}
				if lbl, ok := items[j].Labels[field.Key]; ok {
					valJ, okJ = parseLabelNumeric(lbl.Value)
				}
			} else {
				valI, okI = items[i].Features[field.Key]
				valJ, okJ = items[j].Features[field.Key]
			}

			if !okI && !okJ {
				continue
			}
			if !okI {
				return false
			}
			if !okJ {
				return true
			}

			if field.Ascending {
				if valI < valJ {
					return true
				}
				if valI > valJ {
					return false
				}
			} else {
				if valI > valJ {
					return true
				}
				if valI < valJ {
					return false
				}
			}
		}
		return false
	})
}

// LRNode 是一个使用 RankModel 的排序 Node 示例（不限定模型类型，LR 只是默认实现之一）。
// - 写入 labels：rank_model
// - 更新 item.Score 并按分数降序排序
type LRNode struct {
	Model        model.RankModel
	SortStrategy SortStrategy // 必需，如果为 nil 则使用默认降序排序
	// Explain 控制调试标签输出；nil 表示不输出额外 explain。
	Explain *LRExplainConfig
}

const (
	LabelKeyRankModel           = "rank_model"
	LabelKeyRankLinearRaw       = "rank_linear_raw"
	LabelKeyRankFeaturesMissing = "rank_features_missing"
	LabelKeyRankFeatureCoverage = "rank_feature_coverage"
)

// LRExplainConfig 控制 LR 排序过程中的解释性标签输出。
type LRExplainConfig struct {
	EmitRawScore        bool
	EmitMissingFlag     bool
	EmitFeatureCoverage bool
}

func (n *LRNode) Name() string        { return "rank.model" }
func (n *LRNode) Kind() pipeline.Kind { return pipeline.KindRank }

func (n *LRNode) Process(
	_ context.Context,
	_ *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if n.Model == nil || len(items) == 0 {
		return items, nil
	}

	for _, it := range items {
		if it == nil {
			continue
		}
		score, err := n.Model.Predict(it.Features)
		if err != nil {
			return nil, err
		}
		it.Score = score
		it.PutLabel(LabelKeyRankModel, utils.Label{Value: n.Model.Name(), Source: "rank"})
		n.emitExplainLabels(it)
	}

	// 使用局部变量兜底，避免并发请求中写共享字段导致 data race。
	strategy := n.SortStrategy
	if strategy == nil {
		strategy = &ScoreDescSortStrategy{} // 默认降序
	}
	strategy.Sort(items)

	return items, nil
}

func (n *LRNode) emitExplainLabels(it *core.Item) {
	if n == nil || n.Explain == nil || it == nil {
		return
	}
	features := it.Features
	if features == nil {
		features = map[string]float64{}
	}

	if n.Explain.EmitRawScore {
		raw := lrLinearRaw(n.Model, features, it.Score)
		it.PutLabel(LabelKeyRankLinearRaw, utils.Label{
			Value:  strconv.FormatFloat(raw, 'f', 6, 64),
			Source: "rank",
		})
	}
	if n.Explain.EmitMissingFlag {
		missing := "0"
		if v, ok := features["item_features_missing"]; ok && v > 0 {
			missing = "1"
		}
		it.PutLabel(LabelKeyRankFeaturesMissing, utils.Label{Value: missing, Source: "rank"})
	}
	if n.Explain.EmitFeatureCoverage {
		total, present := 0, 0
		switch m := n.Model.(type) {
		case *model.LRModel:
			total = len(m.Weights)
			for k := range m.Weights {
				if _, ok := features[k]; ok {
					present++
				}
			}
		default:
			return
		}
		if total > 0 {
			coverage := float64(present) / float64(total)
			it.PutLabel(LabelKeyRankFeatureCoverage, utils.Label{
				Value:  strconv.FormatFloat(coverage, 'f', 4, 64),
				Source: "rank",
			})
		}
	}
}

// parseLabelNumeric 尝试将 Label.Value 解析为 float64。
// 对于合并后的多值标签（如 "hot|cf"），取第一个可解析的段。
func parseLabelNumeric(s string) (float64, bool) {
	if v, err := strconv.ParseFloat(s, 64); err == nil {
		return v, true
	}
	if idx := strings.IndexByte(s, '|'); idx >= 0 {
		if v, err := strconv.ParseFloat(s[:idx], 64); err == nil {
			return v, true
		}
	}
	return 0, false
}

func lrLinearRaw(rm model.RankModel, features map[string]float64, score float64) float64 {
	if m, ok := rm.(*model.LRModel); ok {
		raw := m.Bias
		for k, w := range m.Weights {
			raw += w * features[k]
		}
		return raw
	}
	// 未知模型时尽量从 score 反推 logit，边界值返回 0。
	if score <= 0 || score >= 1 {
		return 0
	}
	return math.Log(score / (1 - score))
}
