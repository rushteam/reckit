package recall

import (
	"context"

	"github.com/rushteam/reckit/core"
)

// ExperimentEvaluator 实验分流评估器接口。
// 由业务方实现，对接具体 AB 框架（GrowthBook、Unleash 等）。
type ExperimentEvaluator interface {
	Evaluate(ctx context.Context, rctx *core.RecommendContext, key string) (branch string, ok bool)
}

// ConditionalSource 根据 AB 实验分支选择不同的召回源。
// 当实验未命中或分支未配置时，使用 Default 源（可为 nil 表示空召回）。
type ConditionalSource struct {
	// SourceName 召回源名称。
	SourceName string

	// Evaluator 实验评估器（必需）。
	Evaluator ExperimentEvaluator

	// ExperimentKey 实验键名。
	ExperimentKey string

	// BranchSources 分支 → 召回源映射。
	BranchSources map[string]Source

	// Default 未命中或分支不在 BranchSources 中时的兜底源。
	// nil 表示空召回。
	Default Source
}

func (s *ConditionalSource) Name() string {
	if s.SourceName != "" {
		return s.SourceName
	}
	return "recall.conditional"
}

func (s *ConditionalSource) Recall(ctx context.Context, rctx *core.RecommendContext) ([]*core.Item, error) {
	if s.Evaluator != nil {
		branch, ok := s.Evaluator.Evaluate(ctx, rctx, s.ExperimentKey)
		if ok {
			if src, found := s.BranchSources[branch]; found {
				return src.Recall(ctx, rctx)
			}
		}
	}
	if s.Default != nil {
		return s.Default.Recall(ctx, rctx)
	}
	return nil, nil
}
