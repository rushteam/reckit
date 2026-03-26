package rerank

import (
	"context"
	"fmt"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/filter"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/dsl"
)

// ScoreAdjustMode 单条规则对 Score 的调整方式。
type ScoreAdjustMode string

const (
	ScoreAdjustAdd ScoreAdjustMode = "add"
	ScoreAdjustMul ScoreAdjustMode = "mul"
	ScoreAdjustSet ScoreAdjustMode = "set"
)

// ScoreAdjustValueFunc 可选：从上下文/Item 计算调整量（避免脚本引擎依赖）。
type ScoreAdjustValueFunc func(ctx context.Context, rctx *core.RecommendContext, item *core.Item) (float64, error)

// ScoreAdjustRule 单条规则：先判定是否匹配，再按 Mode 调整分数。
// 匹配条件：若配置 Filter，则 !ShouldFilter 表示命中；若配置 Expr（CEL），须为 true；
// 两者同时配置时为 AND。Expr 使用与 pkg/dsl 一致的 Label DSL（CEL）。
type ScoreAdjustRule struct {
	Filter    filter.Filter
	Expr      string
	Mode      ScoreAdjustMode
	Value     float64
	ValueFunc ScoreAdjustValueFunc

	compiledExpr *dsl.CompiledExpr // 懒编译缓存
}

// ScoreAdjust 按规则顺序对候选提权/改分；典型插入点为精排之后、截断之前。
//
// MatchAllRules 为 false（默认）：每条 item 从首条规则开始，命中一条即停。
// 为 true：对每条 item 应用所有匹配规则的叠加调整。
type ScoreAdjust struct {
	Rules         []ScoreAdjustRule
	MatchAllRules bool

	validated bool // 首次 Process 时校验 Rules 合法性
}

func (n *ScoreAdjust) Name() string {
	return "rerank.score_adjust"
}

func (n *ScoreAdjust) Kind() pipeline.Kind {
	return pipeline.KindReRank
}

func (n *ScoreAdjust) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if !n.validated {
		for ri := range n.Rules {
			r := &n.Rules[ri]
			if r.Filter == nil && r.Expr == "" {
				return nil, fmt.Errorf("rerank.score_adjust: Rules[%d] must set Filter and/or Expr", ri)
			}
		}
		n.validated = true
	}
	for _, item := range items {
		if item == nil {
			continue
		}
		for ri := range n.Rules {
			ok, err := n.ruleMatches(ctx, rctx, &n.Rules[ri], item)
			if err != nil {
				return nil, err
			}
			if !ok {
				continue
			}
			if err := n.apply(ctx, rctx, &n.Rules[ri], item); err != nil {
				return nil, err
			}
			if !n.MatchAllRules {
				break
			}
		}
	}
	return items, nil
}

func (r *ScoreAdjustRule) getCompiledExpr() (*dsl.CompiledExpr, error) {
	if r.compiledExpr != nil {
		return r.compiledExpr, nil
	}
	c, err := dsl.Compile(r.Expr)
	if err != nil {
		return nil, err
	}
	r.compiledExpr = c
	return c, nil
}

func (n *ScoreAdjust) ruleMatches(
	ctx context.Context,
	rctx *core.RecommendContext,
	rule *ScoreAdjustRule,
	item *core.Item,
) (bool, error) {
	if rule.Filter != nil {
		skip, err := rule.Filter.ShouldFilter(ctx, rctx, item)
		if err != nil {
			return false, err
		}
		if skip {
			return false, nil
		}
	}
	if rule.Expr != "" {
		c, err := rule.getCompiledExpr()
		if err != nil {
			return false, err
		}
		ok, err := c.Eval(item, rctx)
		if err != nil {
			return false, err
		}
		if !ok {
			return false, nil
		}
	}
	return true, nil
}

func (n *ScoreAdjust) apply(
	ctx context.Context,
	rctx *core.RecommendContext,
	rule *ScoreAdjustRule,
	item *core.Item,
) error {
	var v float64
	if rule.ValueFunc != nil {
		var err error
		v, err = rule.ValueFunc(ctx, rctx, item)
		if err != nil {
			return err
		}
	} else {
		v = rule.Value
	}
	mode := rule.Mode
	if mode == "" {
		mode = ScoreAdjustAdd
	}
	switch mode {
	case ScoreAdjustAdd:
		item.Score += v
	case ScoreAdjustMul:
		item.Score *= v
	case ScoreAdjustSet:
		item.Score = v
	default:
		return fmt.Errorf("rerank.score_adjust: unknown mode %q", mode)
	}
	return nil
}

// ScoreWeightProvider 按 item ID 提供外部权重乘子（业务可查 Redis/DB 等）。
type ScoreWeightProvider interface {
	Weights(ctx context.Context, rctx *core.RecommendContext, items []*core.Item) (map[string]float64, error)
}

// ScoreWeightApplyMode 将权重应用到 Score 的方式。
type ScoreWeightApplyMode string

const (
	// ScoreWeightMul item.Score *= w（默认）
	ScoreWeightMul ScoreWeightApplyMode = "mul"
	// ScoreWeightAdd item.Score += w
	ScoreWeightAdd ScoreWeightApplyMode = "add"
)

// ScoreWeightBoost 批量拉取 ID→权重，对缺失 ID 视为乘子 1.0（add 模式视为 +0）。
type ScoreWeightBoost struct {
	Provider ScoreWeightProvider
	Mode     ScoreWeightApplyMode
}

func (n *ScoreWeightBoost) Name() string {
	return "rerank.score_weight"
}

func (n *ScoreWeightBoost) Kind() pipeline.Kind {
	return pipeline.KindReRank
}

func (n *ScoreWeightBoost) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if n.Provider == nil {
		return items, nil
	}
	wmap, err := n.Provider.Weights(ctx, rctx, items)
	if err != nil {
		return nil, err
	}
	mode := n.Mode
	if mode == "" {
		mode = ScoreWeightMul
	}
	switch mode {
	case ScoreWeightMul, ScoreWeightAdd:
	default:
		return nil, fmt.Errorf("rerank.score_weight: unknown mode %q", mode)
	}
	for _, item := range items {
		if item == nil {
			continue
		}
		w, ok := wmap[item.ID]
		if !ok {
			continue
		}
		switch mode {
		case ScoreWeightAdd:
			item.Score += w
		case ScoreWeightMul:
			item.Score *= w
		}
	}
	return items, nil
}
