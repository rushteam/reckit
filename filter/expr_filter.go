package filter

import (
	"context"
	"fmt"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/dsl"
)

// ExprFilter 基于 CEL/DSL 表达式的通用过滤器。
// 表达式为 true 的物品被过滤；支持 Label DSL 语法。
//
//	&filter.ExprFilter{Expr: `item.score < 0.1`}
//	&filter.ExprFilter{Expr: `label.category == "adult"`}
type ExprFilter struct {
	Expr string
	// Invert 为 true 时语义翻转：表达式为 true 的物品保留，为 false 的过滤。
	Invert bool

	compiled *dsl.CompiledExpr
}

func (f *ExprFilter) Name() string { return "filter.expr" }

func (f *ExprFilter) getCompiled() (*dsl.CompiledExpr, error) {
	if f.compiled != nil {
		return f.compiled, nil
	}
	c, err := dsl.Compile(f.Expr)
	if err != nil {
		return nil, fmt.Errorf("filter.expr: compile %q: %w", f.Expr, err)
	}
	f.compiled = c
	return c, nil
}

func (f *ExprFilter) ShouldFilter(
	_ context.Context,
	rctx *core.RecommendContext,
	item *core.Item,
) (bool, error) {
	if item == nil {
		return true, nil
	}
	if f.Expr == "" {
		return false, nil
	}
	c, err := f.getCompiled()
	if err != nil {
		return false, err
	}
	ok, err := c.Eval(item, rctx)
	if err != nil {
		return false, fmt.Errorf("filter.expr: eval %q for item %q: %w", f.Expr, item.ID, err)
	}
	if f.Invert {
		return !ok, nil
	}
	return ok, nil
}
