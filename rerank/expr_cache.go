package rerank

import (
	"sync"

	"github.com/rushteam/reckit/pkg/dsl"
)

// exprCache is a thread-safe lazy compile cache for DSL expressions.
// It uses a pointer receiver so the struct containing it can be safely copied
// (e.g. via slice append) before first use.
type exprCache struct {
	once sync.Once
	val  *dsl.CompiledExpr
	err  error
}

func (c *exprCache) get(expr string) (*dsl.CompiledExpr, error) {
	c.once.Do(func() {
		c.val, c.err = dsl.Compile(expr)
	})
	return c.val, c.err
}
