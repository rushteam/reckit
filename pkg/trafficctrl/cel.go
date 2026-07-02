package trafficctrl

import (
	"fmt"
	"log/slog"
	"sync"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

var celEngine = &celEvaluator{
	cache: make(map[string]cel.Program),
}

type celEvaluator struct {
	mu    sync.RWMutex
	env   *cel.Env
	cache map[string]cel.Program
}

func (e *celEvaluator) getEnv() (*cel.Env, error) {
	e.mu.RLock()
	if e.env != nil {
		env := e.env
		e.mu.RUnlock()
		return env, nil
	}
	e.mu.RUnlock()

	e.mu.Lock()
	defer e.mu.Unlock()
	if e.env != nil {
		return e.env, nil
	}

	env, err := cel.NewEnv(
		cel.Variable("attrs", cel.MapType(cel.StringType, cel.DynType)),
	)
	if err != nil {
		return nil, fmt.Errorf("cel: create env: %w", err)
	}
	e.env = env
	return env, nil
}

func (e *celEvaluator) getProgram(expr string) (cel.Program, error) {
	e.mu.RLock()
	if prog, ok := e.cache[expr]; ok {
		e.mu.RUnlock()
		return prog, nil
	}
	e.mu.RUnlock()

	env, err := e.getEnv()
	if err != nil {
		return nil, err
	}

	ast, issues := env.Compile(expr)
	if issues != nil && issues.Err() != nil {
		return nil, fmt.Errorf("cel: compile %q: %w", expr, issues.Err())
	}

	prog, err := env.Program(ast)
	if err != nil {
		return nil, fmt.Errorf("cel: program %q: %w", expr, err)
	}

	e.mu.Lock()
	e.cache[expr] = prog
	e.mu.Unlock()
	return prog, nil
}

// EvalCEL 对属性 map 执行 CEL 表达式，返回布尔结果。
// 表达式中用 attrs["field"] 或 attrs.field 引用属性。
func EvalCEL(expression string, attrs map[string]any) bool {
	prog, err := celEngine.getProgram(expression)
	if err != nil {
		slog.Warn("trafficctrl: cel compile failed",
			slog.String("expression", expression),
			slog.String("error", err.Error()))
		return false
	}

	out, _, err := prog.Eval(map[string]any{"attrs": attrs})
	if err != nil {
		slog.Warn("trafficctrl: cel eval failed",
			slog.String("expression", expression),
			slog.String("error", err.Error()))
		return false
	}

	return isTruthy(out)
}

func isTruthy(v ref.Val) bool {
	if v == nil {
		return false
	}
	if v.Type() == types.BoolType {
		b, ok := v.Value().(bool)
		return ok && b
	}
	return false
}
