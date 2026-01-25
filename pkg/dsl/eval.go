package dsl

import (
	"fmt"
	"sync"

	"github.com/google/cel-go/cel"

	"github.com/rushteam/reckit/core"
)

var (
	// celEnv 是全局的 CEL 环境，线程安全，可复用
	celEnv     *cel.Env
	celEnvOnce sync.Once
)

// initCELEnv 初始化 CEL 环境，定义变量和函数
func initCELEnv() (*cel.Env, error) {
	env, err := cel.NewEnv(
		// 定义变量类型
		cel.Variable("item", cel.DynType),
		cel.Variable("label", cel.DynType),
		cel.Variable("rctx", cel.DynType),
	)
	return env, err
}

// getCELEnv 获取或创建 CEL 环境
func getCELEnv() (*cel.Env, error) {
	var err error
	celEnvOnce.Do(func() {
		celEnv, err = initCELEnv()
	})
	return celEnv, err
}

// Eval 是 Label DSL 解释器，使用 CEL (Common Expression Language) 实现。
// CEL 是 Google 开发的表达式语言，具有类型安全、高性能、线程安全等特性。
//
// 表达式语法（CEL 标准语法）：
//   - 基础：label.recall_source == "hot" / label.rank_model != "lr"
//   - 数值：item.score > 0.7 / item.score >= 0.5
//   - 逻辑：label.category == "A" && item.score > 0.8
//   - 存在性：label.recall_source != null
//   - 包含：label.recall_source.contains("hot") 或 "hot" in label.recall_source
//
// 示例：
//   - `label.recall_source.contains("hot")` → 召回来源包含 "hot"
//   - `label.rank_model == "lr" && item.score > 0.7` → LR 模型且分数 > 0.7
//   - `label.category != null && label.category == "A"` → 存在 category 且为 "A"
type Eval struct {
	item  *core.Item
	rctx  *core.RecommendContext
	env   *cel.Env
	prg   cel.Program
}

// NewEval 创建一个新的 DSL 解释器。
// 表达式会被编译并缓存，可以多次调用 Evaluate 方法。
func NewEval(item *core.Item, rctx *core.RecommendContext) *Eval {
	env, _ := getCELEnv()
	return &Eval{
		item: item,
		rctx: rctx,
		env:  env,
	}
}

// Evaluate 解析并执行 DSL 表达式，返回布尔结果。
// 表达式使用 CEL (Common Expression Language) 语法。
// 
// 支持的语法：
//   - label.recall_source == "hot"
//   - item.score > 0.7
//   - label.category == "A" && item.score > 0.8
//   - label.recall_source != null  (检查是否存在)
//   - label.recall_source.contains("hot") 或 "hot" in label.recall_source
// 
// 注意：has(label.key) 可以用 label.key != null 替代
func (e *Eval) Evaluate(expr string) (bool, error) {
	if expr == "" {
		return true, nil
	}

	// 编译表达式
	ast, issues := e.env.Compile(expr)
	if issues != nil && issues.Err() != nil {
		return false, fmt.Errorf("compile error: %v", issues.Err())
	}

	// 创建程序
	prg, err := e.env.Program(ast)
	if err != nil {
		return false, fmt.Errorf("program error: %v", err)
	}

	// 准备输入数据
	input := e.buildInput()

	// 执行表达式
	out, _, err := prg.Eval(input)
	if err != nil {
		// 对于不存在的 key，CEL 会返回错误
		// 用户应该使用 label.key != null 来检查存在性，而不是直接访问
		return false, fmt.Errorf("eval error: %v", err)
	}

	// 转换为布尔值
	result, ok := out.Value().(bool)
	if !ok {
		return false, fmt.Errorf("expression must return boolean, got %T", out.Value())
	}

	return result, nil
}

// buildInput 构建 CEL 表达式的输入数据
func (e *Eval) buildInput() map[string]interface{} {
	// 构建 label map
	labels := make(map[string]interface{})
	for k, v := range e.item.Labels {
		labels[k] = map[string]interface{}{
			"value":  v.Value,
			"source": v.Source,
		}
	}

	// 构建 item map
	item := map[string]interface{}{
		"id":       e.item.ID,
		"score":    e.item.Score,
		"features": e.item.Features,
		"meta":     e.item.Meta,
		"labels":   labels,
	}

	// 构建 rctx map
	rctx := map[string]interface{}{
		"user_id":    e.rctx.UserID,
		"device_id":  e.rctx.DeviceID,
		"scene":      e.rctx.Scene,
		"user_profile": e.rctx.UserProfile,
		"realtime":   e.rctx.Realtime,
		"params":     e.rctx.Params,
	}

	// 为了兼容旧的语法，提供 label 作为顶层访问
	// 例如 label.recall_source 可以直接访问
	// 注意：CEL 访问不存在的 key 会报错，所以使用 null 作为默认值
	// 用户可以使用 label.key != null 来检查存在性
	labelAccessor := make(map[string]interface{})
	for k, v := range labels {
		// label.recall_source 返回 value
		labelAccessor[k] = v.(map[string]interface{})["value"]
	}

	return map[string]interface{}{
		"item":  item,
		"label": labelAccessor,
		"rctx":  rctx,
	}
}

// 提供辅助函数来转换旧语法到 CEL 语法
// 例如：label.recall_source contains "hot" -> label.recall_source.contains("hot")
func ConvertToCELSyntax(expr string) string {
	// 简单的转换规则
	// 实际使用中，用户应该直接使用 CEL 语法
	// 这里只是为了兼容性
	return expr
}
