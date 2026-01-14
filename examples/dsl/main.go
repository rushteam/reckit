package main

import (
	"fmt"

	"reckit/core"
	"reckit/pkg/dsl"
	"reckit/pkg/utils"
)

func main() {
	// 创建一个测试 Item
	item := core.NewItem(123)
	item.Score = 0.85
	item.Features["ctr"] = 0.15
	item.Features["cvr"] = 0.08
	item.PutLabel("recall_source", utils.Label{Value: "hot|ann", Source: "recall"})
	item.PutLabel("rank_model", utils.Label{Value: "lr", Source: "rank"})
	item.PutLabel("category", utils.Label{Value: "A", Source: "rule"})

	rctx := &core.RecommendContext{
		UserID: 42,
		Scene:  "feed",
	}

	// 创建 DSL 解释器
	eval := dsl.NewEval(item, rctx)

	// 测试各种表达式（使用 CEL 语法）
	exprs := []string{
		`label.recall_source.contains("hot")`,
		`label.rank_model == "lr"`,
		`item.score > 0.7`,
		`label.category == "A" && item.score > 0.8`,
		`label.recall_source != null`,
		`!("nonexist" in label)`,
		`label.recall_source.contains("ann") || label.recall_source.contains("cf")`,
	}

	fmt.Println("DSL 表达式测试:")
	for _, expr := range exprs {
		result, err := eval.Evaluate(expr)
		if err != nil {
			fmt.Printf("  ❌ %s -> 错误: %v\n", expr, err)
		} else {
			status := "✓"
			if !result {
				status = "✗"
			}
			fmt.Printf("  %s %s -> %v\n", status, expr, result)
		}
	}
}
