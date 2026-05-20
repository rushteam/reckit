package rerank

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
)

func TestTopNNode_Basic(t *testing.T) {
	node := &TopNNode{N: 2}
	items := []*core.Item{
		core.NewItem("a"), core.NewItem("b"), core.NewItem("c"),
	}

	out, err := node.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("want 2, got %d", len(out))
	}
}

func TestTopNNode_ZeroN(t *testing.T) {
	node := &TopNNode{N: 0}
	items := []*core.Item{core.NewItem("a"), core.NewItem("b")}

	out, err := node.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("N<=0 should not truncate, got %d", len(out))
	}
}

func TestTopNNode_ConfigResolver_Override(t *testing.T) {
	node := &TopNNode{
		N: 10,
		ConfigResolver: func(rctx *core.RecommendContext) int {
			if v, ok := rctx.Params["topn"]; ok {
				return v.(int)
			}
			return 0
		},
	}

	rctx := &core.RecommendContext{
		UserID: "u1",
		Params: map[string]any{"topn": 2},
	}
	items := []*core.Item{
		core.NewItem("a"), core.NewItem("b"), core.NewItem("c"), core.NewItem("d"),
	}

	out, err := node.Process(context.Background(), rctx, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("ConfigResolver should override N: want 2, got %d", len(out))
	}
}

func TestTopNNode_ConfigResolver_NoOverride(t *testing.T) {
	node := &TopNNode{
		N: 3,
		ConfigResolver: func(rctx *core.RecommendContext) int {
			return 0
		},
	}

	rctx := &core.RecommendContext{UserID: "u1", Params: map[string]any{}}
	items := []*core.Item{
		core.NewItem("a"), core.NewItem("b"), core.NewItem("c"), core.NewItem("d"),
	}

	out, err := node.Process(context.Background(), rctx, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 {
		t.Fatalf("ConfigResolver returned 0 should fallback to N=3: want 3, got %d", len(out))
	}
}
