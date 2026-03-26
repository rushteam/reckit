package rerank

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/utils"
)

func TestScoreAdjust_FirstMatch(t *testing.T) {
	a := core.NewItem("a")
	a.Score = 1
	a.PutLabel("recall_source", utils.Label{Value: "hot", Source: "t"})
	b := core.NewItem("b")
	b.Score = 2
	b.PutLabel("recall_source", utils.Label{Value: "cf", Source: "t"})
	node := &ScoreAdjust{
		Rules: []ScoreAdjustRule{
			{Expr: `label.recall_source == "hot"`, Mode: ScoreAdjustMul, Value: 10},
			{Expr: `true`, Mode: ScoreAdjustAdd, Value: 100},
		},
	}
	out, err := node.Process(context.Background(), &core.RecommendContext{}, []*core.Item{a, b})
	if err != nil {
		t.Fatal(err)
	}
	if out[0].Score != 10 {
		t.Fatalf("a score=%v want 10", out[0].Score)
	}
	if out[1].Score != 102 {
		t.Fatalf("b score=%v want 102 (second rule)", out[1].Score)
	}
}

func TestScoreAdjust_MatchAll(t *testing.T) {
	a := core.NewItem("a")
	a.Score = 1
	a.PutLabel("recall_source", utils.Label{Value: "hot", Source: "t"})
	node := &ScoreAdjust{
		MatchAllRules: true,
		Rules: []ScoreAdjustRule{
			{Expr: `label.recall_source == "hot"`, Mode: ScoreAdjustMul, Value: 2},
			{Expr: `item.score > 0`, Mode: ScoreAdjustAdd, Value: 3},
		},
	}
	out, err := node.Process(context.Background(), &core.RecommendContext{}, []*core.Item{a})
	if err != nil {
		t.Fatal(err)
	}
	if out[0].Score != 5 {
		t.Fatalf("score=%v want 1*2+3=5", out[0].Score)
	}
}

func TestScoreWeightBoost_Mul(t *testing.T) {
	p := &mockWeightProvider{m: map[string]float64{"a": 2, "b": 0.5}}
	a := core.NewItem("a")
	a.Score = 3
	b := core.NewItem("b")
	b.Score = 4
	c := core.NewItem("c")
	c.Score = 5
	node := &ScoreWeightBoost{Provider: p, Mode: ScoreWeightMul}
	out, err := node.Process(context.Background(), &core.RecommendContext{}, []*core.Item{a, b, c})
	if err != nil {
		t.Fatal(err)
	}
	if out[0].Score != 6 || out[1].Score != 2 || out[2].Score != 5 {
		t.Fatalf("scores %v %v %v", out[0].Score, out[1].Score, out[2].Score)
	}
}

func TestScoreWeightBoost_Add(t *testing.T) {
	p := &mockWeightProvider{m: map[string]float64{"a": 1}}
	a := core.NewItem("a")
	a.Score = 3
	node := &ScoreWeightBoost{Provider: p, Mode: ScoreWeightAdd}
	out, err := node.Process(context.Background(), &core.RecommendContext{}, []*core.Item{a})
	if err != nil {
		t.Fatal(err)
	}
	if out[0].Score != 4 {
		t.Fatalf("score=%v", out[0].Score)
	}
}

type mockWeightProvider struct {
	m map[string]float64
}

func (m *mockWeightProvider) Weights(context.Context, *core.RecommendContext, []*core.Item) (map[string]float64, error) {
	return m.m, nil
}
