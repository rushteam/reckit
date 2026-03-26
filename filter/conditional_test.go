package filter

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

type doubleScoreNode struct{}

func (d *doubleScoreNode) Name() string          { return "test.double" }
func (d *doubleScoreNode) Kind() pipeline.Kind    { return pipeline.KindReRank }
func (d *doubleScoreNode) Process(_ context.Context, _ *core.RecommendContext, items []*core.Item) ([]*core.Item, error) {
	for _, it := range items {
		it.Score *= 2
	}
	return items, nil
}

func TestConditionalNode_Active(t *testing.T) {
	items := []*core.Item{core.NewItem("a")}
	items[0].Score = 1.0

	node := &ConditionalNode{
		Cond: ConditionFunc(func(ctx context.Context, rctx *core.RecommendContext) (bool, error) {
			return true, nil
		}),
		Node: &doubleScoreNode{},
	}
	out, err := node.Process(context.Background(), &core.RecommendContext{}, items)
	if err != nil {
		t.Fatal(err)
	}
	if out[0].Score != 2.0 {
		t.Fatalf("score=%f want 2.0", out[0].Score)
	}
}

func TestConditionalNode_Inactive(t *testing.T) {
	items := []*core.Item{core.NewItem("a")}
	items[0].Score = 1.0

	node := &ConditionalNode{
		Cond: ConditionFunc(func(ctx context.Context, rctx *core.RecommendContext) (bool, error) {
			return false, nil
		}),
		Node: &doubleScoreNode{},
	}
	out, err := node.Process(context.Background(), &core.RecommendContext{}, items)
	if err != nil {
		t.Fatal(err)
	}
	if out[0].Score != 1.0 {
		t.Fatalf("score=%f want 1.0", out[0].Score)
	}
}

func TestConditionalNode_NilCond(t *testing.T) {
	items := []*core.Item{core.NewItem("a")}
	items[0].Score = 1.0

	node := &ConditionalNode{Node: &doubleScoreNode{}}
	out, err := node.Process(context.Background(), &core.RecommendContext{}, items)
	if err != nil {
		t.Fatal(err)
	}
	if out[0].Score != 1.0 {
		t.Fatalf("should passthrough when Cond is nil")
	}
}

func TestConditionalNode_KindDelegation(t *testing.T) {
	node := &ConditionalNode{Node: &doubleScoreNode{}}
	if node.Kind() != pipeline.KindReRank {
		t.Fatalf("kind=%v", node.Kind())
	}
}
