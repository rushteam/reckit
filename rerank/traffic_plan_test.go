package rerank

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
)

func TestTrafficPlanNode_NoOp(t *testing.T) {
	a := core.NewItem("a")
	b := core.NewItem("b")
	items := []*core.Item{a, b}
	node := &TrafficPlanNode{Planner: NoOpTrafficPlanner{}}
	out, err := node.Process(context.Background(), &core.RecommendContext{}, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 || out[0] != a || out[1] != b {
		t.Fatalf("unexpected out: %v", out)
	}
}

func TestTrafficPlanNode_EmptyInput(t *testing.T) {
	node := &TrafficPlanNode{Planner: NoOpTrafficPlanner{}}
	out, err := node.Process(context.Background(), &core.RecommendContext{}, nil)
	if err != nil || out != nil {
		t.Fatalf("err=%v out=%v", err, out)
	}
}

func TestTrafficPlanNode_StaticLabels(t *testing.T) {
	a := core.NewItem("a")
	b := core.NewItem("b")
	items := []*core.Item{a, b}
	node := &TrafficPlanNode{
		Planner: &StaticTrafficPlanner{ControlID: "cid", Slot: "3"},
	}
	out, err := node.Process(context.Background(), &core.RecommendContext{}, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("len %d", len(out))
	}
	for _, it := range out {
		if it.Labels[LabelKeyTrafficControlID].Value != "cid" {
			t.Fatalf("control id: %+v", it.Labels)
		}
		if it.Labels[LabelKeyTrafficSlot].Value != "3" {
			t.Fatalf("slot: %+v", it.Labels)
		}
	}
}

func TestTrafficPlanNode_Reorder(t *testing.T) {
	a := core.NewItem("a")
	b := core.NewItem("b")
	items := []*core.Item{a, b}
	planner := reorderPlanner{out: []*core.Item{b, a}}
	node := &TrafficPlanNode{Planner: planner}
	out, err := node.Process(context.Background(), &core.RecommendContext{}, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 || out[0] != b || out[1] != a {
		t.Fatalf("got %v %v", out[0].ID, out[1].ID)
	}
}

func TestTrafficPlanNode_ReorderInvalid(t *testing.T) {
	a := core.NewItem("a")
	b := core.NewItem("b")
	items := []*core.Item{a, b}
	planner := invalidReorderPlanner{}
	node := &TrafficPlanNode{Planner: planner}
	_, err := node.Process(context.Background(), &core.RecommendContext{}, items)
	if err == nil {
		t.Fatal("expected error")
	}
}

type reorderPlanner struct {
	out []*core.Item
}

func (p reorderPlanner) Plan(context.Context, *core.RecommendContext, []*core.Item) (*TrafficPlanResult, error) {
	return &TrafficPlanResult{Items: p.out}, nil
}

type invalidReorderPlanner struct{}

func (invalidReorderPlanner) Plan(context.Context, *core.RecommendContext, []*core.Item) (*TrafficPlanResult, error) {
	return &TrafficPlanResult{Items: []*core.Item{core.NewItem("x")}}, nil
}
