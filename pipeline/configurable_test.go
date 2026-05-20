package pipeline_test

import (
	"context"
	"errors"
	"testing"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

type configurableNode struct {
	name      string
	topK      int
	configured bool
}

func (n *configurableNode) Name() string        { return n.name }
func (n *configurableNode) Kind() pipeline.Kind  { return pipeline.KindReRank }

func (n *configurableNode) ApplyConfig(_ context.Context, rctx *core.RecommendContext) error {
	n.configured = true
	if v, ok := rctx.Params["topk"]; ok {
		n.topK = v.(int)
	}
	return nil
}

func (n *configurableNode) Process(_ context.Context, _ *core.RecommendContext, items []*core.Item) ([]*core.Item, error) {
	if n.topK > 0 && len(items) > n.topK {
		return items[:n.topK], nil
	}
	return items, nil
}

type nonConfigurableNode struct{}

func (n *nonConfigurableNode) Name() string        { return "plain" }
func (n *nonConfigurableNode) Kind() pipeline.Kind  { return pipeline.KindRank }
func (n *nonConfigurableNode) Process(_ context.Context, _ *core.RecommendContext, items []*core.Item) ([]*core.Item, error) {
	return items, nil
}

func TestPipeline_Configurable_ApplyConfigCalled(t *testing.T) {
	node := &configurableNode{name: "cfg_node", topK: 0}
	p := &pipeline.Pipeline{Nodes: []pipeline.Node{node}}

	rctx := &core.RecommendContext{
		UserID: "u1",
		Params: map[string]any{"topk": 2},
	}
	input := []*core.Item{core.NewItem("a"), core.NewItem("b"), core.NewItem("c")}

	out, err := p.Run(context.Background(), rctx, input)
	if err != nil {
		t.Fatal(err)
	}
	if !node.configured {
		t.Fatal("ApplyConfig was not called")
	}
	if len(out) != 2 {
		t.Fatalf("expected 2 items after topK override, got %d", len(out))
	}
}

func TestPipeline_NonConfigurable_SkipsApplyConfig(t *testing.T) {
	node := &nonConfigurableNode{}
	p := &pipeline.Pipeline{Nodes: []pipeline.Node{node}}

	rctx := &core.RecommendContext{UserID: "u1", Params: map[string]any{}}
	input := []*core.Item{core.NewItem("a")}

	out, err := p.Run(context.Background(), rctx, input)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 {
		t.Fatalf("expected 1 item, got %d", len(out))
	}
}

type failConfigNode struct{}

func (n *failConfigNode) Name() string        { return "fail_cfg" }
func (n *failConfigNode) Kind() pipeline.Kind  { return pipeline.KindReRank }
func (n *failConfigNode) ApplyConfig(_ context.Context, _ *core.RecommendContext) error {
	return errors.New("config error")
}
func (n *failConfigNode) Process(_ context.Context, _ *core.RecommendContext, items []*core.Item) ([]*core.Item, error) {
	return items, nil
}

func TestPipeline_Configurable_ErrorHook_Recovery(t *testing.T) {
	node := &failConfigNode{}
	hook := &alwaysRecoverHook{}
	p := &pipeline.Pipeline{
		Nodes:      []pipeline.Node{node},
		ErrorHooks: []pipeline.ErrorHook{hook},
	}

	rctx := &core.RecommendContext{UserID: "u1", Params: map[string]any{}}
	input := []*core.Item{core.NewItem("a")}

	out, err := p.Run(context.Background(), rctx, input)
	if err != nil {
		t.Fatalf("expected recovery, got error: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("expected items to pass through on recovery, got %d", len(out))
	}
}

func TestPipeline_Configurable_ErrorHook_NoRecovery(t *testing.T) {
	node := &failConfigNode{}
	p := &pipeline.Pipeline{Nodes: []pipeline.Node{node}}

	rctx := &core.RecommendContext{UserID: "u1", Params: map[string]any{}}
	input := []*core.Item{core.NewItem("a")}

	_, err := p.Run(context.Background(), rctx, input)
	if err == nil {
		t.Fatal("expected error without recovery hook")
	}
}

type alwaysRecoverHook struct{}

func (h *alwaysRecoverHook) OnNodeError(_ context.Context, _ *core.RecommendContext, _ pipeline.Node, _ error) bool {
	return true
}
