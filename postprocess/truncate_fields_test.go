package postprocess

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/utils"
)

func TestTruncateFields_ClearAll(t *testing.T) {
	it := core.NewItem("1")
	it.Features = map[string]float64{"ctr": 0.5}
	it.Meta = map[string]any{"title": "hello"}
	it.PutLabel("tag", utils.Label{Value: "v"})

	n := &TruncateFieldsNode{ClearFeatures: true, ClearMeta: true, ClearLabels: true}
	out, err := n.Process(context.Background(), nil, []*core.Item{it})
	if err != nil {
		t.Fatal(err)
	}
	if out[0].Features != nil || out[0].Meta != nil || out[0].Labels != nil {
		t.Fatal("fields not cleared")
	}
}

func TestTruncateFields_KeepMetaKeys(t *testing.T) {
	it := core.NewItem("1")
	it.Meta = map[string]any{"title": "hello", "cover": "url", "internal": "secret"}

	n := &TruncateFieldsNode{KeepMetaKeys: []string{"title", "cover"}}
	out, err := n.Process(context.Background(), nil, []*core.Item{it})
	if err != nil {
		t.Fatal(err)
	}
	if len(out[0].Meta) != 2 {
		t.Fatalf("meta keys %d want 2", len(out[0].Meta))
	}
	if _, ok := out[0].Meta["internal"]; ok {
		t.Fatal("internal should be removed")
	}
}
