package feature

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
)

func TestEnrichNode_InitializesNilItemFeatures(t *testing.T) {
	node := &EnrichNode{}
	rctx := &core.RecommendContext{
		UserID: "42",
		Scene:  "feed",
	}
	items := []*core.Item{
		{ID: "item_1"},
	}

	out, err := node.Process(context.Background(), rctx, items)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("unexpected output length: %d", len(out))
	}
	if out[0].Features == nil {
		t.Fatal("item features should be initialized")
	}
}
