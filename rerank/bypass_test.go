package rerank

import (
	"context"
	"strconv"
	"testing"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/utils"
)

func TestBypassExtractAndBoost(t *testing.T) {
	items := []*core.Item{
		makeBypassItem("l0_1", 10, "pool.L0", 1),
		makeBypassItem("l0_2", 9, "pool.L0", 2),
		makeBypassItem("cf_1", 8, "recall.cf", 0),
		makeBypassItem("cf_2", 7, "recall.cf", 0),
		makeBypassItem("l0_3", 6, "pool.L0", 3),
	}

	rctx := &core.RecommendContext{UserID: "u1"}

	extract := &BypassExtractNode{
		GroupName:   "L0",
		LabelKey:    "recall_source",
		LabelValue:  "pool.L0",
		SortByLabel: "pool_rank",
		SortAsc:     true,
	}

	rest, err := extract.Process(context.Background(), rctx, items)
	if err != nil {
		t.Fatalf("extract error: %v", err)
	}
	if len(rest) != 2 {
		t.Fatalf("expected 2 remaining items, got %d", len(rest))
	}
	for _, it := range rest {
		if it.Labels["recall_source"].Value == "pool.L0" {
			t.Error("L0 items should have been extracted")
		}
	}

	boost := &BypassBoostNode{GroupName: "L0"}
	result, err := boost.Process(context.Background(), rctx, rest)
	if err != nil {
		t.Fatalf("boost error: %v", err)
	}
	if len(result) != 5 {
		t.Fatalf("expected 5 items after boost, got %d", len(result))
	}

	// First 3 should be L0 items in pool_rank order
	for i := 0; i < 3; i++ {
		if result[i].Labels["recall_source"].Value != "pool.L0" {
			t.Errorf("result[%d] should be L0, got %s", i, result[i].Labels["recall_source"].Value)
		}
	}
	if result[0].ID != "l0_1" || result[1].ID != "l0_2" || result[2].ID != "l0_3" {
		t.Errorf("L0 items not in pool_rank order: %s, %s, %s",
			result[0].ID, result[1].ID, result[2].ID)
	}
}

func TestBypassBoost_MaxItems(t *testing.T) {
	rctx := &core.RecommendContext{
		UserID: "u1",
		Params: map[string]any{
			bypassParamsPrefix + "test": []*core.Item{
				{ID: "a"}, {ID: "b"}, {ID: "c"},
			},
		},
	}

	boost := &BypassBoostNode{GroupName: "test", MaxItems: 2}
	result, err := boost.Process(context.Background(), rctx, []*core.Item{{ID: "d"}})
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(result) != 3 {
		t.Fatalf("expected 3 items (2 bypass + 1 original), got %d", len(result))
	}
	if result[0].ID != "a" || result[1].ID != "b" || result[2].ID != "d" {
		t.Errorf("unexpected order: %s, %s, %s", result[0].ID, result[1].ID, result[2].ID)
	}
}

func makeBypassItem(id string, score float64, source string, rank int) *core.Item {
	it := core.NewItem(id)
	it.Score = score
	it.Labels["recall_source"] = utils.Label{Value: source, Source: "recall"}
	if rank > 0 {
		it.Labels["pool_rank"] = utils.Label{Value: strconv.Itoa(rank), Source: "recall"}
	}
	return it
}
