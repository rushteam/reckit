package rerank

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/utils"
)

func TestGroupQuota_Softmax(t *testing.T) {
	items := []*core.Item{
		catItem("a1", 0.9, "tech"),
		catItem("a2", 0.8, "tech"),
		catItem("a3", 0.7, "tech"),
		catItem("b1", 0.6, "sports"),
		catItem("b2", 0.5, "sports"),
		catItem("c1", 0.4, "music"),
	}
	node := &GroupQuotaNode{
		N:        4,
		FieldKey: "category",
		Strategy: GroupQuotaSoftmax,
	}
	out, err := node.Process(context.Background(), &core.RecommendContext{}, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 4 {
		t.Fatalf("len=%d", len(out))
	}
}

func TestGroupQuota_Avg(t *testing.T) {
	items := []*core.Item{
		catItem("a1", 0.9, "tech"),
		catItem("a2", 0.8, "tech"),
		catItem("b1", 0.6, "sports"),
		catItem("b2", 0.5, "sports"),
	}
	node := &GroupQuotaNode{
		N:        4,
		FieldKey: "category",
		Strategy: GroupQuotaAvg,
	}
	out, err := node.Process(context.Background(), &core.RecommendContext{}, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 4 {
		t.Fatalf("len=%d", len(out))
	}
	techCount := 0
	for _, it := range out {
		if v, _ := it.GetValue("category"); v == "tech" {
			techCount++
		}
	}
	if techCount != 2 {
		t.Fatalf("tech=%d want 2", techCount)
	}
}

func TestGroupQuota_MinMax(t *testing.T) {
	items := []*core.Item{
		catItem("a1", 0.9, "tech"),
		catItem("a2", 0.8, "tech"),
		catItem("a3", 0.7, "tech"),
		catItem("b1", 0.1, "sports"),
	}
	node := &GroupQuotaNode{
		N:        4,
		FieldKey: "category",
		Strategy: GroupQuotaSoftmax,
		GroupMin: 1,
		GroupMax: 2,
	}
	out, err := node.Process(context.Background(), &core.RecommendContext{}, items)
	if err != nil {
		t.Fatal(err)
	}
	techCount, sportsCount := 0, 0
	for _, it := range out {
		v, _ := it.GetValue("category")
		switch v {
		case "tech":
			techCount++
		case "sports":
			sportsCount++
		}
	}
	if techCount > 2 {
		t.Fatalf("tech=%d want <=2", techCount)
	}
	if sportsCount < 1 {
		t.Fatalf("sports=%d want >=1", sportsCount)
	}
}

func TestGroupQuota_ExprGroups(t *testing.T) {
	items := []*core.Item{
		newItemWithSource("h1", 0.9, "hot"),
		newItemWithSource("h2", 0.8, "hot"),
		newItemWithSource("c1", 0.7, "cf"),
		newItemWithSource("c2", 0.6, "cf"),
	}
	node := &GroupQuotaNode{
		N: 3,
		ExprGroups: []ExprGroup{
			{Name: "hot_group", Expr: `label.recall_source == "hot"`, Quota: 1},
			{Name: "cf_group", Expr: `label.recall_source == "cf"`, Quota: 2},
		},
	}
	out, err := node.Process(context.Background(), &core.RecommendContext{}, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 {
		t.Fatalf("len=%d", len(out))
	}
}

func catItem(id string, score float64, category string) *core.Item {
	it := core.NewItem(id)
	it.Score = score
	it.PutLabel("category", utils.Label{Value: category, Source: "test"})
	return it
}
