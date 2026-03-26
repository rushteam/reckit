package filter

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/utils"
)

func TestExprFilter_Basic(t *testing.T) {
	it := core.NewItem("1")
	it.Score = 0.05
	f := &ExprFilter{Expr: `item.score < 0.1`}
	ok, err := f.ShouldFilter(context.Background(), nil, it)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("should filter low score")
	}

	it2 := core.NewItem("2")
	it2.Score = 0.5
	ok2, err := f.ShouldFilter(context.Background(), nil, it2)
	if err != nil {
		t.Fatal(err)
	}
	if ok2 {
		t.Fatal("should not filter high score")
	}
}

func TestExprFilter_Label(t *testing.T) {
	it := core.NewItem("1")
	it.PutLabel("category", utils.Label{Value: "adult"})
	f := &ExprFilter{Expr: `label.category == "adult"`}
	ok, err := f.ShouldFilter(context.Background(), nil, it)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("should filter adult")
	}
}

func TestExprFilter_Invert(t *testing.T) {
	it := core.NewItem("1")
	it.Score = 0.9
	f := &ExprFilter{Expr: `item.score > 0.5`, Invert: true}
	ok, err := f.ShouldFilter(context.Background(), nil, it)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("invert: score>0.5 should keep")
	}
}
