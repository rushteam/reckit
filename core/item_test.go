package core_test

import (
	"testing"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/utils"
)

func TestGetValue_Labels(t *testing.T) {
	item := core.NewItem("1")
	item.PutLabel("category", utils.Label{Value: "tech"})
	item.Meta["category"] = "ignored"
	item.Features["category"] = 99

	v, ok := item.GetValue("category")
	if !ok || v != "tech" {
		t.Errorf("expected 'tech', got %q (ok=%v)", v, ok)
	}
}

func TestGetValue_Meta(t *testing.T) {
	item := core.NewItem("1")
	item.Meta["author"] = "alice"
	item.Features["author"] = 42

	v, ok := item.GetValue("author")
	if !ok || v != "alice" {
		t.Errorf("expected 'alice', got %q (ok=%v)", v, ok)
	}
}

func TestGetValue_Features_Integer(t *testing.T) {
	item := core.NewItem("1")
	item.Features["category_id"] = 7

	v, ok := item.GetValue("category_id")
	if !ok || v != "7" {
		t.Errorf("expected '7', got %q (ok=%v)", v, ok)
	}
}

func TestGetValue_Features_Float(t *testing.T) {
	item := core.NewItem("1")
	item.Features["score"] = 3.14

	v, ok := item.GetValue("score")
	if !ok || v != "3.14" {
		t.Errorf("expected '3.14', got %q (ok=%v)", v, ok)
	}
}

func TestGetValue_Missing(t *testing.T) {
	item := core.NewItem("1")
	v, ok := item.GetValue("nonexist")
	if ok || v != "" {
		t.Errorf("expected ('', false), got (%q, %v)", v, ok)
	}
}

func TestGetValue_Nil(t *testing.T) {
	item := &core.Item{ID: "1"}
	v, ok := item.GetValue("anything")
	if ok || v != "" {
		t.Errorf("expected ('', false), got (%q, %v)", v, ok)
	}
}
