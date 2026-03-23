package core_test

import (
	"testing"

	"github.com/rushteam/reckit/core"
)

// --- test doubles ---

type abTestExt struct {
	Group string
}

func (e *abTestExt) ExtensionName() string { return "aippy.abtest" }

type ratelimitExt struct {
	QPS int
}

func (e *ratelimitExt) ExtensionName() string { return "infra.ratelimit" }

// --- tests ---

func TestSetExtension_And_GetExtension(t *testing.T) {
	rctx := &core.RecommendContext{UserID: "u1"}
	ab := &abTestExt{Group: "experiment_a"}

	rctx.SetExtension(ab)

	got, ok := rctx.GetExtension("aippy.abtest")
	if !ok {
		t.Fatal("expected extension to be found")
	}
	if got.ExtensionName() != "aippy.abtest" {
		t.Errorf("expected name 'aippy.abtest', got %q", got.ExtensionName())
	}
}

func TestGetExtension_NotRegistered(t *testing.T) {
	rctx := &core.RecommendContext{}
	_, ok := rctx.GetExtension("nonexist")
	if ok {
		t.Error("expected ok=false for unregistered extension")
	}
}

func TestSetExtension_Overwrite(t *testing.T) {
	rctx := &core.RecommendContext{}
	rctx.SetExtension(&abTestExt{Group: "v1"})
	rctx.SetExtension(&abTestExt{Group: "v2"})

	e, ok := core.ExtensionAs[*abTestExt](rctx, "aippy.abtest")
	if !ok {
		t.Fatal("expected extension to be found")
	}
	if e.Group != "v2" {
		t.Errorf("expected overwritten value 'v2', got %q", e.Group)
	}
}

func TestExtensionAs_Success(t *testing.T) {
	rctx := &core.RecommendContext{}
	rctx.SetExtension(&abTestExt{Group: "control"})

	ab, ok := core.ExtensionAs[*abTestExt](rctx, "aippy.abtest")
	if !ok {
		t.Fatal("expected ExtensionAs to succeed")
	}
	if ab.Group != "control" {
		t.Errorf("expected Group='control', got %q", ab.Group)
	}
}

func TestExtensionAs_TypeMismatch(t *testing.T) {
	rctx := &core.RecommendContext{}
	rctx.SetExtension(&abTestExt{Group: "x"})

	_, ok := core.ExtensionAs[*ratelimitExt](rctx, "aippy.abtest")
	if ok {
		t.Error("expected ok=false when type does not match")
	}
}

func TestExtensionAs_NotFound(t *testing.T) {
	rctx := &core.RecommendContext{}
	_, ok := core.ExtensionAs[*abTestExt](rctx, "nonexist")
	if ok {
		t.Error("expected ok=false for missing extension")
	}
}

func TestSetExtension_NilSafe(t *testing.T) {
	var rctx *core.RecommendContext
	rctx.SetExtension(&abTestExt{Group: "x"}) // should not panic

	rctx2 := &core.RecommendContext{}
	rctx2.SetExtension(nil) // should not panic

	_, ok := rctx2.GetExtension("anything")
	if ok {
		t.Error("expected ok=false on empty context")
	}
}

func TestGetExtension_NilReceiver(t *testing.T) {
	var rctx *core.RecommendContext
	_, ok := rctx.GetExtension("anything")
	if ok {
		t.Error("expected ok=false on nil receiver")
	}
}

func TestMultipleExtensions(t *testing.T) {
	rctx := &core.RecommendContext{}
	rctx.SetExtension(&abTestExt{Group: "exp"})
	rctx.SetExtension(&ratelimitExt{QPS: 1000})

	ab, ok := core.ExtensionAs[*abTestExt](rctx, "aippy.abtest")
	if !ok || ab.Group != "exp" {
		t.Errorf("abtest extension: got %+v, ok=%v", ab, ok)
	}

	rl, ok := core.ExtensionAs[*ratelimitExt](rctx, "infra.ratelimit")
	if !ok || rl.QPS != 1000 {
		t.Errorf("ratelimit extension: got %+v, ok=%v", rl, ok)
	}
}
