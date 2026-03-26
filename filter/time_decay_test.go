package filter

import (
	"context"
	"testing"
	"time"

	"github.com/rushteam/reckit/core"
)

func TestTimeDecayFilter(t *testing.T) {
	now := time.Date(2025, 1, 10, 0, 0, 0, 0, time.UTC)
	f := &TimeDecayFilter{
		TimeField: "publish_time",
		MaxAge:    7 * 24 * time.Hour,
		NowFunc:   func() time.Time { return now },
	}

	fresh := core.NewItem("1")
	fresh.Meta = map[string]any{"publish_time": now.Add(-24 * time.Hour)}
	ok1, _ := f.ShouldFilter(context.Background(), nil, fresh)
	if ok1 {
		t.Fatal("fresh item should not be filtered")
	}

	old := core.NewItem("2")
	old.Meta = map[string]any{"publish_time": now.Add(-10 * 24 * time.Hour)}
	ok2, _ := f.ShouldFilter(context.Background(), nil, old)
	if !ok2 {
		t.Fatal("old item should be filtered")
	}
}

func TestTimeDecayFilter_Unix(t *testing.T) {
	now := time.Date(2025, 1, 10, 0, 0, 0, 0, time.UTC)
	f := &TimeDecayFilter{
		TimeField: "ts",
		MaxAge:    24 * time.Hour,
		NowFunc:   func() time.Time { return now },
	}

	it := core.NewItem("1")
	it.Meta = map[string]any{"ts": now.Add(-2 * 24 * time.Hour).Unix()}
	ok, _ := f.ShouldFilter(context.Background(), nil, it)
	if !ok {
		t.Fatal("should filter expired unix ts")
	}
}
