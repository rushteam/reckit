package filter

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
)

func TestQualityGate(t *testing.T) {
	f := &QualityGateFilter{MinScore: 0.3}
	tests := []struct {
		score  float64
		filter bool
	}{
		{0.1, true},
		{0.3, false},
		{0.5, false},
	}
	for _, tt := range tests {
		it := core.NewItem("x")
		it.Score = tt.score
		ok, err := f.ShouldFilter(context.Background(), nil, it)
		if err != nil {
			t.Fatal(err)
		}
		if ok != tt.filter {
			t.Errorf("score=%.1f: got %v want %v", tt.score, ok, tt.filter)
		}
	}
}
