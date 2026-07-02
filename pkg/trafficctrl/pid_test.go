package trafficctrl

import (
	"math"
	"testing"
	"time"
)

func TestPIDController_Update(t *testing.T) {
	pid := NewPIDController(PIDParams{Kp: 5, Ki: 1, Kd: 1})

	out := pid.Update(0.1)
	if out <= 0 {
		t.Errorf("expected positive output for positive error, got %f", out)
	}

	out2 := pid.Update(0.1)
	if out2 <= out {
		t.Errorf("expected increasing output with accumulating integral, got %f <= %f", out2, out)
	}

	out3 := pid.Update(-0.2)
	if out3 >= out2 {
		t.Errorf("expected decreasing output for negative error, got %f >= %f", out3, out2)
	}
}

func TestPIDController_Reset(t *testing.T) {
	pid := NewPIDController(PIDParams{Kp: 5, Ki: 1, Kd: 1})
	pid.Update(0.5)
	pid.Update(0.5)
	pid.Reset()

	fresh := NewPIDController(PIDParams{Kp: 5, Ki: 1, Kd: 1})
	out1 := pid.Update(0.1)
	out2 := fresh.Update(0.1)
	if math.Abs(out1-out2) > 1e-9 {
		t.Errorf("after reset, output differs: %f vs %f", out1, out2)
	}
}

func TestPIDController_IntegralWindup(t *testing.T) {
	pid := NewPIDController(PIDParams{Kp: 0, Ki: 1, Kd: 0})
	for i := 0; i < 1000; i++ {
		pid.Update(1000.0)
	}
	if pid.integral > maxIntegral {
		t.Errorf("expected integral capped at %f, got %f", maxIntegral, pid.integral)
	}
}

func TestPIDController_StateRoundTrip(t *testing.T) {
	pid := NewPIDController(PIDParams{Kp: 5, Ki: 1, Kd: 1})
	pid.Update(0.3)
	pid.Update(0.2)

	state := pid.ToState("task_1", 1.5)
	pid2 := NewPIDController(PIDParams{Kp: 5, Ki: 1, Kd: 1})
	pid2.FromState(state)

	out1 := pid.Update(0.1)
	out2 := pid2.Update(0.1)
	if math.Abs(out1-out2) > 1e-9 {
		t.Errorf("state roundtrip failed: %f vs %f", out1, out2)
	}
}

func TestSignalToBoostFactor(t *testing.T) {
	tests := []struct {
		signal, max, wantMin, wantMax float64
	}{
		{0, 5.0, 0.9, 1.1},
		{10, 5.0, 1.0, 5.0},
		{-10, 5.0, 0.2, 1.0},
		{100, 3.0, 1.5, 3.0},
	}
	for _, tt := range tests {
		got := SignalToBoostFactor(tt.signal, tt.max)
		if got < tt.wantMin || got > tt.wantMax {
			t.Errorf("SignalToBoostFactor(%f, %f) = %f, want [%f, %f]",
				tt.signal, tt.max, got, tt.wantMin, tt.wantMax)
		}
	}
}

func TestRuleSet_MatchAll(t *testing.T) {
	attrs := map[string]any{"age": 25, "country": "CN", "vip": true}

	rs := RuleSet{Rules: []Rule{
		{Field: "age", Op: OpGte, Value: 18},
		{Field: "country", Op: OpEq, Value: "CN"},
	}}
	if !rs.MatchAll(attrs) {
		t.Error("expected match")
	}

	rs2 := RuleSet{Rules: []Rule{
		{Field: "age", Op: OpLt, Value: 18},
	}}
	if rs2.MatchAll(attrs) {
		t.Error("expected no match")
	}

	empty := RuleSet{}
	if !empty.MatchAll(attrs) {
		t.Error("empty ruleset should match all")
	}
}

func TestItemPool_MatchItem(t *testing.T) {
	pool := ItemPool{
		Type:      ItemPoolMixed,
		StaticIDs: []string{"item_1", "item_2"},
		RuleSet:   RuleSet{Rules: []Rule{{Field: "category", Op: OpEq, Value: "tech"}}},
	}

	if !pool.MatchItem("item_1", nil) {
		t.Error("static ID should match")
	}
	if !pool.MatchItem("item_99", map[string]any{"category": "tech"}) {
		t.Error("dynamic rule should match")
	}
	if pool.MatchItem("item_99", map[string]any{"category": "food"}) {
		t.Error("should not match")
	}
}

func TestSchedule_InWindow(t *testing.T) {
	now := time.Date(2024, 6, 15, 14, 0, 0, 0, time.UTC)
	s := Schedule{
		StartTime:        time.Date(2024, 6, 1, 0, 0, 0, 0, time.UTC),
		EndTime:          time.Date(2024, 7, 1, 0, 0, 0, 0, time.UTC),
		DailyActiveHours: [2]int{10, 18},
	}
	if !s.InWindow(now) {
		t.Error("14:00 should be in [10, 18)")
	}

	s2 := Schedule{
		StartTime:        time.Date(2024, 6, 1, 0, 0, 0, 0, time.UTC),
		EndTime:          time.Date(2024, 7, 1, 0, 0, 0, 0, time.UTC),
		DailyActiveHours: [2]int{22, 6},
	}
	night := time.Date(2024, 6, 15, 23, 0, 0, 0, time.UTC)
	if !s2.InWindow(night) {
		t.Error("23:00 should be in [22, 6) (cross-midnight)")
	}
	morning := time.Date(2024, 6, 15, 10, 0, 0, 0, time.UTC)
	if s2.InWindow(morning) {
		t.Error("10:00 should not be in [22, 6)")
	}
}

func TestTarget_IsSatisfied(t *testing.T) {
	tg := Target{Type: TargetGuarantee, Value: 100}
	if !tg.IsSatisfied(100) {
		t.Error("100 >= 100 should be satisfied")
	}
	if tg.IsSatisfied(99) {
		t.Error("99 < 100 should not be satisfied")
	}

	ta := Target{Type: TargetApproximate, Value: 50, Tolerance: 5}
	if !ta.IsSatisfied(48) {
		t.Error("48 in [45, 55] should be satisfied")
	}
	if ta.IsSatisfied(44) {
		t.Error("44 < 45 should not be satisfied")
	}
}
