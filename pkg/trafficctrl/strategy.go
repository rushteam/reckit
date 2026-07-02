package trafficctrl

// BoostType 调控执行策略类型。
type BoostType string

const (
	BoostTypeWeight BoostType = "weight" // 直接加权（无反馈闭环）
	BoostTypePID    BoostType = "pid"    // PID 闭环控制
	BoostTypePin    BoostType = "pin"    // 置顶（强插指定位置）
)

// Strategy 调控执行策略。
type Strategy struct {
	BoostType      BoostType       `json:"boost_type"`
	PIDParams      PIDParams       `json:"pid_params,omitempty"`
	BoostFactor    float64         `json:"boost_factor,omitempty"`
	MaxBoostFactor float64         `json:"max_boost_factor,omitempty"`
	FallbackBoost  float64         `json:"fallback_boost,omitempty"`
	PinPositions   []int           `json:"pin_positions,omitempty"`
	Diversity      DiversityConfig `json:"diversity"`
}

// EffectiveMaxBoost 有效最大加权倍数（默认 5.0）。
func (s *Strategy) EffectiveMaxBoost() float64 {
	if s.MaxBoostFactor > 0 {
		return s.MaxBoostFactor
	}
	return 5.0
}

// PIDParams PID 控制器参数。
type PIDParams struct {
	Kp float64 `json:"kp"`
	Ki float64 `json:"ki"`
	Kd float64 `json:"kd"`
}

// DiversityConfig 被调控物品的多样性约束。
type DiversityConfig struct {
	WindowSize int `json:"window_size"` // 滑动窗口大小
	MaxBoosted int `json:"max_boosted"` // 窗口内最多允许的被调控物品数
	MinGap     int `json:"min_gap"`     // 被调控物品之间的最小间隔
}

// HasDiversity 是否配置了多样性约束。
func (d *DiversityConfig) HasDiversity() bool {
	return d.WindowSize > 0 && d.MaxBoosted > 0
}
