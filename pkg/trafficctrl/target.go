package trafficctrl

// TargetType 调控目标类型。
type TargetType string

const (
	TargetGuarantee   TargetType = "guarantee"   // 保量：≥ 目标值
	TargetApproximate TargetType = "approximate" // 逼近：≈ 目标值
	TargetBoost       TargetType = "boost"       // 加权：分数乘以系数
	TargetSuppress    TargetType = "suppress"     // 打压：分数乘以 <1 系数
	TargetPin         TargetType = "pin"         // 置顶：强制前 N 位
)

// Metric 调控指标。
type Metric string

const (
	MetricExposureCount Metric = "exposure_count"
	MetricExposureRatio Metric = "exposure_ratio"
	MetricClickCount    Metric = "click_count"
	MetricInteractRate  Metric = "interaction_rate"
)

// Granularity 目标颗粒度。
type Granularity string

const (
	GranularityGlobal  Granularity = "global"
	GranularityPerItem Granularity = "per_item"
)

// Target 调控目标定义。
type Target struct {
	Name        string      `json:"name"`
	Type        TargetType  `json:"type"`
	Metric      Metric      `json:"metric"`
	Value       float64     `json:"value"`
	Tolerance   float64     `json:"tolerance"`
	Granularity Granularity `json:"granularity"`
	Window      string      `json:"window"` // 统计时间窗口
}

// IsSatisfied 判断当前指标值是否满足目标。
func (t *Target) IsSatisfied(currentValue float64) bool {
	switch t.Type {
	case TargetGuarantee:
		return currentValue >= t.Value
	case TargetApproximate:
		return currentValue >= t.Value-t.Tolerance && currentValue <= t.Value+t.Tolerance
	default:
		return false
	}
}

// Error 计算目标偏差（PID 控制器输入）。
// 正值 = 需要更多流量，负值 = 需要减少流量。
func (t *Target) Error(currentValue float64) float64 {
	return t.Value - currentValue
}
