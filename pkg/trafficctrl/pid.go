package trafficctrl

import "math"

// PIDController 位置式 PID 控制器，用于流量调控的闭环反馈。
//
// 工作原理：
//   - 输入：目标值与当前实际值的偏差 (error)
//   - 输出：排序分加权信号 (boost signal)
//   - P: 当前偏差越大，调控力度越大
//   - I: 累积偏差用于消除稳态误差（限幅 ±100 防 windup）
//   - D: 偏差变化率用于抑制振荡
type PIDController struct {
	params PIDParams

	prevError  float64
	integral   float64
	iterations int
}

// NewPIDController 创建 PID 控制器。
func NewPIDController(params PIDParams) *PIDController {
	return &PIDController{params: params}
}

const maxIntegral = 100.0

// Update 输入当前误差，返回调控信号。
// error = target - current；正值表示需要更多流量。
func (c *PIDController) Update(err float64) float64 {
	c.iterations++
	c.integral += err
	c.integral = math.Max(-maxIntegral, math.Min(maxIntegral, c.integral))

	derivative := err - c.prevError
	if c.iterations == 1 {
		derivative = 0
	}

	output := c.params.Kp*err + c.params.Ki*c.integral + c.params.Kd*derivative
	c.prevError = err
	return output
}

// Reset 重置控制器状态。
func (c *PIDController) Reset() {
	c.prevError = 0
	c.integral = 0
	c.iterations = 0
}

// PIDState 可序列化的 PID 状态，用于跨请求持久化。
type PIDState struct {
	TaskID     string  `json:"task_id"`
	PrevError  float64 `json:"prev_error"`
	Integral   float64 `json:"integral"`
	Iterations int     `json:"iterations"`
	LastOutput float64 `json:"last_output"`
}

// ToState 导出当前状态。
func (c *PIDController) ToState(taskID string, lastOutput float64) PIDState {
	return PIDState{
		TaskID:     taskID,
		PrevError:  c.prevError,
		Integral:   c.integral,
		Iterations: c.iterations,
		LastOutput: lastOutput,
	}
}

// FromState 从持久化状态恢复。
func (c *PIDController) FromState(state PIDState) {
	c.prevError = state.PrevError
	c.integral = state.Integral
	c.iterations = state.Iterations
}

// SignalToBoostFactor 将 PID 输出信号转换为加权系数。
// signal > 0 → factor > 1（提升）；signal < 0 → factor < 1（抑制）。
// 使用 sigmoid 映射，signal=0 时 factor≈1.0。
func SignalToBoostFactor(signal, maxFactor float64) float64 {
	if maxFactor <= 0 {
		maxFactor = 5.0
	}
	factor := 2.0 / (1.0 + math.Exp(-signal*0.1))
	factor = math.Max(1.0/maxFactor, math.Min(maxFactor, factor))
	return factor
}
