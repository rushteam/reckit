package trafficctrl

import "context"

// TaskRepository 任务配置读取端口。
type TaskRepository interface {
	GetActiveTasks(ctx context.Context, scene string) ([]*Task, error)
	GetTask(ctx context.Context, taskID string) (*Task, error)
}

// MetricsReader 实时指标读取端口（PID 闭环依赖）。
type MetricsReader interface {
	GetMetric(ctx context.Context, taskID string, metric Metric, window string) (float64, error)
}

// PIDStateStore PID 状态持久化端口（跨请求保持积分项等）。
type PIDStateStore interface {
	Load(ctx context.Context, taskID string) (PIDState, error)
	Save(ctx context.Context, state PIDState) error
}

// EventEmitter 调控事件上报端口（异步，不阻塞主流程）。
type EventEmitter interface {
	EmitBoostEvent(ctx context.Context, event BoostEvent)
}

// BoostEvent 单次调控行为记录。
type BoostEvent struct {
	TaskID      string  `json:"task_id"`
	ItemID      string  `json:"item_id"`
	UserID      string  `json:"user_id"`
	Scene       string  `json:"scene"`
	BoostFactor float64 `json:"boost_factor"`
	OrigScore   float64 `json:"orig_score"`
	NewScore    float64 `json:"new_score"`
	Position    int     `json:"position"`
}
