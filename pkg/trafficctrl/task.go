package trafficctrl

import "time"

// TaskStatus 任务状态。
type TaskStatus string

const (
	TaskStatusDraft     TaskStatus = "DRAFT"
	TaskStatusActive    TaskStatus = "ACTIVE"
	TaskStatusPaused    TaskStatus = "PAUSED"
	TaskStatusCompleted TaskStatus = "COMPLETED"
)

// Task 流量调控任务：运营管理的最小调控单元。
type Task struct {
	ID       string     `json:"task_id"`
	Name     string     `json:"name"`
	Scene    string     `json:"scene"`
	Status   TaskStatus `json:"status"`
	Priority int        `json:"priority"` // 数字越大优先级越高

	Schedule Schedule `json:"schedule"`
	Audience Audience `json:"audience"`
	ItemPool ItemPool `json:"item_pool"`
	Targets  []Target `json:"targets"`
	Strategy Strategy `json:"strategy"`

	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// IsActive 判断任务当前是否活跃（状态 + 时间窗口）。
func (t *Task) IsActive(now time.Time) bool {
	if t.Status != TaskStatusActive {
		return false
	}
	return t.Schedule.InWindow(now)
}

// Schedule 任务时间调度。
type Schedule struct {
	StartTime        time.Time `json:"start_time"`
	EndTime          time.Time `json:"end_time"`
	DailyActiveHours [2]int    `json:"daily_active_hours"` // [startHour, endHour)，空则全天
}

// InWindow 判断给定时间是否在调控窗口内。
func (s *Schedule) InWindow(now time.Time) bool {
	if now.Before(s.StartTime) || now.After(s.EndTime) {
		return false
	}
	if s.DailyActiveHours == [2]int{} {
		return true
	}
	hour := now.Hour()
	start, end := s.DailyActiveHours[0], s.DailyActiveHours[1]
	if start <= end {
		return hour >= start && hour < end
	}
	return hour >= start || hour < end
}
