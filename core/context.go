package core

// RecommendContext 承载用户/场景/实时信息，贯穿整个 Pipeline 透传。
// 这里保持结构松散（map/any），以便快速原型；工业化可进一步引入强类型特征/Schema。
type RecommendContext struct {
	UserID   int64
	DeviceID string
	Scene    string

	UserProfile map[string]any
	Realtime    map[string]any
	Params      map[string]any
}
