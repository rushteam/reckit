package core

import "github.com/rushteam/reckit/pkg/utils"

// RecommendContext 承载用户/场景/实时信息，贯穿整个 Pipeline 透传。
type RecommendContext struct {
	UserID   string // 使用 string 类型（通用，支持所有 ID 格式）
	DeviceID string
	Scene    string

	// User 是调用方透传的用户对象（any 类型）。
	// 框架内置 Node 不会直接读取此字段，仅供自定义 Node 做 type assert。
	// 框架读取用户数据的标准通道是 Attributes。
	User any

	// Attributes 是用户级属性 map，框架读取用户数据的标准通道。
	// 包括用户特征（age、gender 等）、向量（user_embedding）、行为序列（recent_clicks）等。
	Attributes map[string]any

	// Labels 是用户级标签，可驱动整个 Pipeline 行为
	// 例如：新用户、重度用户、价格敏感等
	Labels map[string]utils.Label

	// Params 请求级上下文参数，包含：
	// - 请求参数：latitude, longitude, time_of_day, query, device_type 等
	// - 实时特征：realtime_ctr, realtime_exposure 等（建议加 realtime_ 前缀区分）
	Params map[string]any
}

// PutLabel 写入用户级 Label。
func (rctx *RecommendContext) PutLabel(key string, lbl utils.Label) {
	if rctx.Labels == nil {
		rctx.Labels = make(map[string]utils.Label)
	}
	if old, ok := rctx.Labels[key]; ok {
		rctx.Labels[key] = utils.MergeLabel(old, lbl)
		return
	}
	rctx.Labels[key] = lbl
}

// GetLabel 获取用户级 Label。
func (rctx *RecommendContext) GetLabel(key string) (utils.Label, bool) {
	if rctx.Labels == nil {
		return utils.Label{}, false
	}
	lbl, ok := rctx.Labels[key]
	return lbl, ok
}
