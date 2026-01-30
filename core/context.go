package core

import "github.com/rushteam/reckit/pkg/utils"

// RecommendContext 承载用户/场景/实时信息，贯穿整个 Pipeline 透传。
type RecommendContext struct {
	UserID   string // 使用 string 类型（通用，支持所有 ID 格式）
	DeviceID string
	Scene    string

	// User 是强类型用户画像
	User *UserProfile

	// UserProfile 是 map 形式，用于快速原型或动态属性
	// 如果 User 不为空，优先使用 User；否则使用 UserProfile
	UserProfile map[string]any

	// Labels 是用户级标签，可驱动整个 Pipeline 行为
	// 例如：新用户、重度用户、价格敏感等
	Labels map[string]utils.Label

	// Params 请求级上下文参数，包含：
	// - 请求参数：latitude, longitude, time_of_day, query, device_type 等
	// - 实时特征：realtime_ctr, realtime_exposure 等（建议加 realtime_ 前缀区分）
	Params map[string]any
}

// GetUserProfile 获取用户画像。
// 优先返回强类型 UserProfile，如果为空则从 UserProfile map 构建。
func (rctx *RecommendContext) GetUserProfile() *UserProfile {
	if rctx.User != nil {
		return rctx.User
	}
	// 从 UserProfile 构建
	if rctx.UserProfile != nil {
		user := NewUserProfile(rctx.UserID)
		// 提取静态属性
		if age, ok := rctx.UserProfile["age"].(float64); ok {
			user.Age = int(age)
		}
		if gender, ok := rctx.UserProfile["gender"].(string); ok {
			user.Gender = gender
		}
		if location, ok := rctx.UserProfile["location"].(string); ok {
			user.Location = location
		}
		// 提取兴趣
		if interests, ok := rctx.UserProfile["interests"].(map[string]float64); ok {
			user.Interests = interests
		}
		return user
	}
	return nil
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
