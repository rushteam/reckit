package core

import "reckit/pkg/utils"

// RecommendContext 承载用户/场景/实时信息，贯穿整个 Pipeline 透传。
//
// 关键升级：引入 UserProfile 和 Labels
//   - UserProfile: 强类型用户画像（推荐使用）
//   - UserProfileMap: 向后兼容的 map 形式（保留）
//   - Labels: 用户级标签，可驱动整个 Pipeline 行为
type RecommendContext struct {
	UserID   int64
	DeviceID string
	Scene    string

	// UserProfile 是强类型用户画像（推荐使用）
	User *UserProfile

	// UserProfileMap 是向后兼容的 map 形式（保留，用于快速原型）
	// 如果 User 不为空，优先使用 User；否则使用 UserProfileMap
	UserProfile map[string]any

	// Labels 是用户级标签，可驱动整个 Pipeline 行为
	// 例如：新用户、重度用户、价格敏感等
	Labels map[string]utils.Label

	Realtime map[string]any
	Params   map[string]any
}

// GetUserProfile 获取用户画像（兼容方法）。
// 优先返回强类型 UserProfile，如果为空则从 UserProfileMap 构建。
func (rctx *RecommendContext) GetUserProfile() *UserProfile {
	if rctx.User != nil {
		return rctx.User
	}
	// 从 UserProfileMap 构建（向后兼容）
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
