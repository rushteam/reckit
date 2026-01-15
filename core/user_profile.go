package core

import "time"

// UserProfile 是用户画像的核心抽象。
//
// 一句话定义：用户画像 = 推荐 Pipeline 的"全局上下文 + 特征源 + 决策信号"
//
// 它不是某一个 Node，而是：
//  - 被所有 Node 共享
//  - 驱动 Recall / Rank / ReRank
//  - 可以被 Label 打标、回写、持续演进
//
// 设计要点：
//  维度          作用
//  静态属性      冷启动 / 基础过滤
//  长期兴趣      Recall / Rank 核心
//  短期行为      实时调权
//  实验桶        策略切换
//  可更新        Online Learning
type UserProfile struct {
	UserID int64

	// 静态属性（冷启动 / 基础过滤）
	Gender   string // male / female / unknown
	Age      int    // 年龄
	Location string // 地理位置

	// 兴趣画像（长期）- Recall / Rank 核心
	// key: category/tag，value: weight (0-1)
	Interests map[string]float64

	// 行为统计（短期）- 实时调权
	RecentClicks   []int64 // 最近点击的物品 ID
	RecentImpress []int64 // 最近曝光的物品 ID

	// 偏好信号
	PreferTags map[string]float64 // 标签偏好

	// 控制与实验（策略切换）
	Buckets map[string]string // AB / 实验桶，例如 {"diversity": "strong", "recall": "v2"}

	// 元数据
	UpdateTime time.Time // 最后更新时间
}

// NewUserProfile 创建一个新的用户画像。
func NewUserProfile(userID int64) *UserProfile {
	return &UserProfile{
		UserID:        userID,
		Interests:     make(map[string]float64),
		RecentClicks:  make([]int64, 0),
		RecentImpress: make([]int64, 0),
		PreferTags:    make(map[string]float64),
		Buckets:       make(map[string]string),
		UpdateTime:    time.Now(),
	}
}

// UpdateInterest 更新用户兴趣（支持 Online Learning）。
func (p *UserProfile) UpdateInterest(category string, weight float64) {
	if p.Interests == nil {
		p.Interests = make(map[string]float64)
	}
	p.Interests[category] = weight
	p.UpdateTime = time.Now()
}

// AddRecentClick 添加最近点击记录。
func (p *UserProfile) AddRecentClick(itemID int64, maxSize int) {
	if p.RecentClicks == nil {
		p.RecentClicks = make([]int64, 0)
	}
	// 去重
	for _, id := range p.RecentClicks {
		if id == itemID {
			return
		}
	}
	p.RecentClicks = append(p.RecentClicks, itemID)
	// 限制大小
	if maxSize > 0 && len(p.RecentClicks) > maxSize {
		p.RecentClicks = p.RecentClicks[len(p.RecentClicks)-maxSize:]
	}
	p.UpdateTime = time.Now()
}

// AddRecentImpress 添加最近曝光记录。
func (p *UserProfile) AddRecentImpress(itemID int64, maxSize int) {
	if p.RecentImpress == nil {
		p.RecentImpress = make([]int64, 0)
	}
	// 去重
	for _, id := range p.RecentImpress {
		if id == itemID {
			return
		}
	}
	p.RecentImpress = append(p.RecentImpress, itemID)
	// 限制大小
	if maxSize > 0 && len(p.RecentImpress) > maxSize {
		p.RecentImpress = p.RecentImpress[len(p.RecentImpress)-maxSize:]
	}
	p.UpdateTime = time.Now()
}

// SetBucket 设置实验桶。
func (p *UserProfile) SetBucket(key, value string) {
	if p.Buckets == nil {
		p.Buckets = make(map[string]string)
	}
	p.Buckets[key] = value
}

// GetBucket 获取实验桶值。
func (p *UserProfile) GetBucket(key string) string {
	if p.Buckets == nil {
		return ""
	}
	return p.Buckets[key]
}

// HasInterest 检查用户是否有某个兴趣。
func (p *UserProfile) HasInterest(category string, threshold float64) bool {
	if p.Interests == nil {
		return false
	}
	weight, ok := p.Interests[category]
	if !ok {
		return false
	}
	return weight >= threshold
}

// GetInterestWeight 获取兴趣权重。
func (p *UserProfile) GetInterestWeight(category string) float64 {
	if p.Interests == nil {
		return 0
	}
	return p.Interests[category]
}
