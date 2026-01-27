package feedback

import (
	"context"

	"github.com/rushteam/reckit/core"
)

// FeedbackType 反馈类型
type FeedbackType string

const (
	FeedbackTypeImpression FeedbackType = "impression" // 曝光
	FeedbackTypeClick      FeedbackType = "click"      // 点击
	FeedbackTypeConversion FeedbackType = "conversion" // 转化
	FeedbackTypeSkip       FeedbackType = "skip"       // 跳过
	FeedbackTypeLike       FeedbackType = "like"       // 点赞
	FeedbackTypeDislike    FeedbackType = "dislike"    // 不喜欢
)

// FeedbackEvent 反馈事件（轻量级，只包含必要信息）
type FeedbackEvent struct {
	UserID    string            `json:"user_id"`
	ItemID    string            `json:"item_id"`
	Scene     string            `json:"scene"`
	Type      FeedbackType      `json:"type"`
	Timestamp int64             `json:"timestamp"` // Unix 时间戳（秒）
	Position  int               `json:"position"`  // 物品在列表中的位置
	Score     float64           `json:"score"`     // 排序分数
	Labels    map[string]string `json:"labels"`    // 召回来源等标签
	Extras    map[string]any    `json:"extras,omitempty"`
}

// Collector 反馈收集器接口（异步非阻塞）
type Collector interface {
	// RecordImpression 异步记录曝光（不阻塞）
	RecordImpression(ctx context.Context, rctx *core.RecommendContext, items []*core.Item) error

	// RecordClick 异步记录点击
	RecordClick(ctx context.Context, rctx *core.RecommendContext, itemID string, position int) error

	// RecordConversion 异步记录转化
	RecordConversion(ctx context.Context, rctx *core.RecommendContext, itemID string, extras map[string]any) error

	// Close 优雅关闭（等待缓冲数据发送完成）
	Close() error
}
