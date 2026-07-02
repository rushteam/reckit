package core

import "context"

// PrecomputedSimilarStore 预计算 item-to-item 相似度数据访问接口。
//
// 适用于离线预计算的 I2I / Swing / Session 等召回场景：
//   - 离线阶段：通过协同过滤、Swing、矩阵分解等算法计算 item 间相似度，写入存储（如 Redis ZSET）
//   - 在线阶段：通过此接口批量读取预计算结果，避免实时计算开销
//
// 实现参考：
//   - Redis ZSET：key = {prefix}:{item_id}，member = similar_item_id，score = similarity
//   - 本地缓存：适合静态 / 低频更新的相似度数据
type PrecomputedSimilarStore interface {
	// BatchGetSimilar 批量获取预计算的相似物品列表。
	//
	// keys: 查询 key 列表（通常为 {prefix}:{item_id}），由调用方拼接。
	// topK: 每个 key 最多返回 topK 个相似物品。
	//
	// 返回与 keys 等长的二维切片，results[i] 对应 keys[i] 的相似物品（按分数降序）。
	// 某个 key 不存在时对应位置返回空切片而非 error。
	BatchGetSimilar(ctx context.Context, keys []string, topK int) ([][]ScoredMember, error)
}

// VersionedSimilarStore 扩展 PrecomputedSimilarStore，支持版本化数据访问。
//
// 离线 I2I 数据通常按日期版本更新（如 cf:similar:v2026-03-07:{item_id}），
// 通过 GetActiveVersion 获取当前活跃版本号，动态拼接查询 key。
//
// 未实现此接口时，PrecomputedI2IRecall 会直接使用配置的静态前缀。
type VersionedSimilarStore interface {
	PrecomputedSimilarStore
	// GetActiveVersion 获取当前活跃数据版本（如 "2026-03-07"）。
	// 返回空字符串表示版本不可用。
	GetActiveVersion(ctx context.Context, versionKey string) (string, error)
}
