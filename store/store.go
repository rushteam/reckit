package store

// 注意：此包只包含实现，接口定义在 core 和 vector 包。
//
// 内存实现（平替第三方 SDK）：
//   - MemoryStore: 实现 core.Store 和 core.KeyValueStore（平替 Redis）
//   - MemoryVectorService: 实现 core.VectorService 和 core.VectorDatabaseService（平替 Milvus）
//
// 使用示例：
//   // 存储服务
//   var store core.Store = NewMemoryStore()
//   var kvStore core.KeyValueStore = NewMemoryStore()
//
//   // 向量服务
//   var vectorService core.VectorService = NewMemoryVectorService()
//   var dbService core.VectorDatabaseService = NewMemoryVectorService()
//
// 生产环境实现已移至扩展包：
//   - Redis: go get github.com/rushteam/reckit/ext/store/redis
//   - Milvus: go get github.com/rushteam/reckit/ext/vector/milvus
