# config 配置驱动覆盖分析

配置驱动通过**注册机制**工作：各组件在 `init()` 中调用 `config.Register(typeName, builder)`，`config.DefaultFactory()` 返回包含所有已注册类型的 NodeFactory。使用前需在入口处 `import _ "github.com/rushteam/reckit/config/builders"` 以触发内置 Node 注册。覆盖情况如下。

## 一、已覆盖

| 类型 | 注册 key | 实现 | 说明 |
|------|-----------|------|------|
| Recall | `recall.fanout` | Fanout | 支持 sources 内 type: hot；merge_strategy: priority/union/first；dedup、timeout、max_concurrent |
| Recall | `recall.hot` | Hot | ids |
| Recall | `recall.ann` | — | 仅注册，build 返回错误「ann node not fully implemented」 |
| Rank | `rank.lr` | LRNode + LRModel | weights、bias |
| Rank | `rank.rpc` | RPCNode + RPCModel | endpoint、timeout、model_type |
| Rank | `rank.wide_deep` | WideDeepNode + RPCModel | endpoint、timeout、model_type |
| Rank | `rank.two_tower` | TwoTowerNode + RPCModel | endpoint、timeout、model_type |
| Rank | `rank.dnn` | DNNNode + RPCModel | endpoint、timeout、model_type |
| Rank | `rank.din` | DINNode + RPCModel | endpoint、timeout、model_type、max_behavior_seq_len |
| ReRank | `rerank.diversity` | Diversity | label_key |
| ReRank | `rerank.mmoe` | MMoENode | endpoint、timeout、weight_ctr、weight_watch_time、weight_gmv、strip_feature_prefix |
| ReRank | `rerank.topn` | TopNNode | top_n 或 n |
| Filter | `filter` | FilterNode | filters 内 type: blacklist / user_block / exposed（storeAdapter 均为 nil） |
| Feature | `feature.enrich` | EnrichNode | 仅前缀配置；未配置 FeatureService / 提取器 |

## 二、未覆盖的库内实现

### Recall（Fanout 内 source type 仅 hot / ann 占位）

- **U2IRecall / I2IRecall**：需 Store（CF）、SimilarityCalculator、Config 等，未在 factory 中从配置构建。
- **ContentRecall**：需 ContentStore、TopK、Metric、UserPreferencesExtractor。
- **Word2VecRecall / BERTRecall**：需 Model、Store、TopK、Mode 等。
- **TwoTowerRecall**：需 FeatureService、UserTowerService(MLService)、VectorService、TopK、Collection 等。
- **YouTubeDNNRecall / DSSMRecall / GraphRecall / RPCRecall**：需 Endpoint 或 HTTP 客户端。
- **MFRecall（矩阵分解）**：需 Store。
- **UserHistory**：需 UserHistoryStore、KeyPrefix、BehaviorType、TopK、TimeWindow、EnableSimilarExtend。

### Rank

- **rank.wide_deep / rank.two_tower / rank.dnn / rank.din**：已支持，通过 endpoint、timeout、model_type（及 din 的 max_behavior_seq_len）从配置构建，底层使用 RPCModel 调用远程服务。

### Filter

- 黑名单/曝光/用户拉黑从配置可建 FilterNode，但 **StoreAdapter 固定为 nil**，仅适合纯内存 ID 列表；若要从 Store 读黑名单/曝光/拉黑，需在代码里注入 StoreAdapter，无法仅靠当前 config 完成。

### Feature

- **feature.enrich** 只从配置填了前缀，**未配置 FeatureService、UserFeatureExtractor、ItemFeatureExtractor**，实际做特征注入需在代码里注入这些依赖。

## 三、小结

| 维度 | 已覆盖 | 未覆盖 / 部分 |
|------|--------|----------------|
| Recall 顶层 Node | fanout、hot；ann 占位 | — |
| Fanout 内 Source 类型 | hot | u2i、i2i、content、word2vec、bert、two_tower、youtube_dnn、dssm、graph、rpc、mf、user_history |
| Rank | lr、rpc、wide_deep、two_tower、dnn、din | — |
| ReRank | diversity、mmoe、topn | — |
| Filter | blacklist、user_block、exposed（无 Store） | 需 Store 时需代码注入 |
| Feature | enrich（仅前缀） | FeatureService / 提取器需代码注入 |

结论：**Rank、ReRank 已全部支持 config builder**。适合用配置驱动的有：Fanout（hot 源）、Hot、LR/RPC/WideDeep/TwoTower/DNN/DIN 排序、Diversity/MMoE/TopN 重排、无 Store 的 Filter 和仅前缀的 Enrich。未覆盖的主要为：Recall 内需 Store/MLService/VectorService 的源、需 Store 的 Filter、需注入 FeatureService 的 Enrich；这些需在代码中构造并注入依赖，或扩展 factory 的 build 逻辑与配置约定。
