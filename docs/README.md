# Reckit 文档

Reckit 是一个工业级推荐系统工具库，采用 **Pipeline + Node** 架构，通过接口抽象实现高度可扩展性。

## 快速导航

### 入门

| 文档 | 说明 |
|------|------|
| [架构设计](./ARCHITECTURE.md) | 整体架构、设计原则、核心概念 |
| [扩展指南](./EXTENSIBILITY.md) | 如何扩展自定义召回、排序、过滤等 |

### 召回

| 文档 | 说明 |
|------|------|
| [召回算法](./RECALL_ALGORITHMS.md) | 所有召回算法的使用方法 |
| [协同过滤](./COLLABORATIVE_FILTERING.md) | User-CF、Item-CF 详解 |
| [双塔模型](./TWO_TOWER_GUIDE.md) | Two-Tower 召回搭建指南 |
| [Word2Vec/Item2Vec](./WORD2VEC_ITEM2VEC.md) | 词向量/物品向量召回 |

### 排序

| 文档 | 说明 |
|------|------|
| [排序模型](./RANK_MODELS.md) | DNN、Wide&Deep、DIN 等排序模型 |

### 特征

| 文档 | 说明 |
|------|------|
| [Feature 模块](./FEATURE_MODULE.md) | Extractor/Service/EnrichNode 职责与使用 |
| [特征处理](./FEATURE_PROCESSING.md) | 归一化、编码、特征工程 |

### Python 服务

| 文档 | 说明 |
|------|------|
| [模型服务协议](./MODEL_SERVICE_PROTOCOL.md) | Python 服务 API 规范 |
| [生产环境指南](./PYTHON_PRODUCTION_GUIDE.md) | 可观测性、可运维性建议 |
| [训练自动化](./TRAINING_AUTOMATION.md) | 自动化训练流水线 |

## 目录结构

```
docs/
├── README.md                  # 本文件，文档索引
├── ARCHITECTURE.md            # 架构设计
├── EXTENSIBILITY.md           # 扩展指南
├── RECALL_ALGORITHMS.md       # 召回算法
├── COLLABORATIVE_FILTERING.md # 协同过滤
├── TWO_TOWER_GUIDE.md         # 双塔模型
├── WORD2VEC_ITEM2VEC.md       # Word2Vec/Item2Vec
├── RANK_MODELS.md             # 排序模型
├── FEATURE_MODULE.md          # Feature 模块
├── FEATURE_PROCESSING.md      # 特征处理
├── MODEL_SERVICE_PROTOCOL.md  # 模型服务协议
├── PYTHON_PRODUCTION_GUIDE.md # Python 生产指南
└── TRAINING_AUTOMATION.md     # 训练自动化
```

## 相关资源

- [项目 README](../README.md) - 项目概述和快速开始
- [CLAUDE.md](../CLAUDE.md) - AI Coding 指南（完整 API 参考）
- [examples/](../examples/) - 示例代码
