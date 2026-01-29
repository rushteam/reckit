# Reckit 文档索引

本文档提供 Reckit 项目文档的索引和导航。

---

## 📚 快速开始

- [架构设计](./ARCHITECTURE.md) - 了解 Reckit 的整体架构和设计原则
- [Pipeline 功能特性](./PIPELINE_FEATURES.md) - 了解 Pipeline 支持的功能
- [接口与实现完整分析](./INTERFACES_AND_IMPLEMENTATIONS.md) - 查看所有接口的详细定义

---

## 🎯 核心功能文档

### 召回（Recall）
- [召回算法文档](./RECALL_ALGORITHMS.md) - 所有召回算法的介绍
- [Word2Vec / Item2Vec](./WORD2VEC_ITEM2VEC.md) - 文本与物品序列向量、Python 训练、Golang 接入
- [协同过滤](./COLLABORATIVE_FILTERING.md) - 协同过滤算法详解
- [Embedding 能力抽象](./EMBEDDING_ABSTRACT.md) - Embedding 向量召回详解

### 排序（Rank）
- [排序模型文档](./RANK_MODELS.md) - 所有排序模型的介绍
- [双塔模型指南](./TWO_TOWER_GUIDE.md) - 双塔模型的搭建和使用
- [模型选型](./MODEL_SELECTION.md) - 如何选择合适的模型

### 特征（Feature）
- [特征处理](./FEATURE_PROCESSING.md) - 特征处理工具类（归一化、编码等）
- [特征一致性](./FEATURE_CONSISTENCY.md) - 训练与在线特征一致性
- [编码器接口设计](./ENCODER_INTERFACE_DESIGN.md) - 编码器接口设计说明

### 用户画像
- [用户画像文档](./USER_PROFILE.md) - 用户画像的使用和管理

### 训练与部署
- [模型服务协议约束](./MODEL_SERVICE_PROTOCOL.md) - Python 模型服务必遵协议（TorchServe 传输、TorchServeClient、框架接口区分）
- [训练流程自动化](./TRAINING_AUTOMATION.md) - 工业级训练流程自动化指南（数据、编排、版本、发布、监控）
- [Python 生产工业级补充建议](./PYTHON_PRODUCTION_GUIDE.md) - 推理服务与训练的可观测性、安全、可靠性、CI/CD 等补充建议

---

## 🔧 扩展和开发

- [可扩展性指南](./EXTENSIBILITY.md) - 如何扩展 Reckit 功能
- [接口与实现完整分析](./INTERFACES_AND_IMPLEMENTATIONS.md) - 所有接口的详细定义
- [pkg/conv 类型转换与泛型工具](../pkg/conv/README.md) - 类型转换、ConfigGet、MapToFloat64、GetExtraAs 等

---

## 📋 规划和问题

- [开发路线图](./ROADMAP.md) - 未来发展规划
- [TODO 清单](./TODO.md) - 待解决问题清单

---

## 📖 文档结构说明

### 核心文档（必读）
1. **ARCHITECTURE.md** - 架构设计，了解整体设计
2. **PIPELINE_FEATURES.md** - Pipeline 功能特性，了解支持的功能
3. **INTERFACES_AND_IMPLEMENTATIONS.md** - 接口参考手册，查找接口定义

### 功能文档（按需阅读）
- 召回、排序、特征等各模块的详细文档
- 根据实际使用需求选择阅读

### 扩展文档（开发者）
- 可扩展性指南、编码器接口设计等
- 需要扩展功能时参考

### 规划文档（维护者）
- 路线图、TODO 清单
- 了解项目规划和待解决问题

---

## 🔍 文档查找指南

### 我想了解...
- **整体架构** → [ARCHITECTURE.md](./ARCHITECTURE.md)
- **Pipeline 功能** → [PIPELINE_FEATURES.md](./PIPELINE_FEATURES.md)
- **如何扩展功能** → [EXTENSIBILITY.md](./EXTENSIBILITY.md)
- **Word2Vec / Item2Vec 与 Python 训练** → [WORD2VEC_ITEM2VEC.md](./WORD2VEC_ITEM2VEC.md)
- **类型转换 / 泛型工具** → [pkg/conv](../pkg/conv/README.md)
- **接口定义** → [INTERFACES_AND_IMPLEMENTATIONS.md](./INTERFACES_AND_IMPLEMENTATIONS.md)
- **召回算法** → [RECALL_ALGORITHMS.md](./RECALL_ALGORITHMS.md)
- **排序模型** → [RANK_MODELS.md](./RANK_MODELS.md)
- **特征处理** → [FEATURE_PROCESSING.md](./FEATURE_PROCESSING.md)
- **Embedding 召回** → [EMBEDDING_ABSTRACT.md](./EMBEDDING_ABSTRACT.md)
- **双塔模型** → [TWO_TOWER_GUIDE.md](./TWO_TOWER_GUIDE.md)
- **训练流程自动化** → [TRAINING_AUTOMATION.md](./TRAINING_AUTOMATION.md)
- **Python 生产工业级补充建议** → [PYTHON_PRODUCTION_GUIDE.md](./PYTHON_PRODUCTION_GUIDE.md)
- **未来规划** → [ROADMAP.md](./ROADMAP.md)
