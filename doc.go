// Package reckit 是一个推荐系统工具包（Recommender Kit）。
//
// 设计要点：
// - Pipeline-first: 所有推荐逻辑通过 Node 串联（Recall → Rank → ReRank → PostProcess）
// - Labels-first: labels 全链路透传与标准化 merge，支持 explain / 观测 / 策略驱动
// - Node 可扩展: 自定义 Node 即可插拔扩展（本地或 RPC 模型均可）
package reckit

import "github.com/rushteam/reckit/pipeline"

// 轻量 facade：便于用户直接 import "reckit" 使用核心抽象。
type Pipeline = pipeline.Pipeline
type Node = pipeline.Node
type Kind = pipeline.Kind

const (
	KindRecall      = pipeline.KindRecall
	KindFilter      = pipeline.KindFilter
	KindRank        = pipeline.KindRank
	KindReRank      = pipeline.KindReRank
	KindPostProcess = pipeline.KindPostProcess
)
