package config

import (
	"github.com/rushteam/reckit/config/builders"
	"github.com/rushteam/reckit/pipeline"
)

// BuiltinDependencies 是内置 builders 的依赖集合别名。
type BuiltinDependencies = builders.Dependencies

// RegisterBuiltinsWithDeps 将内置 builders 注册到全局注册表（带依赖绑定）。
// 该函数可重复调用，后注册会覆盖同名 builder。
func RegisterBuiltinsWithDeps(deps BuiltinDependencies) {
	Register("recall.fanout", builders.BuildFanoutNode)
	Register("recall.hot", builders.BuildSortedSetNode)
	Register("recall.sorted_set", builders.BuildSortedSetNode)
	Register("recall.ann", builders.BuildANNNode)
	Register("rank.lr", builders.BuildLRNode)
	Register("rank.rpc", builders.BuildRPCNode)
	Register("rank.wide_deep", builders.BuildWideDeepNode)
	Register("rank.two_tower", builders.BuildTwoTowerNode)
	Register("rank.dnn", builders.BuildDNNNode)
	Register("rank.din", builders.BuildDINNode)
	Register("rerank.diversity", builders.BuildDiversityNode)
	Register("rerank.mmoe", builders.BuildMMoENode)
	Register("rerank.topn", builders.BuildTopNNode)
	Register("filter", func(cfg map[string]interface{}) (pipeline.Node, error) {
		return builders.BuildFilterNodeWithDependencies(cfg, deps)
	})
	Register("feature.enrich", func(cfg map[string]interface{}) (pipeline.Node, error) {
		return builders.BuildFeatureEnrichNodeWithDependencies(cfg, deps)
	})
}

// NewFactoryWithBuiltins 创建包含内置 builders 的实例级工厂（推荐）。
func NewFactoryWithBuiltins(deps BuiltinDependencies) *pipeline.NodeFactory {
	return builders.NewFactory(deps)
}

func init() {
	// 默认注册一份“零依赖”内置 builders，保持历史行为兼容。
	RegisterBuiltinsWithDeps(BuiltinDependencies{})
}
