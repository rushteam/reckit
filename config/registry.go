package config

import (
	"fmt"
	"sort"
	"sync"

	"github.com/rushteam/reckit/pipeline"
)

// 使用配置驱动时，需在 main 或入口处 import _ "github.com/rushteam/reckit/config/builders"
// 以触发内置 Node（recall.fanout、recall.hot、rank.lr、rank.rpc 等）的 init 注册。

// NodeBuilder 与 pipeline.NodeBuilder 一致：根据 config 构建 Node。
// 各组件在 init 中调用 Register(typeName, builder) 即可被配置驱动。
type NodeBuilder = pipeline.NodeBuilder

var (
	defaultBuilders   = make(map[string]NodeBuilder)
	defaultBuildersMu sync.RWMutex
)

// Register 注册一种 Node 的构建逻辑，供 DefaultFactory 与配置驱动使用。
// 建议在各组件的 init 中调用，例如：func init() { config.Register("recall.hot", BuildHotNode) }
func Register(typeName string, builder NodeBuilder) {
	if typeName == "" || builder == nil {
		return
	}
	defaultBuildersMu.Lock()
	defer defaultBuildersMu.Unlock()
	defaultBuilders[typeName] = builder
}

// SupportedTypes 返回当前已注册的 Node 类型列表（排序），用于错误提示与校验。
func SupportedTypes() []string {
	defaultBuildersMu.RLock()
	defer defaultBuildersMu.RUnlock()
	types := make([]string, 0, len(defaultBuilders))
	for t := range defaultBuilders {
		types = append(types, t)
	}
	sort.Strings(types)
	return types
}

// DefaultFactory 返回基于当前注册表构建的 NodeFactory，包含所有通过 Register 注册的 Node 类型。
func DefaultFactory() *pipeline.NodeFactory {
	defaultBuildersMu.RLock()
	defer defaultBuildersMu.RUnlock()
	f := pipeline.NewNodeFactory()
	for typeName, builder := range defaultBuilders {
		f.Register(typeName, builder)
	}
	return f
}

// ValidatePipelineConfig 校验 pipeline 配置中所有 node 类型均已注册；若有未支持类型则返回包含已支持列表的错误。
func ValidatePipelineConfig(cfg *pipeline.Config) error {
	if cfg == nil {
		return nil
	}
	supported := SupportedTypes()
	for _, nc := range cfg.Pipeline.Nodes {
		if nc.Type == "" {
			continue
		}
		defaultBuildersMu.RLock()
		_, ok := defaultBuilders[nc.Type]
		defaultBuildersMu.RUnlock()
		if !ok {
			return fmt.Errorf("unsupported node type %q (supported: %v)", nc.Type, supported)
		}
	}
	return nil
}
