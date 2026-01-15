package pipeline

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"

	"gopkg.in/yaml.v3"
)

// Config 是 Pipeline 的配置结构（支持 YAML/JSON）。
type Config struct {
	Pipeline struct {
		Name  string       `yaml:"name" json:"name"`
		Nodes []NodeConfig `yaml:"nodes" json:"nodes"`
	} `yaml:"pipeline" json:"pipeline"`
}

// NodeConfig 是单个 Node 的配置。
type NodeConfig struct {
	Type   string                 `yaml:"type" json:"type"`     // recall.fanout / rank.lr / rerank.diversity 等
	Config map[string]interface{} `yaml:"config" json:"config"` // Node 特定配置
}

// LoadFromYAML 从 YAML 文件加载 Pipeline 配置。
func LoadFromYAML(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse yaml: %w", err)
	}

	return &cfg, nil
}

// LoadFromJSON 从 JSON 文件加载 Pipeline 配置。
func LoadFromJSON(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse json: %w", err)
	}

	return &cfg, nil
}

// BuildPipeline 根据配置构建 Pipeline（需要 NodeFactory 注册 Node 构建器）。
// 注意：factory 应该在独立的 config 包中，避免循环依赖。
func (c *Config) BuildPipeline(factory *NodeFactory) (*Pipeline, error) {
	nodes := make([]Node, 0, len(c.Pipeline.Nodes))

	for _, nc := range c.Pipeline.Nodes {
		node, err := factory.Build(nc.Type, nc.Config)
		if err != nil {
			return nil, fmt.Errorf("build node %s: %w", nc.Type, err)
		}
		nodes = append(nodes, node)
	}

	return &Pipeline{Nodes: nodes}, nil
}

// NodeBuilder 是 Node 构建器函数类型。
type NodeBuilder func(map[string]interface{}) (Node, error)

// NodeFactory 用于根据配置构建 Node 实例。
// 支持线程安全的动态注册，用户可以在运行时注册自定义 Node 类型。
type NodeFactory struct {
	builders map[string]NodeBuilder
	mutex    sync.RWMutex
}

func NewNodeFactory() *NodeFactory {
	return &NodeFactory{
		builders: make(map[string]NodeBuilder),
	}
}

// Register 注册 Node 构建器（线程安全）。
// 用户可以在运行时注册自定义 Node 类型，无需修改库代码。
//
// 示例：
//   factory := pipeline.NewNodeFactory()
//   factory.Register("my.custom.node", func(config map[string]interface{}) (pipeline.Node, error) {
//       // 构建自定义 Node
//       return &MyCustomNode{}, nil
//   })
func (f *NodeFactory) Register(nodeType string, builder NodeBuilder) {
	f.mutex.Lock()
	defer f.mutex.Unlock()
	f.builders[nodeType] = builder
}

// Unregister 取消注册 Node 构建器（线程安全）。
func (f *NodeFactory) Unregister(nodeType string) {
	f.mutex.Lock()
	defer f.mutex.Unlock()
	delete(f.builders, nodeType)
}

// Build 根据类型和配置构建 Node（线程安全）。
func (f *NodeFactory) Build(nodeType string, config map[string]interface{}) (Node, error) {
	f.mutex.RLock()
	builder, ok := f.builders[nodeType]
	f.mutex.RUnlock()
	
	if !ok {
		return nil, fmt.Errorf("unknown node type: %s", nodeType)
	}
	return builder(config)
}

// ListRegisteredTypes 返回所有已注册的 Node 类型（线程安全）。
func (f *NodeFactory) ListRegisteredTypes() []string {
	f.mutex.RLock()
	defer f.mutex.RUnlock()
	
	types := make([]string, 0, len(f.builders))
	for t := range f.builders {
		types = append(types, t)
	}
	return types
}
