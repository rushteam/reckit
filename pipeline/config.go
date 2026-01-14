package pipeline

import (
	"encoding/json"
	"fmt"
	"os"

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

// NodeFactory 用于根据配置构建 Node 实例。
type NodeFactory struct {
	builders map[string]func(map[string]interface{}) (Node, error)
}

func NewNodeFactory() *NodeFactory {
	return &NodeFactory{
		builders: make(map[string]func(map[string]interface{}) (Node, error)),
	}
}

// Register 注册 Node 构建器。
func (f *NodeFactory) Register(nodeType string, builder func(map[string]interface{}) (Node, error)) {
	f.builders[nodeType] = builder
}

// Build 根据类型和配置构建 Node。
func (f *NodeFactory) Build(nodeType string, config map[string]interface{}) (Node, error) {
	builder, ok := f.builders[nodeType]
	if !ok {
		return nil, fmt.Errorf("unknown node type: %s", nodeType)
	}
	return builder(config)
}
