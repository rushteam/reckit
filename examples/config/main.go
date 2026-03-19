package main

import (
	"context"
	"fmt"
	"time"

	"github.com/rushteam/reckit/config/builders"
	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/feature"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/conv"
	"github.com/rushteam/reckit/store"
)

// ---------------------------------------------------------------------------
// 示例：自定义 Node + 依赖注入
// ---------------------------------------------------------------------------

// ItemRepo 是业务侧的外部依赖（如 DB、Redis、RPC client 等）。
type ItemRepo interface {
	FetchBoostItems(ctx context.Context, scene string) ([]string, error)
}

// memItemRepo 是 ItemRepo 的内存实现（仅供演示）。
type memItemRepo struct{ items []string }

func (r *memItemRepo) FetchBoostItems(_ context.Context, _ string) ([]string, error) {
	return r.items, nil
}

// BoostNode 是一个自定义 Node：从 ItemRepo 拉取加权物品，对命中者提升分数。
type BoostNode struct {
	Repo      ItemRepo
	BoostRate float64
}

func (n *BoostNode) Name() string        { return "custom.boost" }
func (n *BoostNode) Kind() pipeline.Kind { return pipeline.KindReRank }
func (n *BoostNode) Process(ctx context.Context, rctx *core.RecommendContext, items []*core.Item) ([]*core.Item, error) {
	boostIDs, err := n.Repo.FetchBoostItems(ctx, rctx.Scene)
	if err != nil {
		return items, nil
	}
	set := make(map[string]struct{}, len(boostIDs))
	for _, id := range boostIDs {
		set[id] = struct{}{}
	}
	for _, item := range items {
		if _, ok := set[item.ID]; ok {
			item.Score *= n.BoostRate
		}
	}
	return items, nil
}

// registerBoostNode 将 BoostNode 注册到指定工厂（实例级注册）。
// 闭包捕获外部依赖（repo），YAML config 字段通过 builder 参数透传。
func registerBoostNode(factory *pipeline.NodeFactory, repo ItemRepo) {
	factory.Register("custom.boost", func(cfg map[string]interface{}) (pipeline.Node, error) {
		return &BoostNode{
			Repo:      repo,
			BoostRate: conv.ConfigGet(cfg, "boost_rate", 1.5),
		}, nil
	})
}

// ---------------------------------------------------------------------------

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 1. 初始化外部依赖（Redis、DB、RPC client 等）
	repo := &memItemRepo{items: []string{"1", "3"}}
	memStore := store.NewMemoryStore()
	defer memStore.Close(ctx)

	// 2. 创建实例级 factory，并绑定外部依赖。
	provider := feature.NewStoreFeatureProvider(memStore, feature.KeyPrefix{})
	featureService := feature.NewBaseFeatureService(provider)
	defer featureService.Close(ctx)

	factory := builders.NewFactory(builders.Dependencies{
		FilterStore:    memStore,
		FeatureService: featureService,
	})

	// 3. 注册自定义 Node —— 闭包捕获依赖，config 字段由 YAML 透传
	registerBoostNode(factory, repo)

	// 4. 加载 YAML 配置
	cfg, err := pipeline.LoadFromYAML("examples/config/pipeline.example.yaml")
	if err != nil {
		fmt.Printf("加载配置失败: %v\n", err)
		return
	}

	// 5. 用实例 factory 构建 Pipeline（包含内置 + 自定义注册的所有 Node）
	p, err := cfg.BuildPipeline(factory)
	if err != nil {
		fmt.Printf("构建 Pipeline 失败: %v\n", err)
		return
	}

	rctx := &core.RecommendContext{
		UserID: "42",
		Scene:  "feed",
		Params: map[string]any{
			"debug": true,
			"hour":  time.Now().Hour(),
		},
	}

	items, err := p.Run(ctx, rctx, nil)
	if err != nil {
		fmt.Printf("运行 Pipeline 失败: %v\n", err)
		return
	}

	fmt.Printf("Pipeline 执行成功，返回 %d 个物品:\n", len(items))
	for i, it := range items {
		fmt.Printf("#%d id=%s score=%.4f labels=%v\n", i, it.ID, it.Score, it.Labels)
	}
}
