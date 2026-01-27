package main

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"log"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/feature"
	"github.com/rushteam/reckit/model"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/rank"
	"github.com/rushteam/reckit/recall"
	"github.com/rushteam/reckit/store"
)

// FeatureVersionMetadata 特征版本元数据
type FeatureVersionMetadata struct {
	Version     string
	CreatedAt   time.Time
	Description string
	Features    []string // 该版本包含的特征列表
	Status      string   // active, deprecated, archived
}

// FeatureVersionRegistry 特征版本注册表
type FeatureVersionRegistry struct {
	versions map[string]*FeatureVersionMetadata
}

// NewFeatureVersionRegistry 创建版本注册表
func NewFeatureVersionRegistry() *FeatureVersionRegistry {
	return &FeatureVersionRegistry{
		versions: make(map[string]*FeatureVersionMetadata),
	}
}

// Register 注册版本
func (r *FeatureVersionRegistry) Register(metadata *FeatureVersionMetadata) {
	r.versions[metadata.Version] = metadata
}

// GetVersion 获取版本信息
func (r *FeatureVersionRegistry) GetVersion(version string) (*FeatureVersionMetadata, error) {
	meta, ok := r.versions[version]
	if !ok {
		return nil, fmt.Errorf("version %s not found", version)
	}
	return meta, nil
}

// ListVersions 列出所有版本
func (r *FeatureVersionRegistry) ListVersions() []string {
	versions := make([]string, 0, len(r.versions))
	for v := range r.versions {
		versions = append(versions, v)
	}
	return versions
}

// VersionConfig 版本配置
type VersionConfig struct {
	DefaultVersion string            // 默认版本
	TrafficSplit   map[string]float64 // 流量分配，例如 {"v2": 0.1, "v1": 0.9}
}

// VersionedFeatureService 版本化特征服务
// 支持多版本特征管理、灰度发布、版本降级
type VersionedFeatureService struct {
	services map[string]feature.FeatureService
	config   *VersionConfig
	registry *FeatureVersionRegistry
}

// NewVersionedFeatureService 创建版本化特征服务
func NewVersionedFeatureService(
	services map[string]feature.FeatureService,
	config *VersionConfig,
	registry *FeatureVersionRegistry,
) *VersionedFeatureService {
	return &VersionedFeatureService{
		services: services,
		config:   config,
		registry: registry,
	}
}

// Name 返回服务名称
func (v *VersionedFeatureService) Name() string {
	return "versioned_feature_service"
}

// GetUserFeatures 获取用户特征（根据版本选择）
func (v *VersionedFeatureService) GetUserFeatures(
	ctx context.Context,
	userID string,
) (map[string]float64, error) {
	version := v.selectVersion(userID)
	service, ok := v.services[version]
	if !ok {
		// 降级到默认版本
		service = v.services[v.config.DefaultVersion]
	}

	features, err := service.GetUserFeatures(ctx, userID)
	if err != nil {
		// 如果新版本失败，尝试降级到旧版本
		if version != v.config.DefaultVersion {
			if fallbackService, ok := v.services[v.config.DefaultVersion]; ok {
				log.Printf("版本 %s 获取失败，降级到 %s", version, v.config.DefaultVersion)
				return fallbackService.GetUserFeatures(ctx, userID)
			}
		}
		return nil, err
	}

	return features, nil
}

// BatchGetUserFeatures 批量获取用户特征
func (v *VersionedFeatureService) BatchGetUserFeatures(
	ctx context.Context,
	userIDs []string,
) (map[string]map[string]float64, error) {
	// 按版本分组
	versionGroups := make(map[string][]string)
	for _, userID := range userIDs {
		version := v.selectVersion(userID)
		versionGroups[version] = append(versionGroups[version], userID)
	}

	// 合并结果
	result := make(map[string]map[string]float64)
	for version, ids := range versionGroups {
		service, ok := v.services[version]
		if !ok {
			service = v.services[v.config.DefaultVersion]
		}

		features, err := service.BatchGetUserFeatures(ctx, ids)
		if err != nil {
			// 降级处理
			if version != v.config.DefaultVersion {
				if fallbackService, ok := v.services[v.config.DefaultVersion]; ok {
					features, err = fallbackService.BatchGetUserFeatures(ctx, ids)
				}
			}
			if err != nil {
				return nil, err
			}
		}

		for k, v := range features {
			result[k] = v
		}
	}

	return result, nil
}

// GetItemFeatures 获取物品特征
func (v *VersionedFeatureService) GetItemFeatures(
	ctx context.Context,
	itemID string,
) (map[string]float64, error) {
	// 物品特征通常使用默认版本（或根据配置）
	version := v.config.DefaultVersion
	service, ok := v.services[version]
	if !ok {
		return nil, fmt.Errorf("default version %s not found", version)
	}

	return service.GetItemFeatures(ctx, itemID)
}

// BatchGetItemFeatures 批量获取物品特征
func (v *VersionedFeatureService) BatchGetItemFeatures(
	ctx context.Context,
	itemIDs []string,
) (map[string]map[string]float64, error) {
	version := v.config.DefaultVersion
	service, ok := v.services[version]
	if !ok {
		return nil, fmt.Errorf("default version %s not found", version)
	}

	return service.BatchGetItemFeatures(ctx, itemIDs)
}

// GetRealtimeFeatures 获取实时特征
func (v *VersionedFeatureService) GetRealtimeFeatures(
	ctx context.Context,
	userID, itemID string,
) (map[string]float64, error) {
	version := v.selectVersion(userID)
	service, ok := v.services[version]
	if !ok {
		service = v.services[v.config.DefaultVersion]
	}

	return service.GetRealtimeFeatures(ctx, userID, itemID)
}

// BatchGetRealtimeFeatures 批量获取实时特征
func (v *VersionedFeatureService) BatchGetRealtimeFeatures(
	ctx context.Context,
	pairs []feature.UserItemPair,
) (map[feature.UserItemPair]map[string]float64, error) {
	version := v.config.DefaultVersion
	service, ok := v.services[version]
	if !ok {
		return nil, fmt.Errorf("default version %s not found", version)
	}

	return service.BatchGetRealtimeFeatures(ctx, pairs)
}

// Close 关闭服务
func (v *VersionedFeatureService) Close(ctx context.Context) error {
	for _, service := range v.services {
		if err := service.Close(ctx); err != nil {
			return err
		}
	}
	return nil
}

// selectVersion 根据用户 ID 选择版本（用于灰度发布）
func (v *VersionedFeatureService) selectVersion(userID string) string {
	if len(v.config.TrafficSplit) == 0 {
		return v.config.DefaultVersion
	}

	// 使用哈希分配流量
	hash := hashString(userID)
	ratio := float64(hash%100) / 100.0

	cumulative := 0.0
	for version, split := range v.config.TrafficSplit {
		cumulative += split
		if ratio < cumulative {
			return version
		}
	}

	return v.config.DefaultVersion
}

// hashString 计算字符串哈希值
func hashString(s string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(s))
	return h.Sum32()
}

// createVersionedFeatureService 创建版本化特征服务
func createVersionedFeatureService(store core.Store) *VersionedFeatureService {
	// 创建版本注册表
	registry := NewFeatureVersionRegistry()

	// 注册 v1 版本
	registry.Register(&FeatureVersionMetadata{
		Version:     "v1",
		CreatedAt:   time.Now().AddDate(0, -1, 0), // 1个月前
		Description: "基础特征版本：年龄、性别、城市",
		Features:    []string{"age", "gender", "city"},
		Status:      "active",
	})

	// 注册 v2 版本
	registry.Register(&FeatureVersionMetadata{
		Version:     "v2",
		CreatedAt:   time.Now(),
		Description: "增强特征版本：新增兴趣标签、行为统计",
		Features:    []string{"age", "gender", "city", "interests", "click_count", "view_count"},
		Status:      "active",
	})

	// 创建 v1 版本的特征服务
	v1KeyPrefix := feature.KeyPrefix{
		User:     "user:features:v1:",
		Item:     "item:features:v1:",
		Realtime: "realtime:features:v1:",
	}
	v1Provider := feature.NewStoreFeatureProvider(store, v1KeyPrefix)
	v1Service := feature.NewBaseFeatureService(v1Provider)

	// 创建 v2 版本的特征服务
	v2KeyPrefix := feature.KeyPrefix{
		User:     "user:features:v2:",
		Item:     "item:features:v2:",
		Realtime: "realtime:features:v2:",
	}
	v2Provider := feature.NewStoreFeatureProvider(store, v2KeyPrefix)
	v2Service := feature.NewBaseFeatureService(v2Provider)

	// 配置版本切换策略
	config := &VersionConfig{
		DefaultVersion: "v2",
		TrafficSplit: map[string]float64{
			"v2": 0.7, // 70% 流量使用 v2
			"v1": 0.3, // 30% 流量使用 v1（灰度）
		},
	}

	// 创建版本化特征服务
	versionedService := NewVersionedFeatureService(
		map[string]feature.FeatureService{
			"v1": v1Service,
			"v2": v2Service,
		},
		config,
		registry,
	)

	return versionedService
}

// prepareVersionedFeatureData 准备不同版本的特征数据
func prepareVersionedFeatureData(ctx context.Context, s core.Store) {
	// v1 版本的用户特征（基础特征）
	v1UserFeatures := map[string]float64{
		"age":    25.0,
		"gender": 1.0,
		"city":   1.0, // beijing
	}
	v1UserData, _ := json.Marshal(v1UserFeatures)
	s.Set(ctx, "user:features:v1:42", v1UserData, 3600)

	// v2 版本的用户特征（增强特征）
	v2UserFeatures := map[string]float64{
		"age":         25.0,
		"gender":      1.0,
		"city":        1.0,
		"interests":   0.8, // 新增：兴趣标签
		"click_count": 150.0,
		"view_count":  500.0,
	}
	v2UserData, _ := json.Marshal(v2UserFeatures)
	s.Set(ctx, "user:features:v2:42", v2UserData, 3600)

	// v1 版本的物品特征
	v1ItemFeatures := map[string]float64{
		"ctr":   0.15,
		"cvr":   0.08,
		"price": 99.0,
	}
	v1ItemData, _ := json.Marshal(v1ItemFeatures)
	s.Set(ctx, "item:features:v1:1", v1ItemData, 3600)
	s.Set(ctx, "item:features:v1:2", v1ItemData, 3600)
	s.Set(ctx, "item:features:v1:3", v1ItemData, 3600)

	// v2 版本的物品特征（增强特征）
	v2ItemFeatures := map[string]float64{
		"ctr":        0.15,
		"cvr":        0.08,
		"price":      99.0,
		"category":   3.0,  // 新增：类别
		"popularity": 0.85, // 新增：热度
	}
	v2ItemData, _ := json.Marshal(v2ItemFeatures)
	s.Set(ctx, "item:features:v2:1", v2ItemData, 3600)
	s.Set(ctx, "item:features:v2:2", v2ItemData, 3600)
	s.Set(ctx, "item:features:v2:3", v2ItemData, 3600)

	fmt.Println("版本化特征数据已准备完成")
	fmt.Println("  - v1 版本：基础特征（age, gender, city）")
	fmt.Println("  - v2 版本：增强特征（新增 interests, click_count, view_count, category, popularity）")
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	fmt.Println("=== 特征版本管理示例 ===")
	fmt.Println()

	// 1. 初始化 Store
	var s core.Store
	redisStore, err := store.NewRedisStore("localhost:6379", 0)
	if err != nil {
		log.Printf("Redis 连接失败，使用内存 Store: %v", err)
		s = store.NewMemoryStore()
	} else {
		s = redisStore
	}
	defer s.Close(ctx)

	// 2. 准备不同版本的特征数据
	prepareVersionedFeatureData(ctx, s)

	// 3. 创建版本化特征服务
	versionedService := createVersionedFeatureService(s)
	defer versionedService.Close(ctx)

	// 4. 演示版本选择
	fmt.Println("\n=== 版本选择演示 ===")
	testUserIDs := []string{"42", "100", "200", "300", "400"}
	for _, userID := range testUserIDs {
		version := versionedService.selectVersion(userID)
		features, err := versionedService.GetUserFeatures(ctx, userID)
		if err != nil {
			log.Printf("用户 %s 获取特征失败: %v", userID, err)
			continue
		}
		fmt.Printf("用户 %s -> 版本: %s, 特征数: %d\n", userID, version, len(features))
		fmt.Printf("  特征: %v\n", features)
	}

	// 5. 创建特征注入节点（使用版本化特征服务）
	enrichNode := &feature.EnrichNode{
		FeatureService:     versionedService,
		UserFeaturePrefix:  "user_",
		ItemFeaturePrefix:  "item_",
		CrossFeaturePrefix: "cross_",
	}

	// 6. 创建排序模型（支持不同版本的特征）
	lr := &rank.LRNode{
		Model: &model.LRModel{
			Bias: 0,
			Weights: map[string]float64{
				// v1 特征权重
				"item_ctr": 1.2,
				"item_cvr": 0.8,
				"user_age": 0.5,
				// v2 新增特征权重
				"user_interests":   0.3,
				"user_click_count": 0.1,
				"item_category":    0.2,
				"item_popularity":  0.4,
			},
		},
	}

	// 7. 构建 Pipeline
	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			// 召回
			&recall.Fanout{
				Sources: []recall.Source{
					&recall.Hot{IDs: []string{"1", "2", "3"}},
				},
				Dedup: true,
			},
			// 特征注入（使用版本化特征服务）
			enrichNode,
			// 排序
			lr,
		},
	}

	// 8. 创建用户上下文
	rctx := &core.RecommendContext{
		UserID: "42",
		Scene:  "feed",
		UserProfile: map[string]any{
			"age":    25.0,
			"gender": 1.0,
			"city":   "beijing",
		},
	}

	// 9. 运行 Pipeline
	fmt.Println("\n=== 运行 Pipeline（使用版本化特征服务）===")
	items, err := p.Run(ctx, rctx, nil)
	if err != nil {
		log.Fatalf("Pipeline 运行失败: %v", err)
	}

	// 10. 输出结果
	fmt.Printf("\n推荐结果（共 %d 个物品）:\n", len(items))
	for i, item := range items {
		if item == nil {
			continue
		}
		fmt.Printf("\n[%d] Item ID: %s, Score: %.4f\n", i+1, item.ID, item.Score)
		fmt.Printf("    特征数量: %d\n", len(item.Features))

		// 显示用户特征
		fmt.Printf("    用户特征: ")
		userFeatureCount := 0
		for k, v := range item.Features {
			if len(k) > 5 && k[:5] == "user_" {
				fmt.Printf("%s=%.2f ", k, v)
				userFeatureCount++
			}
		}
		if userFeatureCount == 0 {
			fmt.Print("无")
		}

		// 显示物品特征
		fmt.Printf("\n    物品特征: ")
		itemFeatureCount := 0
		for k, v := range item.Features {
			if len(k) > 5 && k[:5] == "item_" {
				fmt.Printf("%s=%.2f ", k, v)
				itemFeatureCount++
			}
		}
		if itemFeatureCount == 0 {
			fmt.Print("无")
		}

		// 显示交叉特征
		fmt.Printf("\n    交叉特征: ")
		crossFeatureCount := 0
		for k, v := range item.Features {
			if len(k) > 6 && k[:6] == "cross_" {
				fmt.Printf("%s=%.2f ", k, v)
				crossFeatureCount++
			}
		}
		if crossFeatureCount == 0 {
			fmt.Print("无")
		}
		fmt.Println()
	}

	// 11. 演示版本信息查询
	fmt.Println("\n=== 版本信息查询 ===")
	registry := versionedService.registry
	versions := registry.ListVersions()
	for _, version := range versions {
		meta, err := registry.GetVersion(version)
		if err != nil {
			continue
		}
		fmt.Printf("版本: %s\n", meta.Version)
		fmt.Printf("  描述: %s\n", meta.Description)
		fmt.Printf("  状态: %s\n", meta.Status)
		fmt.Printf("  创建时间: %s\n", meta.CreatedAt.Format("2006-01-02 15:04:05"))
		fmt.Printf("  特征列表: %v\n", meta.Features)
		fmt.Println()
	}

	// 12. 演示版本切换配置
	fmt.Println("=== 版本切换配置 ===")
	fmt.Printf("默认版本: %s\n", versionedService.config.DefaultVersion)
	fmt.Printf("流量分配:\n")
	for version, ratio := range versionedService.config.TrafficSplit {
		fmt.Printf("  %s: %.1f%%\n", version, ratio*100)
	}
}
