package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/feature"
	"github.com/rushteam/reckit/filter"
	"github.com/rushteam/reckit/model"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/rank"
	"github.com/rushteam/reckit/recall"
	"github.com/rushteam/reckit/rerank"
	"github.com/rushteam/reckit/store"
)

// ========== Mock 存储实现 ==========

// MockUserHistoryStore 实现 UserHistoryStore 接口
type MockUserHistoryStore struct {
	store core.Store
}

func NewMockUserHistoryStore(s core.Store) *MockUserHistoryStore {
	return &MockUserHistoryStore{store: s}
}

func (m *MockUserHistoryStore) GetUserHistory(ctx context.Context, userID string, keyPrefix, behaviorType string, timeWindow int64) ([]string, error) {
	key := fmt.Sprintf("%s:%s:%s", keyPrefix, userID, behaviorType)
	data, err := m.store.Get(ctx, key)
	if err != nil {
		if core.IsStoreNotFound(err) {
			return []string{}, nil
		}
		return nil, err
	}

	var items []string
	if err := json.Unmarshal(data, &items); err != nil {
		return nil, err
	}
	return items, nil
}

func (m *MockUserHistoryStore) GetSimilarItems(ctx context.Context, itemIDs []string, topK int) ([]string, error) {
	similar := []string{}
	for _, itemID := range itemIDs {
		var id int
		if _, err := fmt.Sscanf(itemID, "item_%d", &id); err == nil {
			similar = append(similar, fmt.Sprintf("item_%d", id*10))
		}
		if len(similar) >= topK {
			break
		}
	}
	return similar, nil
}

// MockContentStore 实现 ContentStore 接口
type MockContentStore struct {
	store core.Store
}

func NewMockContentStore(s core.Store) *MockContentStore {
	return &MockContentStore{store: s}
}

func (m *MockContentStore) GetItemFeatures(ctx context.Context, itemID string) (map[string]float64, error) {
	key := "content:item:" + itemID
	data, err := m.store.Get(ctx, key)
	if err != nil {
		if core.IsStoreNotFound(err) {
			return make(map[string]float64), nil
		}
		return nil, err
	}

	var features map[string]float64
	if err := json.Unmarshal(data, &features); err != nil {
		return nil, err
	}
	return features, nil
}

func (m *MockContentStore) GetUserPreferences(ctx context.Context, userID string) (map[string]float64, error) {
	key := "content:user:" + userID
	data, err := m.store.Get(ctx, key)
	if err != nil {
		if core.IsStoreNotFound(err) {
			return make(map[string]float64), nil
		}
		return nil, err
	}

	var prefs map[string]float64
	if err := json.Unmarshal(data, &prefs); err != nil {
		return nil, err
	}
	return prefs, nil
}

func (m *MockContentStore) GetSimilarItems(ctx context.Context, itemFeatures map[string]float64, topK int) ([]string, error) {
	return []string{}, nil
}

func (m *MockContentStore) GetAllItems(ctx context.Context) ([]string, error) {
	key := "content:items"
	data, err := m.store.Get(ctx, key)
	if err != nil {
		if core.IsStoreNotFound(err) {
			return []string{}, nil
		}
		return nil, err
	}

	var items []string
	if err := json.Unmarshal(data, &items); err != nil {
		return nil, err
	}
	return items, nil
}

// MockWord2VecStore 实现 Word2VecStore 接口（物品 name/desc 文本、用户序列）
type MockWord2VecStore struct {
	itemTexts map[string]string   // itemID -> 文本（name + desc）
	itemTags  map[string][]string // itemID -> 标签
	allItems  []string
	sequence  func(userID string) []string // 用户行为序列
}

func NewMockWord2VecStore(itemTexts map[string]string, itemTags map[string][]string, allItems []string, sequence func(userID string) []string) *MockWord2VecStore {
	if itemTexts == nil {
		itemTexts = make(map[string]string)
	}
	if itemTags == nil {
		itemTags = make(map[string][]string)
	}
	return &MockWord2VecStore{
		itemTexts: itemTexts,
		itemTags:  itemTags,
		allItems:  allItems,
		sequence:  sequence,
	}
}

func (s *MockWord2VecStore) GetItemText(_ context.Context, itemID string) (string, error) {
	return s.itemTexts[itemID], nil
}

func (s *MockWord2VecStore) GetItemTags(_ context.Context, itemID string) ([]string, error) {
	return s.itemTags[itemID], nil
}

func (s *MockWord2VecStore) GetUserSequence(_ context.Context, userID string, maxLen int) ([]string, error) {
	if s.sequence == nil {
		return nil, nil
	}
	seq := s.sequence(userID)
	if len(seq) > maxLen {
		seq = seq[len(seq)-maxLen:]
	}
	return seq, nil
}

func (s *MockWord2VecStore) GetAllItems(_ context.Context) ([]string, error) {
	return s.allItems, nil
}

// loadWord2VecModel 加载 Word2Vec 模型，优先 JSON，否则内联；失败返回 nil（跳过 Word2Vec 召回）
func loadWord2VecModel() *model.Word2VecModel {
	candidates := []string{
		filepath.Join("examples", "word2vec", "item2vec_vectors.json"),
		filepath.Join("python", "model", "item2vec_vectors.json"),
	}
	for _, p := range candidates {
		data, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		var raw map[string]interface{}
		if err := json.Unmarshal(data, &raw); err != nil {
			continue
		}
		m, err := model.LoadWord2VecFromMap(raw)
		if err == nil && m != nil {
			return m
		}
	}
	return embedWord2VecModel()
}

func embedWord2VecModel() *model.Word2VecModel {
	wordVectors := map[string][]float64{
		"electronics": {0.1, 0.2, 0.3, 0.4},
		"smartphone":  {0.2, 0.3, 0.4, 0.5},
		"tech":        {0.15, 0.25, 0.35, 0.45},
		"game":        {0.2, 0.25, 0.3, 0.35},
		"sports":      {0.18, 0.22, 0.28, 0.32},
		"item_1":      {0.1, 0.2, 0.3, 0.4},
		"item_2":      {0.2, 0.3, 0.4, 0.5},
		"item_3":      {0.3, 0.4, 0.5, 0.6},
		"item_4":      {0.4, 0.5, 0.6, 0.7},
		"item_5":      {0.5, 0.6, 0.7, 0.8},
		"item_6":      {0.6, 0.7, 0.8, 0.9},
		"item_10":     {0.15, 0.25, 0.35, 0.45},
		"item_11":     {0.25, 0.35, 0.45, 0.55},
		"item_12":     {0.35, 0.45, 0.55, 0.65},
		"item_13":     {0.45, 0.55, 0.65, 0.75},
		"item_14":     {0.55, 0.65, 0.75, 0.85},
	}
	return model.NewWord2VecModel(wordVectors, 4)
}

// ========== 辅助函数 ==========

// encodeGender 编码性别：male=1, female=2, unknown=0
func encodeGender(gender string) float64 {
	switch gender {
	case "male":
		return 1.0
	case "female":
		return 2.0
	default:
		return 0.0
	}
}

// encodeRegion 编码地区（简化：使用数字 ID）
func encodeRegion(region string) float64 {
	regionMap := map[string]float64{
		"beijing":  1.0,
		"shanghai": 2.0,
		"guangzhou": 3.0,
		"shenzhen": 4.0,
	}
	if id, ok := regionMap[region]; ok {
		return id
	}
	return 0.0
}

// encodeCategory 编码类别（简化：使用数字 ID）
func encodeCategory(category string) float64 {
	categoryMap := map[string]float64{
		"tech":   1.0,
		"game":   2.0,
		"sports": 3.0,
		"music":  4.0,
		"movie":  5.0,
	}
	if id, ok := categoryMap[category]; ok {
		return id
	}
	return 0.0
}

// setupTestData 设置测试数据
func setupTestData(ctx context.Context, memStore core.Store) error {
	// 1. 用户历史行为（不同时间窗口）
	userHistoryClick, _ := json.Marshal([]string{"item_1", "item_2", "item_3"})
	memStore.Set(ctx, "user:history:user_123:click", userHistoryClick, 0)

	userHistoryView, _ := json.Marshal([]string{"item_4", "item_5"})
	memStore.Set(ctx, "user:history:user_123:view", userHistoryView, 0)

	userHistoryLike, _ := json.Marshal([]string{"item_6"})
	memStore.Set(ctx, "user:history:user_123:like", userHistoryLike, 0)

	// 2. 协同过滤数据
	cfUserData, _ := json.Marshal(map[string]float64{
		"item_1": 1.0, // 点击
		"item_2": 2.0, // 点赞（权重更高）
		"item_3": 1.0,
	})
	memStore.Set(ctx, "cf:user:user_123", cfUserData, 0)

	cfItem1Data, _ := json.Marshal(map[string]float64{
		"user_123": 1.0,
		"user_456": 1.0,
		"user_789": 2.0,
	})
	memStore.Set(ctx, "cf:item:item_1", cfItem1Data, 0)

	cfItem2Data, _ := json.Marshal(map[string]float64{
		"user_123": 2.0,
		"user_456": 1.0,
	})
	memStore.Set(ctx, "cf:item:item_2", cfItem2Data, 0)

	// 3. 内容特征（物品）
	item1Features, _ := json.Marshal(map[string]float64{
		"category": 1.0, // tech
		"price":    99.0,
		"ctr":      0.15,
	})
	memStore.Set(ctx, "content:item:item_1", item1Features, 0)

	item2Features, _ := json.Marshal(map[string]float64{
		"category": 2.0, // game
		"price":    199.0,
		"ctr":      0.20,
	})
	memStore.Set(ctx, "content:item:item_2", item2Features, 0)

	item3Features, _ := json.Marshal(map[string]float64{
		"category": 1.0, // tech
		"price":    299.0,
		"ctr":      0.10,
	})
	memStore.Set(ctx, "content:item:item_3", item3Features, 0)

	// 4. 用户偏好
	userPrefs, _ := json.Marshal(map[string]float64{
		"tech":   0.8,
		"game":   0.6,
		"sports": 0.4,
	})
	memStore.Set(ctx, "content:user:user_123", userPrefs, 0)

	// 5. 热门物品
	if kvStore, ok := memStore.(core.KeyValueStore); ok {
		hotItems := []struct {
			id    int64
			score float64
		}{
			{10, 100.0}, {11, 95.0}, {12, 90.0}, {13, 85.0}, {14, 80.0},
		}
		for _, item := range hotItems {
			kvStore.ZAdd(ctx, "hot:feed", item.score, fmt.Sprintf("item_%d", item.id))
		}
	}

	// 6. 曝光历史
	exposedData, _ := json.Marshal([]struct {
		ItemID    string `json:"item_id"`
		Timestamp int64  `json:"timestamp"`
	}{
		{"item_10", time.Now().Unix() - 1*24*3600}, // 1 天前
		{"item_11", time.Now().Unix() - 2*24*3600}, // 2 天前
	})
	memStore.Set(ctx, "user:exposed:user_123", exposedData, 0)

	// 7. 黑名单
	blacklist, _ := json.Marshal([]string{"item_99", "item_98"})
	memStore.Set(ctx, "blacklist:items", blacklist, 0)

	// 8. 用户特征
	userFeatures, _ := json.Marshal(map[string]float64{
		"user_age":    25.0,
		"user_gender": 1.0, // male
		"user_region": 1.0, // beijing
		"user_click_count": 150.0,
		"user_view_count":  500.0,
	})
	memStore.Set(ctx, "user:features:user_123", userFeatures, 0)

	// 9. 物品特征
	item1Feat, _ := json.Marshal(map[string]float64{
		"item_category": 1.0,
		"item_price":    99.0,
		"item_ctr":      0.15,
		"item_cvr":      0.08,
	})
	memStore.Set(ctx, "item:features:item_1", item1Feat, 0)

	item2Feat, _ := json.Marshal(map[string]float64{
		"item_category": 2.0,
		"item_price":    199.0,
		"item_ctr":      0.20,
		"item_cvr":      0.10,
	})
	memStore.Set(ctx, "item:features:item_2", item2Feat, 0)

	return nil
}

// enrichItemFeatures 为物品注入基础特征
type enrichItemFeatures struct{}

func (n *enrichItemFeatures) Name() string        { return "enrich.item_features" }
func (n *enrichItemFeatures) Kind() pipeline.Kind { return pipeline.KindPostProcess }

func (n *enrichItemFeatures) Process(
	_ context.Context,
	_ *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	// 为物品注入基础特征（从 Meta 或 Store 获取）
	for _, it := range items {
		// 如果 Meta 中有 category，编码为特征
		if category, ok := it.Meta["category"].(string); ok {
			it.Features["category"] = encodeCategory(category)
		}
		// 如果 Meta 中有 price，直接使用
		if price, ok := it.Meta["price"].(float64); ok {
			it.Features["price"] = price
		}
		// 模拟 CTR/CVR（实际应从 Store 获取）
		if _, ok := it.Features["ctr"]; !ok || it.Features["ctr"] == 0 {
			it.Features["ctr"] = 0.1 + float64(len(it.ID)%10)*0.01
		}
		if _, ok := it.Features["cvr"]; !ok || it.Features["cvr"] == 0 {
			it.Features["cvr"] = 0.05 + float64(len(it.ID)%5)*0.01
		}
	}
	return items, nil
}

func main() {
	rankModel := flag.String("rank", "lr", "排序模型: lr | xgb | deepfm")
	enableWord2Vec := flag.Bool("word2vec", true, "是否启用 Word2Vec 召回（有模型时）")
	flag.Parse()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// ========== 1. 初始化 Store ==========
	memStore := store.NewMemoryStore()
	defer memStore.Close()

	// 设置测试数据
	if err := setupTestData(ctx, memStore); err != nil {
		panic(err)
	}
	fmt.Println("✅ 测试数据已设置")

	// ========== 2. 创建存储适配器 ==========
	userHistoryStore := NewMockUserHistoryStore(memStore)
	contentStore := NewMockContentStore(memStore)
	cfStore := recall.NewStoreCFAdapter(memStore, "cf")
	storeAdapter := filter.NewStoreAdapter(memStore)

	// ========== 3. 创建特征服务 ==========
	featureProvider := feature.NewStoreFeatureProvider(
		memStore,
		feature.KeyPrefix{
			User:     "user:features:",
			Item:     "item:features:",
			Realtime: "realtime:features:",
		},
	)
	featureService := feature.NewBaseFeatureService(featureProvider)

	// ========== 4. 构建召回策略（多路并发） ==========
	sources := []recall.Source{
		// 1. 用户历史召回（点击，7 天，最高优先级）
		&recall.UserHistory{
			Store:               userHistoryStore,
			KeyPrefix:           "user:history",
			BehaviorType:        "click",
			TimeWindow:          7 * 24 * 3600, // 7 天
			TopK:                50,
			EnableSimilarExtend: true,
		},
		// 2. I2I 协同过滤召回
		&recall.I2IRecall{
			Store:                cfStore,
			TopKSimilarItems:     10,
			TopKItems:            30,
			SimilarityCalculator: &recall.CosineSimilarity{},
			Config:               &core.DefaultRecallConfig{},
		},
		// 3. 内容召回（基于 category）
		&recall.ContentRecall{
			Store:            contentStore,
			TopK:             20,
			SimilarityMetric: "cosine",
			UserPreferencesExtractor: func(rctx *core.RecommendContext) map[string]float64 {
				if rctx.User != nil {
					return rctx.User.Interests // category -> weight
				}
				return nil
			},
		},
	}

	// 4. 可选：Word2Vec 召回（物品文本/序列相似度）
	w2vModel := loadWord2VecModel()
	priorityWord2Vec := 4
	if *enableWord2Vec && w2vModel != nil {
		w2vStore := NewMockWord2VecStore(
			map[string]string{
				"item_1": "electronics smartphone tech",
				"item_2": "game smartphone mobile",
				"item_3": "electronics laptop tech",
				"item_4": "sports game mobile",
				"item_5": "tech computer laptop",
				"item_6": "sports electronics",
				"item_10": "electronics smartphone",
				"item_11": "game sports",
				"item_12": "tech laptop",
				"item_13": "mobile device",
				"item_14": "electronics tech",
			},
			nil,
			[]string{"item_1", "item_2", "item_3", "item_4", "item_5", "item_6", "item_10", "item_11", "item_12", "item_13", "item_14"},
			func(userID string) []string {
				if userID == "user_123" {
					return []string{"item_1", "item_2"}
				}
				return nil
			},
		)
		sources = append(sources, &recall.Word2VecRecall{
			Model:     w2vModel,
			Store:     w2vStore,
			TopK:      20,
			Mode:      "sequence", // sequence=Item2Vec 序列；text=文本相似
			TextField: "title",
		})
		fmt.Println("✅ Word2Vec 召回已启用（sequence 模式）")
	} else if *enableWord2Vec {
		fmt.Println("⚠️ Word2Vec 未启用：未找到模型（可放置 item2vec_vectors.json 后重试）")
	}
	priorityHot := priorityWord2Vec + 1

	// 5. 热门召回（兜底）
	sources = append(sources, &recall.Hot{
		Store: memStore,
		Key:   "hot:feed",
	})

	weights := map[string]int{
		"recall.user_history": 1,
		"recall.i2i":          2,
		"recall.content":      3,
		"recall.word2vec":     priorityWord2Vec,
		"recall.hot":          priorityHot,
	}

	fanout := &recall.Fanout{
		Sources:       sources,
		Dedup:         true,
		MergeStrategy: &recall.PriorityMergeStrategy{PriorityWeights: weights},
		ErrorHandler:  &recall.IgnoreErrorHandler{},
	}

	// ========== 5. 构建过滤策略 ==========
	filterNode := &filter.FilterNode{
		Filters: []filter.Filter{
			// 1. 黑名单过滤（全局）
			filter.NewBlacklistFilter(nil, storeAdapter, "blacklist:items"),
			// 2. 用户拉黑过滤（示例：无拉黑）
			filter.NewUserBlockFilter(storeAdapter, "user:block"),
			// 3. 已曝光过滤（7 天）
			filter.NewExposedFilter(storeAdapter, "user:exposed", 7*24*3600),
		},
	}

	// ========== 6. 构建特征注入节点 ==========
	enrichNode := &feature.EnrichNode{
		// 使用特征服务（推荐）
		FeatureService: featureService,
		
		// 自定义提取器（作为补充，使用新的 FeatureExtractor 接口）
		// 注意：EnrichNode 仍支持函数类型（向后兼容），但推荐使用 FeatureExtractor
		UserFeatureExtractor: func(rctx *core.RecommendContext) map[string]float64 {
			user := rctx.GetUserProfile()
			if user == nil {
				return nil
			}
			features := map[string]float64{
				"user_age":    float64(user.Age),
				"user_gender": encodeGender(user.Gender),
				"user_region": encodeRegion(user.Location),
			}
			// 从 Interests 提取 category 偏好
			if user.Interests != nil {
				for category, weight := range user.Interests {
					features["user_interest_"+category] = weight
				}
			}
			return features
		},
		
		ItemFeatureExtractor: func(item *core.Item) map[string]float64 {
			features := make(map[string]float64)
			if category, ok := item.Meta["category"].(string); ok {
				features["item_category"] = encodeCategory(category)
			}
			if price, ok := item.Meta["price"].(float64); ok {
				features["item_price"] = price
			}
			// 从 Features 中获取已注入的特征
			if ctr, exists := item.Features["ctr"]; exists {
				features["item_ctr"] = ctr
			}
			if cvr, exists := item.Features["cvr"]; exists {
				features["item_cvr"] = cvr
			}
			return features
		},
		
		CrossFeatureExtractor: func(userFeatures, itemFeatures map[string]float64) map[string]float64 {
			return map[string]float64{
				"cross_age_x_category":    userFeatures["user_age"] * itemFeatures["item_category"],
				"cross_gender_x_price":    userFeatures["user_gender"] * itemFeatures["item_price"],
				"cross_region_x_category": userFeatures["user_region"] * itemFeatures["item_category"],
			}
		},
		
		UserFeaturePrefix:  "user_",
		ItemFeaturePrefix:  "item_",
		CrossFeaturePrefix: "cross_",
	}

	// ========== 7. 构建排序节点 ==========
	// 支持多种排序模型：lr（默认）| xgb | deepfm
	// - lr：本地 LR，无需 Python 服务，直接运行
	// - xgb：RPC 调用 XGBoost 服务（需先启动 python/service/server.py）
	// - deepfm：RPC 调用 DeepFM 服务（需先启动 python/service/deepfm_server.py）
	var rankNode pipeline.Node
	switch strings.ToLower(*rankModel) {
	case "xgb", "xgboost":
		rpcModel := model.NewRPCModel("xgboost", "http://localhost:8080/predict", 5*time.Second)
		rankNode = &rank.RPCNode{Model: rpcModel, StripFeaturePrefix: false}
		fmt.Printf("✅ 排序模型: XGBoost (RPC http://localhost:8080)\n")
	case "deepfm":
		rpcModel := model.NewRPCModel("deepfm", "http://localhost:8080/predict", 10*time.Second)
		rankNode = &rank.RPCNode{Model: rpcModel, StripFeaturePrefix: false}
		fmt.Printf("✅ 排序模型: DeepFM (RPC http://localhost:8080)\n")
	default:
		lrModel := &model.LRModel{
			Bias: 0.1,
			Weights: map[string]float64{
				"user_age":              0.5,
				"user_gender":           0.3,
				"item_category":         1.2,
				"item_ctr":              1.5,
				"item_cvr":              0.8,
				"cross_age_x_category":  0.2,
				"cross_gender_x_price":  0.1,
				"cross_region_x_category": 0.1,
			},
		}
		rankNode = &rank.LRNode{Model: lrModel}
		fmt.Printf("✅ 排序模型: LR (本地)\n")
	}

	// ========== 8. 构建 Pipeline ==========
	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			// 1. 召回（多路并发）
			fanout,
			// 2. 物品特征注入（基础特征）
			&enrichItemFeatures{},
			// 3. 过滤
			filterNode,
			// 4. 特征注入（用户特征 + 交叉特征）
			enrichNode,
			// 5. 排序
			rankNode,
			// 6. 重排（多样性）
			&rerank.Diversity{
				LabelKey: "category", // 按 category 保证多样性
			},
		},
	}

	// ========== 9. 构建请求上下文 ==========
	rctx := &core.RecommendContext{
		UserID: "user_123",
		Scene:  "feed",
		User: &core.UserProfile{
			UserID:   "user_123",
			Age:      25,
			Gender:   "male",
			Location: "beijing",
			Interests: map[string]float64{
				"tech":   0.8,
				"game":   0.6,
				"sports": 0.4,
			},
			RecentClicks: []string{"item_1", "item_2"},
		},
		Params: map[string]any{
			"debug": true,
		},
	}

	// ========== 10. 执行 Pipeline ==========
	fmt.Println("\n🚀 开始执行推荐 Pipeline...")
	items, err := p.Run(ctx, rctx, nil)
	if err != nil {
		fmt.Printf("❌ Pipeline 执行失败: %v\n", err)
		return
	}

	// ========== 11. 输出结果 ==========
	fmt.Printf("\n✅ 推荐结果（共 %d 个物品）:\n", len(items))
	for i, it := range items {
		category := "unknown"
		if c, ok := it.Meta["category"].(string); ok {
			category = c
		}
		recallSource := "unknown"
		if label, ok := it.Labels["recall_source"]; ok {
			recallSource = label.Value
		}
		fmt.Printf("#%d id=%s score=%.4f category=%s recall=%s\n",
			i+1, it.ID, it.Score, category, recallSource)
	}

	// ========== 12. 记录曝光（实际应用中） ==========
	fmt.Println("\n📝 记录曝光（用于下次过滤）...")
	// 实际应用中，这里应该将 items 的 ID 写入 user:exposed:{userID}
	// 示例代码：
	// exposedItems := make([]struct {
	//     ItemID    string `json:"item_id"`
	//     Timestamp int64  `json:"timestamp"`
	// }, len(items))
	// for i, it := range items {
	//     exposedItems[i] = struct {
	//         ItemID    string `json:"item_id"`
	//         Timestamp int64  `json:"timestamp"`
	//     }{it.ID, time.Now().Unix()}
	// }
	// exposedData, _ := json.Marshal(exposedItems)
	// memStore.Set(ctx, "user:exposed:user_123", exposedData, 0)
	fmt.Println("✅ 曝光已记录")

	fmt.Println("\n✨ 推荐系统 Pipeline 执行完成！")
}
