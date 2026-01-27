package recall

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/feature"
	"github.com/rushteam/reckit/pkg/utils"
)

// YouTubeDNNRecall 是基于 YouTube DNN 的召回源。
//
// 流程：用户特征 + 用户历史行为 -> /user_embedding -> 用户向量 -> 向量检索 Item。
// 使用前需训练并启动 YouTube DNN 服务，且 Item Embeddings 已导入 VectorService。
//
//	python train/train_youtube_dnn.py
//	uvicorn service.youtube_dnn_server:app --host 0.0.0.0 --port 8082
type YouTubeDNNRecall struct {
	FeatureService core.FeatureService
	Endpoint       string // HTTP 服务端点，例如 "http://localhost:8082/user_embedding"
	Timeout        time.Duration
	Client         *http.Client

	VectorService core.VectorService
	TopK          int
	Collection    string
	Metric        string

	UserFeatureExtractor feature.FeatureExtractor
	HistoryExtractor     *feature.HistoryExtractor
}

// youtubeUserEmbReq 与 Python /user_embedding 请求一致
type youtubeUserEmbReq struct {
	UserFeatures   map[string]float64 `json:"user_features"`
	HistoryItemIDs []string           `json:"history_item_ids"`
}

// youtubeUserEmbResp 与 Python 响应一致
type youtubeUserEmbResp struct {
	UserEmbedding []float64 `json:"user_embedding"`
}

func (r *YouTubeDNNRecall) Name() string { return "recall.youtube_dnn" }

func (r *YouTubeDNNRecall) Recall(ctx context.Context, rctx *core.RecommendContext) ([]*core.Item, error) {
	userFeat, err := r.getUserFeatures(ctx, rctx)
	if err != nil {
		return nil, fmt.Errorf("youtube_dnn get user features: %w", err)
	}
	hist := r.getHistory(rctx)

	userEmbedding, err := r.fetchUserEmbedding(ctx, userFeat, hist)
	if err != nil {
		return nil, fmt.Errorf("youtube_dnn user embedding: %w", err)
	}
	if len(userEmbedding) == 0 {
		return nil, nil
	}

	topK := r.TopK
	if topK <= 0 {
		topK = 100
	}
	collection := r.Collection
	if collection == "" {
		collection = "youtube_dnn_items"
	}
	metric := r.Metric
	if metric == "" {
		metric = "inner_product"
	}

	req := &core.VectorSearchRequest{
		Collection: collection,
		Vector:     userEmbedding,
		TopK:       topK,
		Metric:     metric,
	}
	res, err := r.VectorService.Search(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("youtube_dnn vector search: %w", err)
	}
	return r.convertToItems(res), nil
}

func (r *YouTubeDNNRecall) getUserFeatures(ctx context.Context, rctx *core.RecommendContext) (map[string]float64, error) {
	if r.UserFeatureExtractor != nil {
		return r.UserFeatureExtractor.Extract(ctx, rctx)
	}
	if r.FeatureService != nil {
		return r.FeatureService.GetUserFeatures(ctx, rctx.UserID)
	}
	// 默认：使用带 "user_" 前缀的抽取器
	defaultExtractor := feature.NewDefaultFeatureExtractor(
		feature.WithFeatureService(r.FeatureService),
		feature.WithFieldPrefix("user_"),
	)
	return defaultExtractor.Extract(ctx, rctx)
}

func (r *YouTubeDNNRecall) getHistory(rctx *core.RecommendContext) []string {
	if r.HistoryExtractor != nil {
		return r.HistoryExtractor.Extract(rctx)
	}
	// 默认：从 User.RecentClicks 获取
	if rctx != nil && rctx.User != nil && len(rctx.User.RecentClicks) > 0 {
		return rctx.User.RecentClicks
	}
	return nil
}

func (r *YouTubeDNNRecall) fetchUserEmbedding(ctx context.Context, userFeat map[string]float64, hist []string) ([]float64, error) {
	if r.Endpoint == "" {
		return nil, fmt.Errorf("endpoint is required")
	}
	client := r.Client
	if client == nil {
		t := r.Timeout
		if t <= 0 {
			t = 5 * time.Second
		}
		client = &http.Client{Timeout: t}
	}

	body := youtubeUserEmbReq{UserFeatures: userFeat, HistoryItemIDs: hist}
	if hist == nil {
		body.HistoryItemIDs = []string{}
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", r.Endpoint, bytes.NewBuffer(raw))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("user_embedding status=%d body=%s", resp.StatusCode, string(b))
	}

	var res youtubeUserEmbResp
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, err
	}
	return res.UserEmbedding, nil
}

func (r *YouTubeDNNRecall) convertToItems(result *core.VectorSearchResult) []*core.Item {
	if result == nil || len(result.Items) == 0 {
		return []*core.Item{}
	}
	items := make([]*core.Item, 0, len(result.Items))
	for _, resItem := range result.Items {
		it := core.NewItem(resItem.ID)
		it.Score = resItem.Score
		it.PutLabel("recall_source", utils.Label{Value: "youtube_dnn", Source: "recall"})
		it.PutLabel("recall_type", utils.Label{Value: "vector_search", Source: "recall"})
		items = append(items, it)
	}
	return items
}

var _ Source = (*YouTubeDNNRecall)(nil)
