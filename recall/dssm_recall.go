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

// DSSMRecall 是基于 DSSM 的 Query-Doc 语义召回源。
//
// 流程：从 rctx 获取 query_features -> /query_embedding -> 向量检索 Doc。
// 使用前需训练并启动 DSSM 服务，且 Doc Embeddings 已导入 VectorService。
//
//	python train/train_dssm.py
//	uvicorn service.dssm_server:app --host 0.0.0.0 --port 8083
type DSSMRecall struct {
	Endpoint string // HTTP 服务端点，例如 "http://localhost:8083/query_embedding"
	Timeout           time.Duration
	Client            *http.Client

	VectorService core.VectorService
	TopK          int
	Collection    string
	Metric        string

	QueryFeatureExtractor feature.FeatureExtractor
}

type dssmQueryEmbReq struct {
	QueryFeatures map[string]float64 `json:"query_features"`
}

type dssmQueryEmbResp struct {
	QueryEmbedding []float64 `json:"query_embedding"`
}

func (r *DSSMRecall) Name() string { return "recall.dssm" }

func (r *DSSMRecall) Recall(ctx context.Context, rctx *core.RecommendContext) ([]*core.Item, error) {
	qf, err := r.getQueryFeatures(ctx, rctx)
	if err != nil {
		return nil, fmt.Errorf("dssm get query features: %w", err)
	}
	if len(qf) == 0 {
		return nil, nil
	}

	queryEmbedding, err := r.fetchQueryEmbedding(ctx, qf)
	if err != nil {
		return nil, fmt.Errorf("dssm query embedding: %w", err)
	}
	if len(queryEmbedding) == 0 {
		return nil, nil
	}

	topK := r.TopK
	if topK <= 0 {
		topK = 100
	}
	collection := r.Collection
	if collection == "" {
		collection = "dssm_docs"
	}
	metric := r.Metric
	if metric == "" {
		metric = "cosine"
	}

	req := &core.VectorSearchRequest{Collection: collection, Vector: queryEmbedding, TopK: topK, Metric: metric}
	res, err := r.VectorService.Search(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("dssm vector search: %w", err)
	}
	return r.convertToItems(res), nil
}

func (r *DSSMRecall) getQueryFeatures(ctx context.Context, rctx *core.RecommendContext) (map[string]float64, error) {
	if r.QueryFeatureExtractor != nil {
		return r.QueryFeatureExtractor.Extract(ctx, rctx)
	}
	// 默认：使用 QueryFeatureExtractor
	defaultExtractor := feature.NewQueryFeatureExtractor()
	return defaultExtractor.Extract(ctx, rctx)
}

func (r *DSSMRecall) fetchQueryEmbedding(ctx context.Context, qf map[string]float64) ([]float64, error) {
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

	body := dssmQueryEmbReq{QueryFeatures: qf}
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
		return nil, fmt.Errorf("query_embedding status=%d body=%s", resp.StatusCode, string(b))
	}

	var res dssmQueryEmbResp
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, err
	}
	return res.QueryEmbedding, nil
}

func (r *DSSMRecall) convertToItems(result *core.VectorSearchResult) []*core.Item {
	if result == nil || len(result.Items) == 0 {
		return []*core.Item{}
	}
	items := make([]*core.Item, 0, len(result.Items))
	for _, resItem := range result.Items {
		it := core.NewItem(resItem.ID)
		it.Score = resItem.Score
		it.PutLabel("recall_source", utils.Label{Value: "dssm", Source: "recall"})
		it.PutLabel("recall_type", utils.Label{Value: "vector_search", Source: "recall"})
		if r.Metric != "" {
			it.PutLabel("recall_metric", utils.Label{Value: r.Metric, Source: "recall"})
		}
		items = append(items, it)
	}
	return items
}

var _ Source = (*DSSMRecall)(nil)
