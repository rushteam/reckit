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
	"github.com/rushteam/reckit/pkg/utils"
)

// GraphRecall 是基于图嵌入（Node2Vec/GraphSAGE）的召回源。
//
// 调用图召回服务 /recall，传入 user_id、top_k，返回相似用户 ID 列表（如「关注页」召回）。
// 使用前需训练 Node2Vec 并启动图召回服务：
//
//	python train/train_node2vec.py --edges data/graph_edges.csv
//	uvicorn service.graph_recall_server:app --host 0.0.0.0 --port 8084
type GraphRecall struct {
	Endpoint string
	Timeout  time.Duration
	Client   *http.Client
	TopK     int
}

type graphRecallReq struct {
	UserID string `json:"user_id"`
	TopK   int    `json:"top_k"`
}

type graphRecallResp struct {
	ItemIDs []string  `json:"item_ids"`
	Scores  []float64 `json:"scores,omitempty"`
}

func (r *GraphRecall) Name() string { return "recall.graph" }

func (r *GraphRecall) Recall(ctx context.Context, rctx *core.RecommendContext) ([]*core.Item, error) {
	if r.Endpoint == "" {
		return nil, fmt.Errorf("graph recall endpoint is required")
	}
	userID := ""
	if rctx != nil {
		userID = rctx.UserID
	}
	if userID == "" {
		return nil, nil
	}

	topK := r.TopK
	if topK <= 0 {
		topK = 20
	}

	client := r.Client
	if client == nil {
		t := r.Timeout
		if t <= 0 {
			t = 5 * time.Second
		}
		client = &http.Client{Timeout: t}
	}

	body := graphRecallReq{UserID: userID, TopK: topK}
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
		return nil, fmt.Errorf("graph recall rpc: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("graph recall status=%d body=%s", resp.StatusCode, string(b))
	}

	var res graphRecallResp
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, err
	}
	return r.convertToItems(res), nil
}

func (r *GraphRecall) convertToItems(res graphRecallResp) []*core.Item {
	if len(res.ItemIDs) == 0 {
		return []*core.Item{}
	}
	items := make([]*core.Item, 0, len(res.ItemIDs))
	for i, id := range res.ItemIDs {
		it := core.NewItem(id)
		if i < len(res.Scores) {
			it.Score = res.Scores[i]
		}
		it.PutLabel("recall_source", utils.Label{Value: "graph", Source: "recall"})
		it.PutLabel("recall_type", utils.Label{Value: "node2vec", Source: "recall"})
		items = append(items, it)
	}
	return items
}

var _ Source = (*GraphRecall)(nil)
