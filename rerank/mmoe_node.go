package rerank

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/utils"
)

// MMoENode 是基于 MMoE 多目标模型的重排节点。
//
// 调用远程 MMoE 服务（如 Python mmoe_server），获取每个 item 的
// ctr、watch_time、gmv 等多任务分数，按配置权重加权求和后更新 item.Score 并重排。
//
// 使用前需先训练并启动 MMoE 服务：
//
//	python train/train_mmoe.py
//	uvicorn service.mmoe_server:app --host 0.0.0.0 --port 8081
//
// 使用示例：
//
//	node := &rerank.MMoENode{
//	    Endpoint:       "http://localhost:8081/predict",
//	    Timeout:        5 * time.Second,
//	    WeightCTR:      1.0,
//	    WeightWatchTime: 0.01,
//	    WeightGMV:      1e-6,
//	}
type MMoENode struct {
	Endpoint string
	Timeout  time.Duration
	Client   *http.Client

	// 多目标权重。加权分 = WeightCTR*ctr + WeightWatchTime*watch_time + WeightGMV*gmv
	WeightCTR       float64
	WeightWatchTime float64
	WeightGMV       float64

	// 可选：去掉特征名前缀再发给 MMoE（与 RPCNode 行为一致）
	StripFeaturePrefix bool
}

// mmoePredictReq 与 Python /predict 请求格式一致
type mmoePredictReq struct {
	FeaturesList []map[string]float64 `json:"features_list"`
}

// taskScores 与 Python TaskScores 一致
type taskScores struct {
	CTR       float64 `json:"ctr"`
	WatchTime float64 `json:"watch_time"`
	GMV       float64 `json:"gmv"`
}

// mmoePredictResp 与 Python /predict 响应格式一致
type mmoePredictResp struct {
	ScoresList []taskScores `json:"scores_list"`
}

func (n *MMoENode) Name() string { return "rerank.mmoe" }
func (n *MMoENode) Kind() pipeline.Kind { return pipeline.KindReRank }

func (n *MMoENode) Process(
	ctx context.Context,
	_ *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if n.Endpoint == "" || len(items) == 0 {
		return items, nil
	}
	client := n.Client
	if client == nil {
		t := n.Timeout
		if t <= 0 {
			t = 5 * time.Second
		}
		client = &http.Client{Timeout: t}
	}

	featuresList := make([]map[string]float64, 0, len(items))
	valid := make([]*core.Item, 0, len(items))
	for _, it := range items {
		if it == nil {
			continue
		}
		f := it.Features
		if f == nil {
			f = make(map[string]float64)
		}
		if n.StripFeaturePrefix {
			f = n.stripPrefix(f)
		}
		featuresList = append(featuresList, f)
		valid = append(valid, it)
	}
	if len(featuresList) == 0 {
		return items, nil
	}

	reqBody := mmoePredictReq{FeaturesList: featuresList}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("mmoe marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, "POST", n.Endpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("mmoe create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("mmoe rpc call: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("mmoe rpc error: status=%d body=%s", resp.StatusCode, string(body))
	}

	var result mmoePredictResp
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("mmoe decode response: %w", err)
	}
	if len(result.ScoresList) != len(valid) {
		return nil, fmt.Errorf("mmoe scores count mismatch: want %d got %d", len(valid), len(result.ScoresList))
	}

	wCTR, wWatch, wGMV := n.WeightCTR, n.WeightWatchTime, n.WeightGMV
	for i, it := range valid {
		s := result.ScoresList[i]
		score := wCTR*s.CTR + wWatch*s.WatchTime + wGMV*s.GMV
		it.Score = score
		it.PutLabel("rerank_model", utils.Label{Value: "mmoe", Source: "rerank"})
		it.PutLabel("mmoe_ctr", utils.Label{Value: fmt.Sprintf("%.4f", s.CTR), Source: "rerank"})
		it.PutLabel("mmoe_watch_time", utils.Label{Value: fmt.Sprintf("%.2f", s.WatchTime), Source: "rerank"})
		it.PutLabel("mmoe_gmv", utils.Label{Value: fmt.Sprintf("%.2f", s.GMV), Source: "rerank"})
	}

	sort.SliceStable(items, func(i, j int) bool {
		if items[i] == nil {
			return false
		}
		if items[j] == nil {
			return true
		}
		return items[i].Score > items[j].Score
	})
	return items, nil
}

func (n *MMoENode) stripPrefix(features map[string]float64) map[string]float64 {
	out := make(map[string]float64)
	prefixes := []string{"item_", "user_", "cross_", "scene_"}
	for k, v := range features {
		key := k
		for _, p := range prefixes {
			if strings.HasPrefix(k, p) {
				key = strings.TrimPrefix(k, p)
				break
			}
		}
		out[key] = v
	}
	return out
}

// 确保 MMoENode 实现 pipeline.Node
var _ pipeline.Node = (*MMoENode)(nil)
