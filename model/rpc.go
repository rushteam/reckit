package model

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// RPCModel 是通过 RPC/HTTP 调用外部模型服务的 RankModel 实现。
// 支持 GBDT、XGBoost、TensorFlow Serving、TorchServe 等。
type RPCModel struct {
	name     string
	Endpoint string // 例如 "http://localhost:8080/predict"
	Timeout  time.Duration
	Client   *http.Client
}

func NewRPCModel(name, endpoint string, timeout time.Duration) *RPCModel {
	if timeout == 0 {
		timeout = 5 * time.Second
	}
	return &RPCModel{
		name:     name,
		Endpoint: endpoint,
		Timeout:  timeout,
		Client: &http.Client{
			Timeout: timeout,
		},
	}
}

func (m *RPCModel) Name() string {
	return m.name
}

// Predict 调用远程模型服务进行预测（单个特征，内部调用批量接口）。
func (m *RPCModel) Predict(features map[string]float64) (float64, error) {
	scores, err := m.PredictBatch([]map[string]float64{features})
	if err != nil {
		return 0, err
	}
	if len(scores) == 0 {
		return 0, fmt.Errorf("empty response")
	}
	return scores[0], nil
}

// PredictBatch 调用远程模型服务进行批量预测。
// 请求格式（JSON）：
//
//	{"features_list": [{"ctr": 0.15, "cvr": 0.08, ...}, ...]}
//
// 响应格式（JSON）：
//
//	{"scores": [0.85, 0.72, ...]}
func (m *RPCModel) PredictBatch(featuresList []map[string]float64) ([]float64, error) {
	if m.Client == nil {
		m.Client = &http.Client{Timeout: m.Timeout}
	}

	if len(featuresList) == 0 {
		return []float64{}, nil
	}

	// 构建请求
	reqBody := map[string]any{
		"features_list": featuresList,
	}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", m.Endpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// 发送请求
	resp, err := m.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("rpc call: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("rpc error: status=%d, read body failed: %w", resp.StatusCode, err)
		}
		return nil, fmt.Errorf("rpc error: status=%d, body=%s", resp.StatusCode, string(body))
	}

	// 解析响应
	var result struct {
		Scores []float64 `json:"scores"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	if len(result.Scores) != len(featuresList) {
		return nil, fmt.Errorf("response scores count mismatch: expected %d, got %d", len(featuresList), len(result.Scores))
	}

	return result.Scores, nil
}

