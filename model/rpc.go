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
//
// 统一协议：Endpoint 为完整 URL，如 "http://localhost:8080/predictions/xgb"；
// 请求体 {"data": [{"feature_a": 0.1, ...}, ...]}，响应体 {"predictions": [0.85, ...]}。
// 仅支持该协议，不兼容旧格式（features_list/scores）。
type RPCModel struct {
	name     string
	Endpoint string // 如 "http://localhost:8080/predictions/xgb" 或 "http://localhost:8080/predictions/deepfm"
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
// 统一协议：请求 {"data": [...]}，响应 {"predictions": [...]}。
func (m *RPCModel) PredictBatch(featuresList []map[string]float64) ([]float64, error) {
	if m.Client == nil {
		m.Client = &http.Client{Timeout: m.Timeout}
	}

	if len(featuresList) == 0 {
		return []float64{}, nil
	}

	reqBody := map[string]any{"data": featuresList}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", m.Endpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := m.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("rpc call: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("rpc error: status=%d, body=%s", resp.StatusCode, string(body))
	}

	var result struct {
		Predictions []float64 `json:"predictions"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	if len(result.Predictions) != len(featuresList) {
		return nil, fmt.Errorf("response predictions count mismatch: expected %d, got %d", len(featuresList), len(result.Predictions))
	}
	return result.Predictions, nil
}

