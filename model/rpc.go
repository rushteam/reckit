package model

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/rushteam/reckit/core"
)

// RPCModel 是通过 RPC/HTTP 调用外部模型服务的 RankModel 实现。
//
// 两种使用方式：
//
//  1. 标准方式（推荐）：通过 core.MLService（如 KServeClient）调用，使用 KServe V2 协议：
//     model := NewRPCModelFromService("deepfm", kserveClient)
//
//  2. 直连方式（兼容旧代码）：直接指定 Endpoint，使用自定义协议（{"data": [...]}, {"predictions": [...]}）：
//     model := NewRPCModel("deepfm", "http://localhost:8080/predictions/deepfm", 5*time.Second)
type RPCModel struct {
	name     string
	Endpoint string
	Timeout  time.Duration
	Client   *http.Client

	// Service 标准 ML 服务接口（推荐）。
	// 设置后 PredictBatch 通过此接口调用，忽略 Endpoint/Client。
	Service core.MLService
}

// NewRPCModelFromService 使用 core.MLService（如 KServeClient）创建 RPCModel。
// 这是推荐方式，走标准协议（KServe V2 / TF Serving 等）。
func NewRPCModelFromService(name string, svc core.MLService) *RPCModel {
	return &RPCModel{
		name:    name,
		Service: svc,
	}
}

// NewRPCModel 使用直连 HTTP 创建 RPCModel（兼容旧代码）。
// 请求 {"data": [...]}, 响应 {"predictions": [...]}。
// 新代码推荐使用 NewRPCModelFromService + KServeClient。
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
// 如果 Service 已设置，通过标准 MLService 接口调用；否则走直连 HTTP。
func (m *RPCModel) PredictBatch(featuresList []map[string]float64) ([]float64, error) {
	if len(featuresList) == 0 {
		return []float64{}, nil
	}

	if m.Service != nil {
		return m.predictViaService(featuresList)
	}
	return m.predictViaHTTP(featuresList)
}

// predictViaService 通过标准 MLService 接口（KServe V2 等）调用。
func (m *RPCModel) predictViaService(featuresList []map[string]float64) ([]float64, error) {
	req := &core.MLPredictRequest{
		Features: featuresList,
	}
	resp, err := m.Service.Predict(context.Background(), req)
	if err != nil {
		return nil, fmt.Errorf("ml service predict: %w", err)
	}
	if len(resp.Predictions) == 0 {
		return nil, fmt.Errorf("empty predictions from ml service")
	}
	// 如果服务返回单值但请求多个样本，展开
	if len(resp.Predictions) == 1 && len(featuresList) > 1 {
		v := resp.Predictions[0]
		out := make([]float64, len(featuresList))
		for i := range out {
			out[i] = v
		}
		return out, nil
	}
	return resp.Predictions, nil
}

// predictViaHTTP 通过直连 HTTP 调用（兼容旧协议）。
// 请求 {"data": [...]}, 响应 {"predictions": [...]}。
func (m *RPCModel) predictViaHTTP(featuresList []map[string]float64) ([]float64, error) {
	if m.Client == nil {
		m.Client = &http.Client{Timeout: m.Timeout}
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

