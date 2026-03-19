package model

import (
	"context"
	"fmt"

	"github.com/rushteam/reckit/core"
)

// RankModel 是排序阶段的最小抽象：输入特征，输出一个可比较的分数。
//
// 定位：**本地轻量模型**（LR、GBDT 等可在进程内计算的模型）。
// 签名刻意保持简洁（无 context、单条 in/out），适合嵌入式使用。
//
// 如果需要调用远程模型服务（TF Serving、KServe、TorchServe 等），
// 推荐直接使用 core.MLService 接口 + RPCNode，或通过 MLServiceAdapter 桥接：
//
//	localModel := model.MLServiceAdapter("deepfm", kserveClient)
type RankModel interface {
	Name() string
	Predict(features map[string]float64) (float64, error)
}

// MLServiceAdapter 将 core.MLService（远程推理服务）适配为 RankModel 接口。
// 每次 Predict 调用会发起一次远程请求（单条），适合低 QPS 场景或兼容已有 RankModel 消费方。
// 高 QPS 场景建议直接使用 RPCNode + MLService 的批量接口。
func MLServiceAdapter(name string, svc core.MLService) RankModel {
	return &mlServiceRankModel{name: name, svc: svc}
}

type mlServiceRankModel struct {
	name string
	svc  core.MLService
}

func (m *mlServiceRankModel) Name() string { return m.name }

func (m *mlServiceRankModel) Predict(features map[string]float64) (float64, error) {
	resp, err := m.svc.Predict(context.Background(), &core.MLPredictRequest{
		Features: []map[string]float64{features},
	})
	if err != nil {
		return 0, fmt.Errorf("ml service predict: %w", err)
	}
	if len(resp.Predictions) == 0 {
		return 0, fmt.Errorf("empty predictions from ml service")
	}
	return resp.Predictions[0], nil
}
