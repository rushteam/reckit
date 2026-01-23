package rank

import (
	"context"
	"sort"
	"strings"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/model"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/utils"
)

// RPCNode 是通过 RPC 调用外部模型服务的排序 Node。
// 支持 GBDT、XGBoost、TensorFlow Serving 等。
type RPCNode struct {
	// Model 排序模型，用于 RPC 预测
	Model model.RankModel

	// StripFeaturePrefix 是否去掉特征名前缀后再发给模型服务。
	//
	// - false（默认）：不去掉前缀，直接传递 EnrichNode 产出的特征名（如 user_age、item_ctr、cross_age_x_ctr）。
	//   训练时 FEATURE_COLUMNS 需使用带前缀的名称，与在线一致。
	//
	// - true：去掉 user_、item_、cross_、scene_ 等前缀，转换为无前缀特征名（如 age、ctr、age_x_ctr）。
	//   适用于训练时 FEATURE_COLUMNS 使用无前缀名称的旧模型或外部模型。
	StripFeaturePrefix bool
}

func (n *RPCNode) Name() string        { return "rank.rpc" }
func (n *RPCNode) Kind() pipeline.Kind { return pipeline.KindRank }

func (n *RPCNode) Process(
	ctx context.Context,
	_ *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if n.Model == nil || len(items) == 0 {
		return items, nil
	}

	// 收集所有有效的特征
	validItems := make([]*core.Item, 0, len(items))
	featuresList := make([]map[string]float64, 0, len(items))

	for _, it := range items {
		if it == nil {
			continue
		}
		// 按 StripFeaturePrefix 决定是否去掉特征前缀；默认不去掉，与 FEATURE_COLUMNS 带前缀对齐
		features := it.Features
		if features == nil {
			features = make(map[string]float64)
		}
		if n.StripFeaturePrefix {
			features = n.stripFeaturePrefix(features)
		}
		validItems = append(validItems, it)
		featuresList = append(featuresList, features)
	}

	if len(featuresList) == 0 {
		return items, nil
	}

	// 批量预测
	rpcModel, ok := n.Model.(*model.RPCModel)
	if !ok {
		// 如果不是 RPCModel，回退到单个预测
		for i, it := range validItems {
			score, err := n.Model.Predict(featuresList[i])
			if err != nil {
				return nil, err
			}
			it.Score = score
			it.PutLabel("rank_model", utils.Label{Value: n.Model.Name(), Source: "rank"})
			it.PutLabel("rank_type", utils.Label{Value: "rpc", Source: "rank"})
		}
	} else {
		// 使用批量预测
		scores, err := rpcModel.PredictBatch(featuresList)
		if err != nil {
			return nil, err
		}
		for i, it := range validItems {
			it.Score = scores[i]
			it.PutLabel("rank_model", utils.Label{Value: n.Model.Name(), Source: "rank"})
			it.PutLabel("rank_type", utils.Label{Value: "rpc", Source: "rank"})
		}
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

// stripFeaturePrefix 去掉 user_、item_、cross_、scene_ 等前缀，得到无前缀特征名。
// 仅在 StripFeaturePrefix == true 时使用，用于兼容训练时 FEATURE_COLUMNS 为无前缀的模型。
// 例如：item_ctr -> ctr, user_age -> age, cross_age_x_ctr -> age_x_ctr, scene_id -> id
func (n *RPCNode) stripFeaturePrefix(features map[string]float64) map[string]float64 {
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
