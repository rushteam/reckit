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
	Model model.RankModel
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
		// 转换特征名：去掉前缀（user_, item_, cross_），以匹配 Python 训练时的特征名
		features := n.normalizeFeatures(it.Features)
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

// normalizeFeatures 将带前缀的特征名转换为原始特征名，以匹配 Python 训练时的特征名
// 例如：item_ctr -> ctr, user_age -> age, cross_age_x_ctr -> age_x_ctr
func (n *RPCNode) normalizeFeatures(features map[string]float64) map[string]float64 {
	normalized := make(map[string]float64)
	prefixes := []string{"item_", "user_", "cross_"}

	for k, v := range features {
		// 尝试去掉前缀
		originalKey := k
		for _, prefix := range prefixes {
			if strings.HasPrefix(k, prefix) {
				originalKey = strings.TrimPrefix(k, prefix)
				break
			}
		}
		normalized[originalKey] = v
	}

	return normalized
}
