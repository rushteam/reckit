package model

// RankModel 是排序阶段的最小抽象：输入特征，输出一个可比较的分数。
// 具体实现可以是本地模型（LR/GBDT）或远程 RPC（Torch/TF Serving/向量服务）。
type RankModel interface {
	Name() string
	Predict(features map[string]float64) (float64, error)
}
