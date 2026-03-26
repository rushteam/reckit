package rerank

import (
	"context"
	"math"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

// SSDDiversityNode 基于滑动子空间多样性（Sliding Subspace Diversity）的 embedding 级重排。
// 论文: https://arxiv.org/pdf/2107.05204
//
// 算法：
//  1. 从 item.Meta[EmbeddingKey] 读取 []float64 向量
//  2. 每步选 argmax(relevance + γ·‖emb‖) 的物品
//  3. 将已选物品的 embedding 分量从所有候选中投影扣除（Gram-Schmidt）
//  4. 滑动窗口到期后恢复最早选入物品的投影分量
//
// 与 DPP 相比计算更轻量（无需构建 n×n 核矩阵），适合在线低延迟场景。
type SSDDiversityNode struct {
	N            int
	Gamma        float64
	WindowSize   int
	EmbeddingKey string
	NormalizeEmb bool
	ScoreNorm    ScoreNormMode
}

func (n *SSDDiversityNode) Name() string {
	return "rerank.ssd_diversity"
}

func (n *SSDDiversityNode) Kind() pipeline.Kind {
	return pipeline.KindReRank
}

func (n *SSDDiversityNode) Process(
	_ context.Context,
	_ *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if len(items) <= 1 {
		return items, nil
	}

	topN := n.N
	if topN <= 0 {
		topN = len(items)
	}
	if topN > len(items) {
		topN = len(items)
	}

	embKey := n.EmbeddingKey
	if embKey == "" {
		embKey = "embedding"
	}

	gamma := n.Gamma
	if gamma == 0 {
		gamma = 0.25
	}
	windowSize := n.WindowSize
	if windowSize <= 1 {
		windowSize = 5
	}

	N := len(items)
	embeddings, dim, err := loadEmbeddings(items, embKey, "rerank.ssd_diversity", n.NormalizeEmb, true)
	if err != nil {
		return nil, err
	}

	relevance := normalizeScores(items, n.ScoreNorm)

	return ssdSlidingWindow(items, embeddings, relevance, dim, N, topN, gamma, windowSize)
}

func ssdSlidingWindow(
	items []*core.Item,
	embeddings [][]float64,
	relevance []float64,
	dim, itemCount, topN int,
	gamma float64,
	windowSize int,
) ([]*core.Item, error) {
	selected := make(map[int]bool, topN)
	indices := make([]int, 0, topN)

	bQueue := make([]int, 0, windowSize)
	pQueue := make([][]float64, 0, windowSize)

	idx := maxIdxFloat(relevance)
	if idx < 0 {
		return items, nil
	}
	selected[idx] = true
	indices = append(indices, idx)
	volume := gamma

	for t := 1; t < topN; t++ {
		if t > windowSize && len(bQueue) > 0 {
			oldIdx := bQueue[0]
			oldProj := pQueue[0]
			bQueue = bQueue[1:]
			pQueue = pQueue[1:]

			embOld := embeddings[oldIdx]
			for j := 0; j < itemCount; j++ {
				if selected[j] {
					continue
				}
				for d := 0; d < dim; d++ {
					embeddings[j][d] += oldProj[j] * embOld[d]
				}
			}
		}

		projections := make([]float64, itemCount)
		embI := embeddings[idx]
		dotII := vecDot(embI, embI)
		for j := 0; j < itemCount; j++ {
			if selected[j] {
				continue
			}
			if dotII == 0 {
				projections[j] = 0
			} else {
				projections[j] = vecDot(embeddings[j], embI) / dotII
			}
			if math.IsNaN(projections[j]) || math.IsInf(projections[j], 0) {
				projections[j] = 1.0
			}
			for d := 0; d < dim; d++ {
				embeddings[j][d] -= projections[j] * embI[d]
			}
		}

		bQueue = append(bQueue, idx)
		pQueue = append(pQueue, projections)

		bestIdx := -1
		bestQ := math.Inf(-1)
		for i := 0; i < itemCount; i++ {
			if selected[i] {
				continue
			}
			norm := vecNorm(embeddings[i])
			if math.IsNaN(norm) || math.IsInf(norm, 0) {
				norm = 0.5
			}
			q := relevance[i] + volume*norm
			if q > bestQ {
				bestQ = q
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}

		idx = bestIdx
		selected[idx] = true
		indices = append(indices, idx)
	}

	out := make([]*core.Item, 0, len(indices))
	for _, i := range indices {
		out = append(out, items[i])
	}
	return out, nil
}
