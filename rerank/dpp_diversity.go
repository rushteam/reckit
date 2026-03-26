package rerank

import (
	"context"
	"math"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

// DPPDiversityNode 基于 Determinantal Point Process 的 embedding 级多样性重排。
//
// 算法：
//  1. 从 item.Meta[EmbeddingKey] 读取 []float64 向量
//  2. 构建 DPP 核矩阵 K = diag(r) · S · diag(r)，r_i = exp(α · score_i)
//  3. Greedy MAP 近似选取 top-N
//  4. 窗口化处理大列表（WindowSize）
type DPPDiversityNode struct {
	N            int
	Alpha        float64
	WindowSize   int
	EmbeddingKey string
	NormalizeEmb bool
	ScoreNorm    ScoreNormMode
}

func (n *DPPDiversityNode) Name() string {
	return "rerank.dpp_diversity"
}

func (n *DPPDiversityNode) Kind() pipeline.Kind {
	return pipeline.KindReRank
}

func (n *DPPDiversityNode) Process(
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

	embeddings, dim, err := loadEmbeddings(items, embKey, "rerank.dpp_diversity", n.NormalizeEmb, false)
	if err != nil {
		return nil, err
	}

	relevance := normalizeScores(items, n.ScoreNorm)

	kernel := n.buildKernel(embeddings, relevance, dim, len(items))

	var indices []int
	if n.WindowSize > 0 && topN > n.WindowSize {
		indices = dppGreedyWindowed(kernel, len(items), topN, n.WindowSize)
	} else {
		indices = dppGreedy(kernel, len(items), topN, nil)
	}

	out := make([]*core.Item, 0, len(indices))
	for _, idx := range indices {
		if idx >= 0 && idx < len(items) {
			out = append(out, items[idx])
		}
	}
	return out, nil
}

// buildKernel 构建 DPP 核矩阵 K = diag(r) · S · diag(r)。
// 特征行: [emb, 1] / sqrt(2) → S = F·Fᵀ。
func (n *DPPDiversityNode) buildKernel(embeddings [][]float64, relevance []float64, dim, itemSize int) []float64 {
	alpha := n.Alpha
	featureDim := dim + 1
	features := make([]float64, itemSize*featureDim)
	r := make([]float64, itemSize)
	invSqrt2 := 1.0 / math.Sqrt2

	for i := range embeddings {
		row := features[i*featureDim : (i+1)*featureDim]
		copy(row, embeddings[i])
		row[dim] = 1.0
		for j := range row {
			row[j] *= invSqrt2
		}
		r[i] = math.Exp(alpha * relevance[i])
	}

	kernel := make([]float64, itemSize*itemSize)
	for i := 0; i < itemSize; i++ {
		fi := features[i*featureDim : (i+1)*featureDim]
		for j := i; j < itemSize; j++ {
			fj := features[j*featureDim : (j+1)*featureDim]
			sim := vecDot(fi, fj)
			v := r[i] * sim * r[j]
			kernel[i*itemSize+j] = v
			kernel[j*itemSize+i] = v
		}
	}
	return kernel
}

func dppGreedy(kernel []float64, n, topN int, excluded []bool) []int {
	const eps = 1e-10
	if topN > n {
		topN = n
	}

	d2 := make([]float64, n)
	for i := 0; i < n; i++ {
		if excluded != nil && excluded[i] {
			d2[i] = math.NaN()
		} else {
			d2[i] = kernel[i*n+i]
		}
	}

	j := maxIdxFloat(d2)
	if j < 0 {
		return nil
	}
	Y := make([]int, 0, topN)
	Y = append(Y, j)

	c := make([][]float64, 0, topN)

	for len(Y) < topN {
		dj := d2[j]
		if dj < eps {
			break
		}
		sqrtDj := math.Sqrt(dj)

		e := make([]float64, n)
		for i := 0; i < n; i++ {
			e[i] = kernel[j*n+i]
		}
		for _, ck := range c {
			cj := ck[j]
			for i := 0; i < n; i++ {
				e[i] -= cj * ck[i]
			}
		}
		invDj := 1.0 / sqrtDj
		for i := 0; i < n; i++ {
			e[i] *= invDj
		}
		c = append(c, e)

		for i := 0; i < n; i++ {
			d2[i] -= e[i] * e[i]
		}
		d2[j] = math.NaN()
		j = maxIdxFloat(d2)
		if j < 0 {
			break
		}
		Y = append(Y, j)
	}

	if len(Y) < topN {
		for i := 0; i < n; i++ {
			if (excluded == nil || !excluded[i]) && !inIntSlice(Y, i) {
				Y = append(Y, i)
				if len(Y) == topN {
					break
				}
			}
		}
	}
	return Y
}

func dppGreedyWindowed(kernel []float64, n, topN, windowSize int) []int {
	result := make([]int, 0, topN)
	excluded := make([]bool, n)

	fullRounds := topN / windowSize
	remainder := topN % windowSize

	for i := 0; i < fullRounds; i++ {
		sub := dppGreedy(kernel, n, windowSize, excluded)
		for _, idx := range sub {
			excluded[idx] = true
		}
		result = append(result, sub...)
	}
	if remainder > 0 {
		sub := dppGreedy(kernel, n, remainder, excluded)
		result = append(result, sub...)
	}
	return result
}

func inIntSlice(a []int, v int) bool {
	for _, x := range a {
		if x == v {
			return true
		}
	}
	return false
}
