package rerank

import (
	"fmt"
	"math"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/conv"
)

func vecDot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func vecNorm(a []float64) float64 {
	return math.Sqrt(vecDot(a, a))
}

func vecNormalize(dst, src []float64) {
	n := vecNorm(src)
	if n == 0 {
		copy(dst, src)
		return
	}
	inv := 1.0 / n
	for i := range src {
		dst[i] = src[i] * inv
	}
}

func vecScale(dst []float64, s float64, src []float64) {
	for i := range src {
		dst[i] = src[i] * s
	}
}

func vecSub(dst, a, b []float64) {
	for i := range a {
		dst[i] = a[i] - b[i]
	}
}

func vecAdd(dst, a, b []float64) {
	for i := range a {
		dst[i] = a[i] + b[i]
	}
}

// maxIdxFloat 返回 slice 中最大值的索引，跳过 NaN 和 -Inf。
func maxIdxFloat(a []float64) int {
	best := -1
	bestVal := math.Inf(-1)
	for i, v := range a {
		if math.IsNaN(v) || math.IsInf(v, -1) {
			continue
		}
		if v > bestVal {
			bestVal = v
			best = i
		}
	}
	return best
}

// ---------------------------------------------------------------------------
// DPP / SSD 共享：embedding 加载 + 分数标准化
// ---------------------------------------------------------------------------

// ScoreNormMode 相关性分数标准化模式。
type ScoreNormMode int

const (
	ScoreNormNone   ScoreNormMode = iota
	ScoreNormZScore               // Z-Score: (x-μ)/σ
	ScoreNormMinMax               // Min-Max → [ε, 1]
)

func normalizeScores(items []*core.Item, mode ScoreNormMode) []float64 {
	scores := make([]float64, len(items))
	for i, it := range items {
		scores[i] = it.Score
	}
	switch mode {
	case ScoreNormZScore:
		mean, variance := meanVariance(scores)
		if variance > 0 {
			std := math.Sqrt(variance)
			for i := range scores {
				scores[i] = (scores[i] - mean) / std
			}
		}
	case ScoreNormMinMax:
		minS, maxS := scores[0], scores[0]
		for _, s := range scores[1:] {
			if s < minS {
				minS = s
			}
			if s > maxS {
				maxS = s
			}
		}
		span := maxS - minS
		if span > 0 {
			eps := 1e-6
			for i := range scores {
				scores[i] = ((scores[i]-minS)/span)*(1-eps) + eps
			}
		}
	}
	return scores
}

func meanVariance(data []float64) (float64, float64) {
	if len(data) == 0 {
		return 0, 0
	}
	mean := 0.0
	for _, v := range data {
		mean += v
	}
	mean /= float64(len(data))
	variance := 0.0
	for _, v := range data {
		d := v - mean
		variance += d * d
	}
	variance /= float64(len(data))
	return mean, variance
}

// loadEmbeddings 从 item.Meta[key] 加载 embedding 向量。
// clone=true 时复制一份（SSD 会原地修改）；normalize=true 时做 L2 归一化。
func loadEmbeddings(items []*core.Item, key, nodeName string, normalize, clone bool) ([][]float64, int, error) {
	embeddings := make([][]float64, len(items))
	dim := 0
	for i, it := range items {
		raw, ok := it.Meta[key]
		if !ok {
			return nil, 0, fmt.Errorf("%s: item %q missing Meta[%q]", nodeName, it.ID, key)
		}
		emb, err := toFloat64Slice(raw)
		if err != nil {
			return nil, 0, fmt.Errorf("%s: item %q Meta[%q]: %w", nodeName, it.ID, key, err)
		}
		if len(emb) == 0 {
			return nil, 0, fmt.Errorf("%s: item %q has empty embedding", nodeName, it.ID)
		}
		if dim == 0 {
			dim = len(emb)
		} else if len(emb) != dim {
			return nil, 0, fmt.Errorf("%s: item %q embedding dim %d != %d", nodeName, it.ID, len(emb), dim)
		}

		if clone {
			c := make([]float64, len(emb))
			copy(c, emb)
			emb = c
		}
		if normalize {
			norm := vecNorm(emb)
			if norm > 0 {
				if !clone {
					normed := make([]float64, len(emb))
					vecScale(normed, 1.0/norm, emb)
					emb = normed
				} else {
					vecScale(emb, 1.0/norm, emb)
				}
			}
		}
		embeddings[i] = emb
	}
	return embeddings, dim, nil
}

func toFloat64Slice(v any) ([]float64, error) {
	switch t := v.(type) {
	case []float64:
		return t, nil
	case []any:
		out := make([]float64, 0, len(t))
		for _, x := range t {
			f, ok := conv.ToFloat64(x)
			if !ok {
				return nil, fmt.Errorf("cannot convert %T to float64", x)
			}
			out = append(out, f)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported embedding type %T", v)
	}
}
