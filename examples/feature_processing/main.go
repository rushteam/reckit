package main

import (
	"fmt"
	"log"

	"github.com/rushteam/reckit/feature"
)

func main() {
	fmt.Println("=== 特征处理工具类示例 ===")
	fmt.Println()

	// 1. Z-score 标准化示例
	fmt.Println("1. Z-score 标准化")
	mean := map[string]float64{
		"age":   30.0,
		"price": 100.0,
	}
	std := map[string]float64{
		"age":   10.0,
		"price": 50.0,
	}
	zScoreNormalizer := feature.NewZScoreNormalizer(mean, std)
	features := map[string]float64{
		"age":   25.0,
		"price": 150.0,
	}
	normalized := zScoreNormalizer.Normalize(features)
	fmt.Printf("  原始特征: %v\n", features)
	fmt.Printf("  标准化后: %v\n", normalized)
	fmt.Println()

	// 2. Min-Max 归一化示例
	fmt.Println("2. Min-Max 归一化")
	min := map[string]float64{
		"age":   18.0,
		"price": 10.0,
	}
	max := map[string]float64{
		"age":   60.0,
		"price": 200.0,
	}
	minMaxNormalizer := feature.NewMinMaxNormalizer(min, max)
	normalized = minMaxNormalizer.Normalize(features)
	fmt.Printf("  原始特征: %v\n", features)
	fmt.Printf("  归一化后: %v\n", normalized)
	fmt.Println()

	// 3. Log 变换示例
	fmt.Println("3. Log 变换")
	logNormalizer := feature.NewLogNormalizer()
	countFeatures := map[string]float64{
		"click_count": 1000.0,
		"view_count":  5000.0,
	}
	normalized = logNormalizer.Normalize(countFeatures)
	fmt.Printf("  原始特征: %v\n", countFeatures)
	fmt.Printf("  Log 变换后: %v\n", normalized)
	fmt.Println()

	// 4. One-Hot 编码示例
	fmt.Println("4. One-Hot 编码")
	categories := map[string][]string{
		"gender": {"male", "female", "unknown"},
		"city":   {"beijing", "shanghai", "guangzhou"},
	}
	oneHotEncoder := feature.NewOneHotEncoder(categories).WithPrefix("onehot")
	categoricalFeatures := map[string]interface{}{
		"gender": "male",
		"city":   "beijing",
	}
	encoded := oneHotEncoder.EncodeFeatures(categoricalFeatures)
	fmt.Printf("  原始特征: %v\n", categoricalFeatures)
	fmt.Printf("  One-Hot 编码后: %v\n", encoded)
	fmt.Println()

	// 5. Label 编码示例
	fmt.Println("5. Label 编码")
	labelMap := map[string]map[string]int{
		"level": {"low": 0, "medium": 1, "high": 2},
	}
	labelEncoder := feature.NewLabelEncoder(labelMap)
	levelFeatures := map[string]interface{}{
		"level": "high",
	}
	encoded = labelEncoder.EncodeFeatures(levelFeatures)
	fmt.Printf("  原始特征: %v\n", levelFeatures)
	fmt.Printf("  Label 编码后: %v\n", encoded)
	fmt.Println()

	// 6. Hash 编码示例
	fmt.Println("6. Hash 编码")
	hashEncoder := feature.NewHashEncoder(1000).WithPrefix("hash")
	idFeatures := map[string]interface{}{
		"user_id": "user_12345",
		"item_id": "item_67890",
	}
	encoded = hashEncoder.EncodeFeatures(idFeatures)
	fmt.Printf("  原始特征: %v\n", idFeatures)
	fmt.Printf("  Hash 编码后: %v\n", encoded)
	fmt.Println()

	// 6.1. Embedding 编码示例
	fmt.Println("6.1. Embedding 编码（嵌入编码）")
	embeddings := map[string]map[string][]float64{
		"category": {
			"electronics": []float64{0.1, 0.2, 0.3, 0.4},
			"clothing":    []float64{0.2, 0.1, 0.4, 0.3},
			"books":       []float64{0.3, 0.3, 0.2, 0.2},
		},
	}
	embeddingEncoder := feature.NewEmbeddingEncoder(embeddings).WithPrefix("emb")
	embeddingFeatures := map[string]interface{}{
		"category": "electronics",
	}
	encoded = embeddingEncoder.EncodeFeatures(embeddingFeatures)
	fmt.Printf("  原始特征: %v\n", embeddingFeatures)
	fmt.Printf("  Embedding 编码后: %v\n", encoded)
	fmt.Println("  说明: Embedding 编码需要预训练的 embedding 表，通常通过深度学习模型训练得到")
	fmt.Println()

	// 6.2. Ordinal 编码示例
	fmt.Println("6.2. Ordinal 编码（有序编码）")
	orderMap := map[string][]string{
		"level": {"low", "medium", "high"},
		"size":  {"S", "M", "L", "XL"},
	}
	ordinalEncoder := feature.NewOrdinalEncoder(orderMap)
	ordinalFeatures := map[string]interface{}{
		"level": "high",
		"size":  "L",
	}
	encoded = ordinalEncoder.EncodeFeatures(ordinalFeatures)
	fmt.Printf("  原始特征: %v\n", ordinalFeatures)
	fmt.Printf("  Ordinal 编码后: %v\n", encoded)
	fmt.Println()

	// 6.3. Count 编码示例
	fmt.Println("6.3. Count 编码（计数编码）")
	counts := map[string]map[string]int64{
		"category": {
			"electronics": 10000,
			"clothing":    5000,
			"books":       2000,
		},
	}
	countEncoder := feature.NewCountEncoder(counts)
	countFeaturesForEncode := map[string]interface{}{
		"category": "electronics",
	}
	encoded = countEncoder.EncodeFeatures(countFeaturesForEncode)
	fmt.Printf("  原始特征: %v\n", countFeaturesForEncode)
	fmt.Printf("  Count 编码后: %v\n", encoded)
	fmt.Println()

	// 6.4. WoE 编码示例
	fmt.Println("6.4. WoE 编码（证据权重编码）")
	woeMap := map[string]map[string]float64{
		"category": {
			"electronics": 0.5,
			"clothing":    -0.3,
			"books":       0.1,
		},
	}
	woeEncoder := feature.NewWOEEncoder(woeMap)
	woeFeatures := map[string]interface{}{
		"category": "electronics",
	}
	encoded = woeEncoder.EncodeFeatures(woeFeatures)
	fmt.Printf("  原始特征: %v\n", woeFeatures)
	fmt.Printf("  WoE 编码后: %v\n", encoded)
	fmt.Println("  说明: WoE 编码常用于风控和信用评分场景")
	fmt.Println()

	// 7. 交叉特征生成示例
	fmt.Println("7. 交叉特征生成")
	userFeatures := map[string]float64{
		"age":    25.0,
		"gender": 1.0,
	}
	itemFeatures := map[string]float64{
		"ctr":   0.15,
		"price": 99.0,
	}
	generator := feature.NewCrossFeatureGenerator(
		[]string{"age", "gender"},
		[]string{"ctr", "price"},
	).WithOperations([]string{"multiply", "divide"})
	crossFeatures := generator.Generate(userFeatures, itemFeatures)
	fmt.Printf("  用户特征: %v\n", userFeatures)
	fmt.Printf("  物品特征: %v\n", itemFeatures)
	fmt.Printf("  交叉特征: %v\n", crossFeatures)
	fmt.Println()

	// 8. 分桶示例
	fmt.Println("8. 等宽分桶")
	numBins := map[string]int{
		"age":   5,
		"price": 10,
	}
	binner := feature.NewEqualWidthBinner(min, max, numBins)
	binned := binner.BinFeatures(features)
	fmt.Printf("  原始特征: %v\n", features)
	fmt.Printf("  分桶后: %v\n", binned)
	fmt.Println()

	// 9. 缺失值处理示例
	fmt.Println("9. 缺失值处理")
	handler := feature.NewMissingValueHandler("zero", 0.0).
		WithDefaultValues(map[string]float64{
			"age":    25.0,
			"gender": 0.0,
		})
	incompleteFeatures := map[string]float64{
		"age": 30.0,
		// gender 和 city 缺失
	}
	requiredFeatures := []string{"age", "gender", "city"}
	handled := handler.Handle(incompleteFeatures, requiredFeatures)
	fmt.Printf("  原始特征: %v\n", incompleteFeatures)
	fmt.Printf("  必需特征: %v\n", requiredFeatures)
	fmt.Printf("  处理后: %v\n", handled)
	fmt.Println()

	// 10. 特征选择示例
	fmt.Println("10. 特征选择")
	allFeatures := map[string]float64{
		"age":           25.0,
		"gender":        1.0,
		"ctr":           0.15,
		"price":         99.0,
		"debug_feature": 1.0,
		"other":         100.0,
	}
	selector := feature.NewFeatureSelector([]string{"age", "gender", "ctr"}).
		WithExcludedFeatures([]string{"debug_feature"})
	selected := selector.Select(allFeatures)
	fmt.Printf("  所有特征: %v\n", allFeatures)
	fmt.Printf("  选择后: %v\n", selected)
	fmt.Println()

	// 11. 特征统计示例
	fmt.Println("11. 特征统计")
	values := []float64{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
	stats := feature.ComputeStatistics(values)
	fmt.Printf("  数据: %v\n", values)
	fmt.Printf("  均值: %.2f\n", stats.Mean)
	fmt.Printf("  标准差: %.2f\n", stats.Std)
	fmt.Printf("  最小值: %.2f\n", stats.Min)
	fmt.Printf("  最大值: %.2f\n", stats.Max)
	fmt.Printf("  中位数: %.2f\n", stats.Median)
	fmt.Printf("  P25: %.2f\n", stats.P25)
	fmt.Printf("  P75: %.2f\n", stats.P75)
	fmt.Printf("  P95: %.2f\n", stats.P95)
	fmt.Printf("  P99: %.2f\n", stats.P99)
	fmt.Println()

	// 12. 完整流程示例
	fmt.Println("12. 完整特征处理流程")
	fmt.Println("  原始特征 -> 缺失值处理 -> 编码 -> 归一化 -> 交叉特征 -> 最终特征")
	
	// 步骤1: 缺失值处理
	rawFeatures := map[string]float64{
		"age": 25.0,
		// gender 缺失
	}
	step1 := handler.Handle(rawFeatures, []string{"age", "gender", "city"})
	fmt.Printf("  步骤1 (缺失值处理): %v\n", step1)
	
	// 步骤2: 编码（假设 gender 需要编码）
	genderEncoded := oneHotEncoder.EncodeWithKey("gender", 1)
	fmt.Printf("  步骤2 (编码): gender -> %v\n", genderEncoded)
	
	// 步骤3: 归一化
	step3 := zScoreNormalizer.Normalize(step1)
	fmt.Printf("  步骤3 (归一化): %v\n", step3)
	
	// 步骤4: 交叉特征
	step4 := generator.Generate(step3, itemFeatures)
	fmt.Printf("  步骤4 (交叉特征): %v\n", step4)
	
	// 步骤5: 合并所有特征
	finalFeatures := make(map[string]float64)
	for k, v := range step3 {
		finalFeatures["user_"+k] = v
	}
	for k, v := range genderEncoded {
		finalFeatures[k] = v
	}
	for k, v := range step4 {
		finalFeatures["cross_"+k] = v
	}
	fmt.Printf("  最终特征: %v\n", finalFeatures)

	fmt.Println("\n=== 示例完成 ===")
}

func init() {
	// 设置日志格式
	log.SetFlags(log.LstdFlags | log.Lshortfile)
}
