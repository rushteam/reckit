package feature

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
)

func TestDefaultFeatureExtractor_ExtractFromParams(t *testing.T) {
	tests := []struct {
		name           string
		opts           []DefaultFeatureExtractorOption
		params         map[string]any
		wantFeatures   map[string]float64
		wantKeyPrefix  string
	}{
		{
			name: "extract all params without prefix (default)",
			opts: []DefaultFeatureExtractorOption{},
			params: map[string]any{
				"latitude":  39.9042,
				"longitude": 116.4074,
				"count":     int(10),
			},
			wantFeatures: map[string]float64{
				"latitude":  39.9042,
				"longitude": 116.4074,
				"count":     10.0,
			},
		},
		{
			name: "extract params with custom prefix",
			opts: []DefaultFeatureExtractorOption{
				WithParamsPrefix("ctx_"),
			},
			params: map[string]any{
				"latitude": 39.9042,
			},
			wantFeatures: map[string]float64{
				"ctx_latitude": 39.9042,
			},
		},
		{
			name: "extract only specified keys",
			opts: []DefaultFeatureExtractorOption{
				WithParamsKeys([]string{"latitude", "longitude"}),
			},
			params: map[string]any{
				"latitude":  39.9042,
				"longitude": 116.4074,
				"other":     999.0,
			},
			wantFeatures: map[string]float64{
				"latitude":  39.9042,
				"longitude": 116.4074,
			},
		},
		{
			name: "skip non-float64 values",
			opts: []DefaultFeatureExtractorOption{},
			params: map[string]any{
				"latitude": 39.9042,
				"name":     "test",          // string, skip
				"tags":     []string{"a"},   // slice, skip
			},
			wantFeatures: map[string]float64{
				"latitude": 39.9042,
			},
		},
		{
			name: "params disabled explicitly",
			opts: []DefaultFeatureExtractorOption{
				WithIncludeParams(false),
			},
			params: map[string]any{
				"latitude": 39.9042,
			},
			wantFeatures: map[string]float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			extractor := NewDefaultFeatureExtractor(tt.opts...)
			rctx := &core.RecommendContext{
				UserID: "test_user",
				Params: tt.params,
			}

			features, err := extractor.Extract(context.Background(), rctx)
			if err != nil {
				t.Fatalf("Extract() error = %v", err)
			}

			// Check expected features
			for key, wantVal := range tt.wantFeatures {
				if gotVal, ok := features[key]; !ok {
					t.Errorf("missing expected feature %q", key)
				} else if gotVal != wantVal {
					t.Errorf("feature %q = %v, want %v", key, gotVal, wantVal)
				}
			}


			// Check specified keys only
			if len(extractor.ParamsKeys) > 0 {
				for key := range features {
					found := false
					for _, allowedKey := range extractor.ParamsKeys {
						if key == extractor.ParamsPrefix+allowedKey {
							found = true
							break
						}
					}
					// Only check params_ prefixed keys
					if len(key) > len(extractor.ParamsPrefix) && !found {
						prefix := extractor.ParamsPrefix
						if prefix == "" {
							prefix = "params_"
						}
						if len(key) > len(prefix) && key[:len(prefix)] == prefix {
							t.Errorf("unexpected feature %q not in ParamsKeys", key)
						}
					}
				}
			}
		})
	}
}

func TestDefaultFeatureExtractor_CombineWithUserProfile(t *testing.T) {
	extractor := NewDefaultFeatureExtractor(
		WithParamsPrefix("ctx_"),
	)

	rctx := &core.RecommendContext{
		UserID: "test_user",
		User: &core.UserProfile{
			Age:    25,
			Gender: "male",
		},
		Params: map[string]any{
			"latitude":    39.9042,
			"time_of_day": 14.5,
		},
	}

	features, err := extractor.Extract(context.Background(), rctx)
	if err != nil {
		t.Fatalf("Extract() error = %v", err)
	}

	// Check user features
	if features["age"] != 25.0 {
		t.Errorf("age = %v, want 25.0", features["age"])
	}
	if features["gender"] != 1.0 { // male = 1.0
		t.Errorf("gender = %v, want 1.0", features["gender"])
	}

	// Check params features
	if features["ctx_latitude"] != 39.9042 {
		t.Errorf("ctx_latitude = %v, want 39.9042", features["ctx_latitude"])
	}
	if features["ctx_time_of_day"] != 14.5 {
		t.Errorf("ctx_time_of_day = %v, want 14.5", features["ctx_time_of_day"])
	}
}
