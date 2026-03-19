package feature

import "testing"

func TestNewStoreFeatureProvider_PanicOnNilStore(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic when store is nil")
		}
	}()

	_ = NewStoreFeatureProvider(nil, KeyPrefix{})
}
