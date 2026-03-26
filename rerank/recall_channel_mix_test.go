package rerank

import (
	"context"
	"math/rand"
	"testing"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/utils"
)

func TestPrimaryRecallChannel_Pipe(t *testing.T) {
	it := core.NewItem("x")
	it.PutLabel("recall_source", utils.Label{Value: "hot|cf", Source: "s"})
	if p := PrimaryRecallChannel(it, "recall_source"); p != "hot" {
		t.Fatalf("got %q", p)
	}
}

func TestRecallChannelMix_FixedSlot(t *testing.T) {
	h := core.NewItem("h")
	h.PutLabel("recall_source", utils.Label{Value: "hot", Source: "r"})
	c := core.NewItem("c")
	c.PutLabel("recall_source", utils.Label{Value: "cf", Source: "r"})
	items := []*core.Item{h, c}
	mix := &RecallChannelMix{
		OutputSize: 3,
		Rules: []ChannelRule{
			{
				Kind:     ChannelSlotKindFixed,
				Channels: []string{"cf"},
				FixedSlots: []int{0},
			},
			{
				Kind:     ChannelSlotKindFixed,
				Channels: []string{"hot"},
				FixedSlots: []int{2},
			},
		},
	}
	out, err := mix.Process(context.Background(), &core.RecommendContext{}, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 || out[0].ID != "c" || out[1].ID != "h" {
		t.Fatalf("got len=%d ids %v %v", len(out), out[0].ID, out[1].ID)
	}
}

func TestRecallChannelMix_RemainderDiscard(t *testing.T) {
	a := core.NewItem("a")
	a.PutLabel("recall_source", utils.Label{Value: "hot", Source: "r"})
	b := core.NewItem("b")
	b.PutLabel("recall_source", utils.Label{Value: "hot", Source: "r"})
	mix := &RecallChannelMix{
		OutputSize:      1,
		RemainderPolicy: RemainderDiscard,
		Rules: []ChannelRule{
			{
				Kind:     ChannelSlotKindFixed,
				Channels: []string{"hot"},
				FixedSlots: []int{0},
			},
		},
	}
	out, err := mix.Process(context.Background(), &core.RecommendContext{}, []*core.Item{a, b})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 || out[0].ID != "a" {
		t.Fatalf("len=%d id=%s", len(out), out[0].ID)
	}
}

func TestRecallChannelMix_RandomDeterministic(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	a := core.NewItem("a")
	a.PutLabel("recall_source", utils.Label{Value: "hot", Source: "r"})
	b := core.NewItem("b")
	b.PutLabel("recall_source", utils.Label{Value: "hot", Source: "r"})
	mix := &RecallChannelMix{
		OutputSize: 4,
		Rand:       rng,
		Rules: []ChannelRule{
			{
				Kind:            ChannelSlotKindRandom,
				Channels:        []string{"hot"},
				RandomSlotStart: 1,
				RandomSlotEnd:   3,
				RandomCount:     1,
			},
		},
	}
	out, err := mix.Process(context.Background(), &core.RecommendContext{}, []*core.Item{a, b})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) < 2 {
		t.Fatalf("len=%d", len(out))
	}
}
