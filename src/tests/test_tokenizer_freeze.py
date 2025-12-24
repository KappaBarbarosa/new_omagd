"""
Test tokenizer freezing in Stage 2.
"""
import torch
import sys
sys.path.insert(0, '/home/marl2025/new_omagd/src')

from modules.graph_reconstructers.codebook import VectorQuantizer
from modules.graph_reconstructers.node_wise_tokenizer import Tokenizer

def test_vq_ema_in_eval_mode():
    print("=" * 60)
    print("TEST: VQ EMA Update Behavior in Eval Mode")
    print("=" * 60)
    
    vq = VectorQuantizer(code_dim=32, n_codes=1024)
    
    # Set to eval mode
    vq.eval()
    print(f"VQ training mode: {vq.training}")
    
    # Get initial state
    initial_codebook = vq.embedding.weight.clone()
    initial_cluster_size = vq.cluster_size.clone()
    initial_embed_avg = vq.embed_avg.clone()
    
    # Forward with training=False explicitly
    h = torch.randn(100, 32)
    
    # Test 1: Forward with training=True (should update EMA even in eval mode!)
    print("\n1. Forward with training=True:")
    vq.forward(h, training=True)
    
    cluster_changed_1 = not torch.allclose(initial_cluster_size, vq.cluster_size)
    embed_changed_1 = not torch.allclose(initial_embed_avg, vq.embed_avg)
    print(f"   Cluster size changed: {cluster_changed_1}")
    print(f"   Embed avg changed: {embed_changed_1}")
    
    # Reset
    vq.cluster_size.copy_(initial_cluster_size)
    vq.embed_avg.copy_(initial_embed_avg)
    
    # Test 2: Forward with training=False (should NOT update EMA)
    print("\n2. Forward with training=False:")
    vq.forward(h, training=False)
    
    cluster_changed_2 = not torch.allclose(initial_cluster_size, vq.cluster_size)
    embed_changed_2 = not torch.allclose(initial_embed_avg, vq.embed_avg)
    print(f"   Cluster size changed: {cluster_changed_2}")
    print(f"   Embed avg changed: {embed_changed_2}")
    
    print("\n" + "=" * 60)
    if cluster_changed_1 or embed_changed_1:
        print("⚠️  training=True updates EMA (as expected)")
    if not cluster_changed_2 and not embed_changed_2:
        print("✅ training=False does NOT update EMA (correct!)")
    else:
        print("❌ training=False still updates EMA (BUG!)")
    print("=" * 60)


def test_tokenizer_training_mode():
    print("\n" + "=" * 60)
    print("TEST: Tokenizer Training Mode Propagation")
    print("=" * 60)
    
    tokenizer = Tokenizer(in_dim=8, hid=32, code_dim=16, n_codes=512)
    
    # Set to eval
    tokenizer.eval()
    
    print(f"Tokenizer.training: {tokenizer.training}")
    print(f"Tokenizer.vq.training: {tokenizer.vq.training}")
    
    # Check encode_to_tokens uses self.training
    data = {"x": torch.randn(4, 6, 8), "node_types": torch.zeros(4, 6)}
    
    initial_cluster = tokenizer.vq.cluster_size.clone()
    
    # Forward in eval mode
    for _ in range(5):
        with torch.no_grad():
            tokenizer.encode_to_tokens(data)
    
    cluster_changed = not torch.allclose(initial_cluster, tokenizer.vq.cluster_size)
    print(f"\nAfter 5 forwards in eval mode:")
    print(f"  Cluster size changed: {cluster_changed}")
    
    if cluster_changed:
        print("  ❌ BUG: EMA updated in eval mode!")
    else:
        print("  ✅ PASSED: EMA not updated in eval mode")


if __name__ == "__main__":
    test_vq_ema_in_eval_mode()
    test_tokenizer_training_mode()
