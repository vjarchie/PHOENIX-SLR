# Quick system check for training readiness
import torch
import sys

print("=" * 50)
print("SYSTEM CHECK FOR TRAINING")
print("=" * 50)

print(f"\nPython: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")

if torch.cuda.is_available():
    print(f"\n[OK] CUDA available")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  BF16 supported: {torch.cuda.is_bf16_supported()}")
    
    # Quick bf16 test
    try:
        x = torch.randn(100, 100, device='cuda', dtype=torch.bfloat16)
        y = x @ x.T
        print(f"  BF16 compute test: PASSED")
    except Exception as e:
        print(f"  BF16 compute test: FAILED ({e})")
else:
    print("\n[WARNING] CUDA not available - will use CPU (slow)")

print("\n" + "=" * 50)
print("Ready to train!" if torch.cuda.is_available() else "Warning: GPU not available")
print("=" * 50)

