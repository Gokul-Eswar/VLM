import torch
import torchvision

print("=" * 50)
print("GPU SETUP VERIFICATION")
print("=" * 50)

# Check PyTorch
print(f"\nPyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")

# Check CUDA
print(f"\nCUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    try:
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except:
        print("Could not retrieve GPU Memory info")

    # Test GPU computation
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print("\n✅ GPU computation test: PASSED")
    except Exception as e:
        print(f"\n❌ GPU computation test: FAILED ({e})")
else:
    print("\n⚠️ WARNING: CUDA not available. Using CPU only.")

print("=" * 50)
