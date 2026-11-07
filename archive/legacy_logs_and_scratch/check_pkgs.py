import torch
import transformers
print(f"Torch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
print(f"Transformers: {transformers.__version__}")
