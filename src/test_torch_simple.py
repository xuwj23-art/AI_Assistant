# 新建一个 test_torch_simple.py 文件
import torch
print("CUDA可用:", torch.cuda.is_available())
print("PyTorch版本:", torch.__version__)

# 测试一个简单的张量运算
try:
    a = torch.randn(3, 3)
    print("张量创建成功")
    print(a @ a.t())
    print("✅ PyTorch 核心功能正常")
except Exception as e:
    print("❌ 运行失败:", e)