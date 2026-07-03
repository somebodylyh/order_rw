import torch
import time

# 指定你想霸占的显卡ID，例如 'cuda:0' 或 'cuda:1'
device_id = 'cuda:0' 
device = torch.device(device_id)

# 显存计算：1GB float32 数据大约需要 256 * 1024 * 1024 个元素
# 你自己跑实验用 5GB，你想再额外占位 18GB，这样总共占用 23GB，对方就跑不了 10GB 的任务了
gb_to_occupy = 18 

print(f"正在 {device_id} 上分配 {gb_to_occupy}GB 的虚拟显存...")

try:
    # 生成指定大小的无用张量，并强制放置在GPU上
    dummy_tensor = torch.empty((gb_to_occupy, 256, 1024, 1024), dtype=torch.float32, device=device)
    print("显存占用成功！保持运行中... (按 Ctrl+C 释放)")
    
    # 死循环保持进程不死，显存不被释放
    while True:
        time.sleep(100)
except RuntimeError as e:
    print(f"分配失败，可能是显卡剩余显存不足：{e}")