import torch

def occupy_gpu_memory(gpu_id, memory_size_mb):
    # 选择指定的 GPU
    torch.cuda.set_device(gpu_id)
    # 计算需要占用的显存大小（以字节为单位）
    memory_size_bytes = memory_size_mb * 1024 * 1024
    # 计算所需的元素数量
    num_elements = memory_size_bytes // 4
    # 创建一个大的张量并将其移动到指定的 GPU 上
    tensor = torch.zeros(num_elements, dtype=torch.float32, device=torch.device('cuda'))
    print(f"成功占用 {memory_size_mb} MB 的 GPU 显存。")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("程序已停止，显存占用已释放。")

if __name__ == "__main__":
    gpu_id = 0  # 指定要占用的 GPU 编号
    memory_size_mb = 2048  # 指定要占用的显存大小（MB）
    occupy_gpu_memory(gpu_id, memory_size_mb)