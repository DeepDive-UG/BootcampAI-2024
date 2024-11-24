import numpy as np
import torch
import time

matrix_size = 1000  # Adjust this value for different matrix sizes
matrix_a = np.random.rand(matrix_size, matrix_size)
matrix_b = np.random.rand(matrix_size, matrix_size)

start_time = time.time()
result_cpu = np.dot(matrix_a, matrix_b)
end_time = time.time()
cpu_time = end_time - start_time
print(f"CPU time: {cpu_time:.4f} seconds")

# Convert NumPy arrays to PyTorch tensors
tensor_a = torch.from_numpy(matrix_a).cuda()
tensor_b = torch.from_numpy(matrix_b).cuda()

start_time = time.time()
result_gpu = torch.matmul(tensor_a, tensor_b)
end_time = time.time()
gpu_time = end_time - start_time
print(f"GPU time: {gpu_time:.4f} seconds")

print(f"Speedup: {cpu_time / gpu_time:.2f}x")


