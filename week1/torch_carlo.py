import torch
import matplotlib.pyplot as plt

# Set the number of points to generate
num_points = 1000000

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Generate random points on the GPU
x = (torch.rand(num_points, device=device) * 2) - 1
y = (torch.rand(num_points, device=device) * 2) - 1

# Calculate the distance from the origin on the GPU
distance = torch.sqrt(x**2 + y**2)

# Count the points inside the circle on the GPU
inside_circle = (distance <= 1).sum()

# Estimate pi
pi_estimate = 4 * inside_circle / num_points

# Print the estimated value of pi
print("Estimated value of pi:", pi_estimate.item())

# Move data back to CPU for visualization
x = x.cpu()
y = y.cpu()
distance = distance.cpu()

# Visualize the points
plt.figure(figsize=(6, 6))
plt.scatter(x, y, c=(distance <= 1), s=5)
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Monte Carlo Estimation of Pi {"GPU Accelerated" if torch.cuda.is_available() else "On CPU"}")
plt.show()