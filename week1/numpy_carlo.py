import numpy as np
import matplotlib.pyplot as plt

# Set the number of points to generate
num_points = 1000000

x = np.random.uniform(-1, 1, num_points)
y = np.random.uniform(-1, 1, num_points)

distance = np.sqrt(x**2 + y**2)

inside_circle = (distance <= 1).sum()

print(inside_circle)

pi_estimate = 4 * inside_circle / num_points

print(f"Estimated Pi Value: {pi_estimate}")

# Visualize the points
plt.figure(figsize=(6, 6))
plt.scatter(x, y, c=(distance <= 1), s=5)
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Monte Carlo Estimation of Pi NumPy")
plt.show()