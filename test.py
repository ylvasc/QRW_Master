import matplotlib.pyplot as plt
import numpy as np

# Create synthetic data for a contour plot
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Create the contour plot
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X, Y, Z, 20, cmap="viridis")

# Check the contour object
print(contour)

# Save the plot as a PDF and PNG
plt.savefig("test_plot.pdf", dpi=300, bbox_inches="tight")
plt.savefig("test_plot.png", dpi=300, bbox_inches="tight")

# Close the figure to release memory
plt.close(fig)
