import numpy as np

# Generate some sample data
data = np.array([10, 15, 20, 25, 30])

# Calculate the mean of the data
mean_value = np.mean(data)

# Mean centering by subtracting the mean from each data point
mean_centered_data = data - mean_value

# Print the original data and mean-centered data
print("Original Data:", data)
print("Mean-Centered Data:", mean_centered_data)
