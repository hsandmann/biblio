import numpy as np
import pandas as pd

np.random.seed(42)

n_per_class = 500

# Class means and covariances
mean0 = [0, 0, 0]
cov0  = [[1.5, 0.2, 0.1], [0.2, 1.8, 0.3], [0.1, 0.3, 1.2]]

mean1 = [5, 0, 0]
cov1  = [[1.4, 0.15, 0.05], [0.15, 1.6, 0.25], [0.05, 0.25, 1.1]]

mean2 = [2.5, 5, 1.5]
cov2  = [[1.3, 0.4, 0.2], [0.4, 2.0, 0.5], [0.2, 0.5, 1.4]]

data0 = np.random.multivariate_normal(mean0, cov0, n_per_class)
data1 = np.random.multivariate_normal(mean1, cov1, n_per_class)
data2 = np.random.multivariate_normal(mean2, cov2, n_per_class)

X = np.vstack([data0, data1, data2])
labels = np.repeat([0, 1, 2], n_per_class)

df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3'])
df['Class'] = labels

# Save to CSV
df.to_csv('dataset_1500_3d_3classes.csv', index=False)

print("File saved:", df.shape)  # Should show (1500, 4)