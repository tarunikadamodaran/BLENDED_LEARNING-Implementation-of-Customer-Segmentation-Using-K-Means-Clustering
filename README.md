# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Load the Data**  
   Import the dataset to start the clustering analysis process.

2. **Explore the Data**  
   Analyze the dataset to understand distributions, patterns, and key characteristics.

3. **Select Relevant Features**  
   Identify the most informative features to improve clustering accuracy and relevance.

4. **Preprocess the Data**  
   Clean and scale the data to prepare it for clustering.

5. **Determine Optimal Number of Clusters**  
   Use techniques like the elbow method to find the ideal number of clusters.

6. **Train the Model with K-Means Clustering**  
   Apply the K-Means algorithm to group data points into clusters based on similarity.

7. **Analyze and Visualize Clusters**  
   Examine and visualize the resulting clusters to interpret patterns and relationships.
   
## Program:
```
/*
Program to implement customer segmentation using K-Means clustering on the Mall Customers dataset.
Developed by: TARUNIKA D
RegisterNumber:  212223040227
*/


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Load the dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/CustomerData.csv"
data = pd.read_csv(url)

# Step 2: Explore the data
# Display the first few rows of the dataset along with column names for inspection
print(data.head())
print(data.columns)

# Step 3: Select the relevant features for clustering
# Here, we are using 'Age', 'Annual Income (k$)', and 'Spending Score (1-100)' for the clustering process
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

# Step 4: Data Preprocessing
# Standardize the features to enhance the performance of the K-Means algorithm
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Identify the optimal number of clusters using the Elbow Method
inertia_values = []  # List to store the inertia values (within-cluster sum of squares)
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow curve to visualize the optimal number of clusters
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia_values, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Step 6: Train the K-Means model with the chosen number of clusters
# Based on the elbow plot, we select the optimal number of clusters, which is 4 in this case
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(X_scaled)

# Step 7: Analyze and visualize the clusters
# Add the cluster labels to the original dataset
data['Cluster'] = kmeans.labels_

# Calculate and display the silhouette score to evaluate the quality of clustering
sil_score = silhouette_score(X_scaled, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')

# Visualize the clusters based on 'Annual Income (k$)' and 'Spending Score (1-100)'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis', s=100, alpha=0.7)
plt.title('Customer Segmentation based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()

```

## Output:
<img width="863" alt="ML EXP 9 IMAGE" src="https://github.com/user-attachments/assets/0ceb42d3-7312-4027-ba45-cede7ff212c6">



## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
