############################################
# Name: Discovering Patterns
# Author: Adrian G. Zucco
# Date: 2024-02-25
# Email: adrigabzu@sund.ku.dk
############################################

# %%
import pandas as pd
import numpy as np
from skimpy import skim
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import shap

# %% ########################## READ THE DATA ############################

# Original source https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
url = "https://raw.githubusercontent.com/AshNumpy/Sleep-Health-ML-Project/main/Datasets/cleaned-dataset.csv"
data = pd.read_csv(url)

# %% Display the first 5 rows of the dataframe
data.head()

# %% Describe the dataframe
skim(data)

# %% Show unique values in text columns
for col in data.select_dtypes(include="object").columns:
    print(f"Column: {col}")
    print(data[col].unique())
    print("")


# %% ########################## Data encoding ############################
# Encode Occupation, Gender and Sleep disorder one hot encoding
data_encoded = pd.get_dummies(
    data.drop("Person ID", axis=1), columns=["Occupation", "Gender", "Sleep Disorder"]
)

# Encode BMI Category as ordinal encoding
bmi_categories = {"Normal Weight": 1, "Normal": 1, "Overweight": 2, "Obese": 3}
data_encoded.replace({"BMI Category": bmi_categories}, inplace=True)

# %% Summary statistics after encoding the categorical variables
skim(data_encoded)


# %% ######################### Visualization ############################
# Transform only boolean columns to integer
bool_cols = [col for col in data_encoded if data_encoded[col].dtype == bool]
data_encoded[bool_cols] = data_encoded[bool_cols].astype(int)

# Heatmap anb cluster individuals normalized by the features
sns.clustermap(data_encoded, annot=False, standard_scale=1, figsize=(10, 10))

# %% Generate Spearman (non-linear) correlation plot
correlation_matrix = data_encoded.corr(method="spearman")
sns.clustermap(correlation_matrix, annot=False, cmap="coolwarm")


# %% ######################## Clustering ###############################

# Cluster individuals with HDBSCAN
# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_encoded)

# Reduce the dimensionality
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Cluster the data with HDBSCAN
# clusterer = HDBSCAN(min_cluster_size=10)

# Cluster the data with KMeans
clusterer = KMeans(n_clusters=3)
clusterer.fit(data_scaled)

# Summary of clustering
print(f"Number of clusters: {len(np.unique(clusterer.labels_))}")
print(f"Number of unclassified individuals: {np.sum(clusterer.labels_ == -1)}")

# %%
# Plot the clusters
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusterer.labels_, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("HDBSCAN Clustering")
plt.show()

# %% ####################### Supervised learning #####################

# Fit random forest to predict Quality of Sleep
# Split the data
X = data_encoded.drop("Quality of Sleep", axis=1)
y = data_encoded["Quality of Sleep"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2024
)

# Fit the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# %% ########################## MODEL EXPLAINABILITY ###################

# Create the explainer with SHAP
explainer = shap.TreeExplainer(model)

# Calculate the shap values
shap_values = explainer.shap_values(X)

# Plot the shap values
shap.summary_plot(shap_values, X)

# %%
# Interaction plot between stress and age
shap.dependence_plot("Stress Level", shap_values, X, interaction_index="Age")
