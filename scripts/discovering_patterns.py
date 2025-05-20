############################################
# Name: Explore patterns in a public health dataset
# Author: Adrian G. Zucco
# Date: 2025-05-20
# Email: adrigabzu@sund.ku.dk
############################################

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import umap
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skimpy import skim

# %% ########################## READ THE DATA ############################

# Read the synthetic datasets
df_ind = pd.read_csv("../data/individuals.csv")
df_loc = pd.read_csv("../data/locations.csv")
df_fam = pd.read_csv("../data/families.csv")

# Merge location features onto individuals
df = df_ind.merge(df_loc, on="Location", how="left")

# Merge family income onto individuals
df = df.merge(
    df_fam[["Person ID", "Family Income", "Family ID"]],
    on="Person ID",
    how="left",
)

# Look at the merged frame
print(f"Shape of the merged DataFrame: {df.shape}")
print(df.head())

# %% ########################## OVERVIEW ############################
# Quick skim of the merged data
print("\nSummary statistics of the merged DataFrame:")
skim(df)

# Show unique values in categorical text columns
print("\nUnique values in categorical columns:")
for col in ["Occupation", "Sex", "Location"]:
    if col in df.columns:
        print(f"Column: {col} → {df[col].unique()}\n")
    else:
        print(f"Column: {col} not found in DataFrame.\n")


# %% ########################## EXPLORATORY PAIRPLOTS ####################
# Pairplot to explore relationships between variables in original datasets

# Pairplot for individuals (df_ind)
print("\nGenerating Pairplot for Individuals Features...")
sns.pairplot(df_ind, corner=True, plot_kws={"alpha": 0.5, "s": 15})
plt.suptitle("Pairplot of Individuals Features", y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent suptitle overlap
plt.show()

# Pairplot for locations (df_loc)
print("\nGenerating Pairplot for Location Features...")
sns.pairplot(df_loc, corner=True, plot_kws={"alpha": 0.5, "s": 15})
plt.suptitle("Pairplot of Location Features", y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout
plt.show()

# %% ########################## ENCODING ############################

# Drop Person ID, Family ID, and Location (original categorical) before modeling
# Location features are already merged if needed.
to_drop = ["Person ID", "Family ID", "Location"]
data_noids = df.drop(columns=to_drop, errors="ignore") # errors='ignore' if some columns might not exist

# One‐hot encode Occupation and Sex
data_encoded = pd.get_dummies(
    data_noids, columns=["Occupation", "Sex"], drop_first=False
)

# All features should now be numeric
print("\nSummary statistics of the encoded DataFrame:")
skim(data_encoded)
print(data_encoded.head())

# %% ######################### VISUALIZATION OF ENCODED DATA ############

# Clustered heatmap (standardized)
# Convert boolean columns (from get_dummies) to int for compatibility with clustermap
data_encoded_heatmap = data_encoded.copy()
for col in data_encoded_heatmap.select_dtypes(include=["bool"]).columns:
    data_encoded_heatmap[col] = data_encoded_heatmap[col].astype(int)

print("\nGenerating Feature Clustermap...")
sns.clustermap(
    data_encoded_heatmap, standard_scale=1, cmap="vlag", figsize=(12, 12)
) # Increased figsize
plt.suptitle("Feature Clustermap (Standardized)", y=1.02)
plt.show()

# %%
# Spearman correlation - Non-linear correlation (ranked)
print("\nGenerating Spearman Correlation Clustermap...")
corr_spearman = data_encoded.corr(method="spearman")
sns.clustermap(corr_spearman, cmap="coolwarm", annot=False, figsize=(10, 10)) # annot=False for cleaner look with many features
plt.suptitle("Spearman Correlation Matrix", y=1.02)
plt.show()

# %% ########################## CLUSTERING ##########################

# Standardize data before PCA and UMAP
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_encoded)

# --- PCA to 2D ---
print("\nPerforming PCA...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by PC1: {explained_variance[0]:.3f}")
print(f"Explained variance by PC2: {explained_variance[1]:.3f}")
print(f"Total explained variance by 2 PCs: {sum(explained_variance):.3f}")


# --- PCA with loadings plot ---
print("\nGenerating PCA plot with loadings...")
plt.figure(figsize=(10, 8))
# Scatter plot of the first two principal components
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=15, alpha=0.6, c="blue", label="Individuals")
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.title("PCA: Individuals in 2D PC Space with Feature Loadings")

# Plot loadings (arrows for each feature)
loadings = pca.components_.T
feature_names = data_encoded.columns
# Adjust scaler_factor based on the spread of your data in PC space
# scaler_factor = np.sqrt(np.max(X_pca)) * 0.5 # Heuristic for arrow length
scaler_factor = 6
for i, feature in enumerate(feature_names):
    plt.arrow(
        0,
        0,
        loadings[i, 0] * scaler_factor,
        loadings[i, 1] * scaler_factor,
        color="r",
        alpha=0.7,
        head_width=0.05 * scaler_factor * 0.1, # Adjust head_width relative to arrow length
    )
    plt.text(
        loadings[i, 0] * scaler_factor * 1.15, # Offset text slightly
        loadings[i, 1] * scaler_factor * 1.15,
        feature,
        color="g",
        fontsize=8,
        ha="center",
        va="center"
    )
plt.grid(True, linestyle="--", alpha=0.7)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# %%  --- UMAP + HDBSCAN clustering ---
print("\nPerforming UMAP and HDBSCAN clustering...")
umap_model = umap.UMAP(
    n_neighbors=15, min_dist=0.1, n_components=2, random_state=42
)
X_umap = umap_model.fit_transform(X_scaled)

# HDBSCAN clustering on UMAP embedding
clusterer_umap = HDBSCAN(min_cluster_size=15) # min_cluster_size can be tuned
labels_umap = clusterer_umap.fit_predict(X_umap)

n_clusters_found = len(np.unique(labels_umap)) - (1 if -1 in np.unique(labels_umap) else 0)
n_noise_points = np.sum(labels_umap == -1)
print(f"Clusters found by HDBSCAN: {n_clusters_found}")
print(f"Unclassified (noise) points: {n_noise_points}")

plt.figure(figsize=(8, 7))
unique_labels = np.unique(labels_umap)
palette = sns.color_palette("deep", n_colors=len(unique_labels)) # Use "deep" or other suitable palette
sns.scatterplot(
    x=X_umap[:, 0],
    y=X_umap[:, 1],
    hue=labels_umap,
    palette=palette,
    s=20,
    alpha=0.7,
    legend="full",
)
plt.title("HDBSCAN Clusters on UMAP Embedding")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# %% --- Compute summary statistics for the top 5 biggest clusters ---
print("\nComputing summary statistics for top clusters...")
# Add cluster labels to the original (pre-encoding, pre-scaling) DataFrame for interpretability
df_clusters = df.copy() # Use original df for richer features if needed
# Ensure 'Person ID' exists if data_encoded was derived from df after dropping IDs
# If df_clusters is based on 'df', it has 'Person ID'.
# If data_encoded was used to get X_scaled, their indices should align.
# Assuming indices of df and data_encoded (and thus X_scaled/X_umap) are aligned:
if len(df_clusters) == len(labels_umap):
    df_clusters["cluster"] = labels_umap
else:
    print("Warning: Length mismatch between df_clusters and labels_umap. Skipping cluster summaries on df.")
    # Fallback to using data_encoded if lengths mismatch, though less interpretable
    # This case should be investigated if it occurs.
    if len(data_encoded) == len(labels_umap):
        print("Attempting cluster summaries on data_encoded instead.")
        df_clusters_encoded = data_encoded.copy()
        df_clusters_encoded["cluster"] = labels_umap
        df_clusters = df_clusters_encoded # Redefine for the loop below
    else:
        print("Critical: Cannot align cluster labels with any DataFrame. Skipping summaries.")
        df_clusters = None


if df_clusters is not None:
    # Exclude noise points (-1)
    valid_clusters_df = df_clusters[df_clusters["cluster"] != -1]

    if not valid_clusters_df.empty:
        # Find top 5 clusters by size
        top_clusters = (
            valid_clusters_df["cluster"].value_counts().nlargest(5).index.tolist()
        )
        print(f"Top 5 clusters by size: {top_clusters}")

        # Summary statistics for each top cluster using skimpy
        for c in top_clusters:
            cluster_data = valid_clusters_df[valid_clusters_df["cluster"] == c]
            print(
                f"\n--- Cluster {c} (n={len(cluster_data)}) ---"
            )
            skim(cluster_data.drop(columns=["cluster"])) # Drop cluster column before skimming
    else:
        print("No valid clusters found (excluding noise) for summary statistics.")
else:
    print("Skipping cluster summary statistics due to previous errors.")


# %% #################### SUPERVISED LEARNING #######################

# We will predict 'Sleep problems' (assuming it's a binary column in data_encoded)
target_column = "Sleep problems"

if target_column not in data_encoded.columns:
    raise ValueError(f"Target column '{target_column}' not found in data_encoded.")

X = data_encoded.drop(columns=[target_column])
y = data_encoded[target_column].astype(int) # Ensure target is integer

print(f"\nTarget variable distribution:\n{y.value_counts(normalize=True)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2025, stratify=y # Stratify for imbalanced classes
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

clf = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight="balanced" # Handles imbalanced classes
)
print("\nTraining RandomForestClassifier...")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1] # Probabilities for the positive class

print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))

roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)
print(f"ROC AUC: {roc_auc:.3f}")
print(f"PR AUC (Average Precision): {pr_auc:.3f}")

# Plot confusion matrix
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
# Print confusion matrix as a nicely formatted table
cm_df = pd.DataFrame(
    cm,
    index=["Actual No Sleep Problems", "Actual Sleep Problems"],
    columns=["Predicted No Sleep Problems", "Predicted Sleep Problems"]
)
print("\nConfusion Matrix (rows: actual, columns: predicted):")
print(cm_df.to_string())

# %% ####################### MODEL EXPLAINABILITY ###################
print("\nGenerating SHAP explanations...")

# SHAP for RandomForestClassifier
explainer = shap.TreeExplainer(clf)
# Calculate SHAP values for the test set for faster computation if X is very large
# Or use a sample of X: shap.sample(X, 100)
# For consistency with original, using full X (can be slow for large datasets)
shap_values_rf = explainer.shap_values(X_test) # Using X_test for explainability of test predictions

# Summary plot for class “1” (has sleep problems)
# shap_values_rf will be a list of two arrays for binary classification:
# one for class 0, one for class 1. We usually explain class 1.
print("\nGenerating SHAP Summary Plot (Bar)...")
plt.figure() # Create a figure context for SHAP plot if needed
shap.summary_plot(shap_values_rf[1], X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Predicting Sleep Problems)")
plt.tight_layout()
plt.show()

print("\nGenerating SHAP Summary Plot (Dot)...")
plt.figure()
shap.summary_plot(shap_values_rf[1], X_test, show=False)
plt.title("SHAP Feature Impact (Predicting Sleep Problems)")
plt.tight_layout()
plt.show()


# Dependence plot for a key feature, e.g., 'Stress Levels' vs 'Age'
# Ensure 'Stress Levels' and 'Age' are in X_test.columns
if "Stress Levels" in X_test.columns and "Age" in X_test.columns:
    print("\nGenerating SHAP Dependence Plot for 'Stress Levels'...")
    plt.figure()
    shap.dependence_plot(
        "Stress Levels",
        shap_values_rf[1],
        X_test,
        interaction_index="Age", # Color by 'Age'
        show=False
    )
    plt.title("SHAP Dependence Plot: Stress Levels (Interaction with Age)")
    plt.tight_layout()
    plt.show()
elif "Stress Levels" in X_test.columns:
    print("\nGenerating SHAP Dependence Plot for 'Stress Levels' (no interaction)...")
    plt.figure()
    shap.dependence_plot(
        "Stress Levels",
        shap_values_rf[1],
        X_test,
        interaction_index=None,
        show=False
    )
    plt.title("SHAP Dependence Plot: Stress Levels")
    plt.tight_layout()
    plt.show()
else:
    print("Column 'Stress Levels' not found for SHAP dependence plot.")

print("\nScript finished.")
