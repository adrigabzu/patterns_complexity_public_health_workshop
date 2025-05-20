## =============================================================================
## Description: Explore patterns in a public health dataset
## Author:      Adrian G. Zucco
## Date:        2025-05-19 (Last revised: 2025-05-20)
## Email:       adrigabzu@sund.ku.dk
## =============================================================================

## ---- 0. SETUP: INSTALL AND LOAD PACKAGES ----

# Execute this section only if packages are not already installed.
# Method 1: Using pak (recommended for handling dependencies)
# install.packages("pak")
# pak::pkg_install(c(
#   "skimr", "tidyverse", "tidymodels", "corrplot", "lightgbm", "bonsai",
#   "treeshap", "shapviz", "GGally", "umap", "dbscan", "viridis", "ggrepel"
# ))

# Method 2: Using install.packages (fallback)
# install.packages(c(
#   "skimr", "tidyverse", "tidymodels", "corrplot", "lightgbm","bonsai",
#   "treeshap", "shapviz", "GGally", "umap", "dbscan", "viridis", "ggrepel"
# ))

# Load required libraries
library(tidyverse)    # For data manipulation (dplyr, ggplot2, readr, etc.)
library(skimr)        # For summary statistics
library(tidymodels)   # For modeling workflows (recipes, rsample, parsnip, yardstick)
library(corrplot)     # For visualizing correlation matrices
library(ranger)       # For random forest models (Note: ranger is loaded but not explicitly used in the provided script. If intended for other models, it's fine.)
library(treeshap)     # For SHAP values with tree-based models (Used by shapviz backend)
library(shapviz)      # For visualizing SHAP values
library(GGally)       # For extended ggplot2 pairplots (ggpairs)
library(umap)         # For UMAP dimensionality reduction
library(dbscan)       # For HDBSCAN clustering
library(viridis)      # For viridis color palettes
library(ggrepel)      # For repelling text labels in ggplot2
library(lightgbm)     # For LightGBM models (bonsai engine uses this; good to have explicitly loaded for direct model object interaction if needed)
library(bonsai)       # For Bonsai (LightGBM, XGBoost, Spark) tidymodels integration


## ---- 1. DATA LOADING AND MERGING ----

# Read synthetic CSVs
# Ensure these paths are correct for your local environment
df_ind <- read_csv("../data/individuals.csv")
df_loc <- read_csv("../data/locations.csv")
df_fam <- read_csv("../data/families.csv")

# Merge datasets
df <- df_ind %>%
  left_join(df_loc, by = "Location") %>%
  left_join(df_fam, by = "Person ID") # Assumes "Person ID" links individuals to family income

# Initial data inspection
glimpse(df)
skim(df)


## ---- 2. EXPLORATORY DATA ANALYSIS (EDA) ----

# Pairplots for individuals and locations features

# Pairplot for individuals features
GGally::ggpairs(
  df_ind,
  progress = FALSE, # Good for non-interactive runs
  title = "Pairplot of Individuals Features"
)

# Pairplot for locations features
GGally::ggpairs(
  df_loc,
  progress = FALSE,
  title = "Pairplot of Location Features"
)

# Show unique values in character columns of the merged dataframe
df %>%
  select(where(is.character)) %>%
  map(unique)


## ---- 3. DATA PREPROCESSING: ENCODING ----

# Define the recipe for preprocessing
# We'll predict 'Sleep problems' (0/1)
# Remove identifiers and the original 'Location' text column (already joined features)
# One-hot encode categorical predictors
rec <- recipe(`Sleep problems` ~ ., data = df) %>%
  step_mutate(`Sleep problems` = factor(`Sleep problems`)) %>% # Ensure target is factor for classification
  step_rm(`Person ID`, `Family ID`, Location) %>% # Location text column, not features
  step_dummy(all_nominal_predictors(), one_hot = TRUE) # one_hot = TRUE is a good default

# Prepare the recipe (estimate parameters from data) and bake (apply to data)
data_encoded <- rec %>%
  prep() %>%
  bake(new_data = NULL) |>
  select(-any_of(c("Sex_M", "Sex_male"))) # More robustly remove one level of Sex if 'Sex_F'/'Sex_female' exists
# Or, if 'Sex' had more than two levels, this might need adjustment
# For a binary 'Sex' (e.g. M/F), step_dummy creates Sex_F, Sex_M.
# Removing one (e.g. Sex_M) is standard to avoid multicollinearity.
# Ensure the column name is correct based on your data.

# Inspect encoded data
skim(data_encoded)
glimpse(data_encoded)

## ---- 4. VISUALIZATION OF ENCODED DATA ----

# Heatmap of scaled features (excluding the target variable)
data_encoded %>%
  select(-`Sleep problems`) %>%
  scale() %>% # Center and scale features
  as.matrix() %>%
  heatmap(
    Colv = NA, Rowv = NA, # Turn off dendrograms for a cleaner look
    col = viridis::viridis(100),
    main = "Scaled Feature Heatmap",
    margins = c(5, 10) # Adjust margins (bottom, left)
  )

# Spearman correlation plot (excluding the target variable)
# Spearman is good for non-linear relationships and ordinal data.
M <- data_encoded %>%
  select(-`Sleep problems`) %>%
  cor(method = "spearman", use = "pairwise.complete.obs") # Handles missing data by pairwise deletion

inverted_corr_palette <- rev(corrplot::COL2('RdBu', 200))

corrplot(
  M,
  method = "color",
  type = "upper",    # Show upper triangle
  order = "hclust",  # Order by hierarchical clustering
  col = inverted_corr_palette,
  tl.col = "black",
  tl.srt = 45,       # Rotate text labels
  title = "Spearman Correlation of Features",
  mar = c(0, 0, 1, 0) # Adjust margins (bottom, left, top, right)
)

## ---- 5. UNSUPERVISED LEARNING: CLUSTERING ----

set.seed(2025) # For reproducibility
features_for_clustering <- data_encoded %>% select(-`Sleep problems`)

### ---- 5.1 K-means Clustering ----
# Note: The number of centers (k=3) is fixed here. In practice,
# you might use methods like elbow plot or silhouette score to determine optimal k.
km_model <- kmeans(
  scale(features_for_clustering), # Scaling is important for k-means
  centers = 3,
  nstart = 25 # Run with multiple random starts
)

# PCA for visualizing K-means clusters
pca_res_km <- prcomp(features_for_clustering, scale. = TRUE) # Scaling is crucial for PCA

cat("\n--- PCA Loadings (Rotation Matrix) ---\n")
print(round(pca_res_km$rotation, 3))

pca_variance <- pca_res_km$sdev^2
prop_variance_explained <- pca_variance / sum(pca_variance)
cumulative_variance_explained <- cumsum(prop_variance_explained)

explained_variance_df <- tibble(
  PC = 1:length(prop_variance_explained),
  Proportion_of_Variance = prop_variance_explained,
  Cumulative_Proportion = cumulative_variance_explained
)

cat("\n--- Proportion of Variance Explained by Each PC (Top 10) ---\n")
print(explained_variance_df %>% select(PC, Proportion_of_Variance) %>% head(10))
cat("\n--- Cumulative Proportion of Variance Explained (Top 10) ---\n")
print(explained_variance_df %>% select(PC, Cumulative_Proportion) %>% head(10))

# Prepare data for combined plot
pca_df_km <- as_tibble(pca_res_km$x[, 1:2]) %>% # PC scores
  mutate(cluster = factor(km_model$cluster))

loadings_data <- as_tibble(pca_res_km$rotation[, 1:2], rownames = "Variable") %>% # Loadings
  rename(PC1_loading = PC1, PC2_loading = PC2)

pc1_var_explained <- prop_variance_explained[1] * 100
pc2_var_explained <- prop_variance_explained[2] * 100

# Calculate scaling factor for loadings to overlay them on the scores plot
# This factor scales the loading vectors to be visually appropriate on the score plot.
# We aim for the longest loading arrow to be a noticeable fraction of the plot's extent.
plot_range_x <- diff(range(pca_df_km$PC1, na.rm = TRUE))
plot_range_y <- diff(range(pca_df_km$PC2, na.rm = TRUE))
# Handle cases where range might be zero (e.g., if only one data point or all values are same for a PC)
if (is.na(plot_range_x) || plot_range_x == 0) plot_range_x <- 1
if (is.na(plot_range_y) || plot_range_y == 0) plot_range_y <- 1

plot_diagonal <- sqrt(plot_range_x^2 + plot_range_y^2)

loading_lengths <- sqrt(loadings_data$PC1_loading^2 + loadings_data$PC2_loading^2)
max_raw_loading_length <- max(loading_lengths, na.rm = TRUE)

loading_scale_factor <- 1 # Default if max_raw_loading_length is zero or NA
if (!is.na(max_raw_loading_length) && max_raw_loading_length > 1e-6) {
  # Target length for the longest arrow on the plot (e.g., 30-40% of the plot diagonal)
  target_arrow_length <- plot_diagonal * 0.35
  loading_scale_factor <- target_arrow_length / max_raw_loading_length
}

# Combined K-means Cluster and PCA Loadings Plot
plot_kmeans_pca_with_loadings <- ggplot(pca_df_km, aes(x = PC1, y = PC2)) +
  geom_point(aes(color = cluster), alpha = 0.6, size = 2) + # K-means clusters
  # Add Loadings Arrows
  geom_segment(
    data = loadings_data,
    aes(
      x = 0, y = 0,
      xend = PC1_loading * loading_scale_factor,
      yend = PC2_loading * loading_scale_factor
    ),
    arrow = arrow(length = unit(0.2, "cm")),
    color = "gray20", # Dark color for loading arrows
    linewidth = 0.6,
    inherit.aes = FALSE
  ) +
  # Add Loadings Labels
  ggrepel::geom_text_repel(
    data = loadings_data,
    aes(
      x = PC1_loading * loading_scale_factor * 1.1, # Offset labels from arrow tips
      y = PC2_loading * loading_scale_factor * 1.1,
      label = Variable
    ),
    color = "gray10", # Dark color for loading labels
    size = 3,
    inherit.aes = FALSE,
    max.overlaps = Inf, # Ensure all labels are shown
    segment.color = "gray50",
    segment.size = 0.3,
    box.padding = unit(0.35, "lines"),
    point.padding = unit(0.5, "lines")
  ) +
  labs(
    title = "K-means Clusters and PCA Loadings (PC1 vs PC2)",
    x = paste0("PC1 (", sprintf("%.1f%%", pc1_var_explained), " Variance)"),
    y = paste0("PC2 (", sprintf("%.1f%%", pc2_var_explained), " Variance)"),
    color = "K-means Cluster"
  ) +
  theme_minimal(base_size = 12) +
  coord_fixed() + # Ensures aspect ratio is 1, crucial for interpreting PCA
  scale_color_viridis_d(option = "C") # Example: Using Viridis for cluster colors

print(plot_kmeans_pca_with_loadings)

# The separate loadings_plot is now removed as its information is integrated above.

### ---- 5.2 HDBSCAN Clustering ----

# UMAP for dimensionality reduction before HDBSCAN
# Scaling features before UMAP is generally a good practice
umap_res <- umap::umap(
  scale(features_for_clustering),
  n_neighbors = 15,     # UMAP parameters, may require tuning
  min_dist = 0.1,
  random_state = 2025   # For reproducibility
)
umap_df <- as_tibble(umap_res$layout, .name_repair = "minimal") %>%
  setNames(c("UMAP1", "UMAP2"))

# HDBSCAN clustering on UMAP embeddings
hdb_clusters <- dbscan::hdbscan(umap_df, minPts = 10) # minPts is a key HDBSCAN parameter
umap_df$cluster <- factor(hdb_clusters$cluster) # 0 represents noise points

cat("\n--- HDBSCAN Clustering Results ---\n")
cat("HDBSCAN - Clusters found (excluding noise):", length(unique(hdb_clusters$cluster[hdb_clusters$cluster != 0])), "\n")
cat("HDBSCAN - Unclassified (noise) points (cluster 0):", sum(hdb_clusters$cluster == 0), "\n")

ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = cluster)) +
  geom_point(size = 1.5, alpha = 0.7) +
  labs(
    title = "HDBSCAN Clusters on UMAP Embedding",
    x = "UMAP Dimension 1",
    y = "UMAP Dimension 2",
    color = "HDBSCAN Cluster"
  ) +
  theme_minimal() +
  scale_color_viridis_d(option = "D") # Good color scale for discrete clusters

### ---- 5.3 Summary Statistics for Top HDBSCAN Clusters ----
# Add cluster assignments to the original (unencoded) dataframe for interpretability
df_with_hdbscan_clusters <- df %>%
  mutate(cluster_hdbscan = factor(hdb_clusters$cluster)) # Assuming df has same number of rows as features_for_clustering

valid_hdbscan_clusters <- df_with_hdbscan_clusters %>% filter(cluster_hdbscan != 0) # Exclude noise

if (nrow(valid_hdbscan_clusters) > 0) {
  top_hdbscan_clusters_ids <- valid_hdbscan_clusters %>%
    count(cluster_hdbscan, sort = TRUE) %>%
    slice_head(n = 5) %>% # Look at top 5 largest clusters
    pull(cluster_hdbscan)
  
  cat("\n--- Summary Statistics for Top 5 HDBSCAN Clusters (Original Features) ---\n")
  cat("Top 5 clusters by size (excluding noise):", paste(top_hdbscan_clusters_ids, collapse = ", "), "\n")
  
  for (c_val in top_hdbscan_clusters_ids) {
    cat("\n--- HDBSCAN Cluster", as.character(c_val),
        "(n =", sum(valid_hdbscan_clusters$cluster_hdbscan == c_val), ") ---\n")
    cluster_data_to_skim <- valid_hdbscan_clusters %>%
      filter(cluster_hdbscan == c_val) %>%
      select(-cluster_hdbscan) # Don't skim the cluster ID itself
    print(skimr::skim(cluster_data_to_skim))
  }
} else {
  cat("\nNo valid (non-noise) HDBSCAN clusters found to summarize.\n")
}

## ---- 6. SUPERVISED LEARNING: CLASSIFICATION ----

set.seed(2025) # For reproducibility of data splitting and model training
split <- initial_split(data_encoded, prop = 0.80, strata = `Sleep problems`) # Stratification is good
train_data <- training(split)
test_data  <- testing(split)

# Define LightGBM model specification using bonsai
lgbm_spec <- boost_tree(
  mode = "classification",
  trees = 100 # Number of boosting rounds (nrounds); often tuned
  # mtry, min_n, tree_depth can also be tuned.
  # learn_rate is a key parameter, default is 0.1 for lightgbm via parsnip/bonsai.
) %>%
  set_engine("lightgbm", seed = 2025) # Specify the lightgbm engine and set seed for its RNG

# Create workflow
wf <- workflow() %>%
  add_model(lgbm_spec) %>%
  add_formula(`Sleep problems` ~ .) # Use all other columns as predictors

# Fit the LightGBM model
cat("\n--- Training LightGBM model ---\n")
lgbm_fit <- wf %>% fit(data = train_data)
cat("LightGBM model training complete.\n")

# Make predictions
# Get class probabilities and predicted class
preds_df <- lgbm_fit %>%
  predict(test_data, type = "prob") %>%
  bind_cols(predict(lgbm_fit, test_data)) %>% # .pred_class
  bind_cols(test_data %>% select(`Sleep problems`)) # truth

# Define metrics
classification_metrics <- metric_set(
  accuracy, roc_auc, sensitivity, specificity, pr_auc # recall is sensitivity
)

# Determine the positive class label (assuming binary classification 0/1 or similar)
# Yardstick's event_level = "second" assumes the second factor level is the event/positive class.
# Ensure this aligns with your data's factor levels for 'Sleep problems'.
# E.g., if levels are factor(c("0", "1")), "1" is the second level.
positive_class_level <- levels(train_data$`Sleep problems`)[2]
prob_column_for_positive_class <- paste0(".pred_", positive_class_level)

# Evaluate the model
eval_results <- preds_df %>%
  classification_metrics(
    truth = `Sleep problems`,
    estimate = .pred_class,
    !!sym(prob_column_for_positive_class), # Probabilities for the positive class
    event_level = "second" # Specify the event level
  )
print(eval_results)

# Confusion matrix
conf_mat_res <- conf_mat(
  data = preds_df, truth = `Sleep problems`, estimate = .pred_class
)
print(conf_mat_res)

autoplot(conf_mat_res, type = "heatmap") +
  labs(title = "Confusion Matrix for Sleep Problems Prediction (LightGBM)") +
  theme_minimal()


## ---- 7. MODEL EXPLAINABILITY: SHAP VALUES ----

# Extract the fitted LightGBM model object (lgb.Booster)
actual_lgbm_model <- lgbm_fit %>%
  extract_fit_parsnip() %>%
  .$fit

# Prepare the predictor data from the TEST set for which SHAP values will be calculated.
X_pred_shap <- test_data %>%
  select(-`Sleep problems`) %>%
  as.data.frame() # Ensure it's a data.frame

# Prepare the predictor data from the TRAINING set to use as a reference/background
# for SHAP calculations. This is important for tree-based ensemble methods.
# It should contain only the predictor columns.
X_background_shap <- train_data %>%
  select(-`Sleep problems`) %>% # Ensure only predictors are included
  as.data.frame()
# If X_background_shap is very large, consider sampling for performance:
# X_background_shap <- train_data %>% select(-`Sleep problems`) %>% sample_n(min(nrow(.), 1000)) %>% as.data.frame()


cat("\n--- Calculating SHAP values for LightGBM (this may take a moment) ---\n")

# Create a shapviz object.
# X_pred: The data for which to compute SHAP values (predictions to be explained).
# X: The background dataset used to calculate expected values. Using (a sample of)
#    the training data is a common and recommended practice.
sv <- shapviz::shapviz(
  actual_lgbm_model,
  X_pred = data.matrix(X_pred_shap), # treeshap often prefers matrix for X_pred
  X = X_pred_shap              # Use prepared training predictors as background
)
cat("SHAP value calculation complete.\n")

# --- Overall Feature Importance Plots ---

# 1. SHAP Bar Plot: Mean absolute SHAP values (global importance)
plot_shap_importance_bar <- sv_importance(sv, kind = "bar", max_display = 15) +
  ggtitle("Overall Feature Importance (Mean |SHAP|) - LightGBM") +
  theme_minimal(base_size = 10)
print(plot_shap_importance_bar)

# 2. SHAP Summary Plot (Beeswarm): Shows SHAP values for each instance and feature
plot_shap_summary <- sv_importance(sv, kind = "beeswarm", max_display = 15) +
  ggtitle("SHAP Summary Plot (Beeswarm) - LightGBM") +
  theme_minimal(base_size = 10)
print(plot_shap_summary)

# Glimpse the data used for SHAP predictions (to check column names for dependence plots)
cat("\n--- Glimpse of data used for SHAP predictions (X_pred_shap) ---\n")
glimpse(X_pred_shap)

# --- SHAP Dependence Plots ---
# These plots show how a feature's value relates to its SHAP value,
# optionally colored by an interaction feature.
# Ensure the variable names 'Physical Activity Level', 'Stress Levels', 'Age'
# exist as column names in X_pred_shap.

required_vars_for_dependence <- c("Physical Activity Level", "Stress Levels", "Age")
available_vars_in_pred_data <- colnames(X_pred_shap)

plot_shap_dependence <- function(shap_viz_object, feature_var, color_var = NULL, data_for_check) {
  if (!(feature_var %in% colnames(data_for_check))) {
    cat(paste0("\nSkipping SHAP dependence plot for '", feature_var, "' as it's not in X_pred_shap.\n"))
    return(invisible(NULL))
  }
  if (!is.null(color_var) && !(color_var %in% colnames(data_for_check))) {
    cat(paste0("\nWarning: color_var '", color_var, "' for SHAP dependence plot of '", feature_var, "' not found. Plotting without color_var.\n"))
    color_var <- NULL # Plot without color if color_var is missing
  }
  
  plot_title <- paste0("SHAP Dependence: ", feature_var)
  if (!is.null(color_var)) {
    plot_title <- paste0(plot_title, " (Color: ", color_var, ")")
  }
  
  print(
    sv_dependence(shap_viz_object, v = feature_var, color_var = color_var) +
      ggtitle(plot_title) +
      theme_minimal(base_size = 10)
  )
}

plot_shap_dependence(sv, "Physical Activity Level", "Age", X_pred_shap)
plot_shap_dependence(sv, "Stress Levels", "Age", X_pred_shap)
# Add more dependence plots as needed:
# plot_shap_dependence(sv, "SomeOtherFeature", "AnotherFeature", X_pred_shap)


cat("\n--- End of Script ---\n")


