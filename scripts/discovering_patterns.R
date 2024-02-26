## Description: Explore patterns in a public health dataset
##
## Author: Adrian G. Zucco
## Date Created: 2024-02-25
## Email: adrigabzu@sund.ku.dk
## ------------------------------------

# Uncomment and execute the first time if needed
# install.packages("pak", repos = "https://r-lib.github.io/p/pak/devel/")
# pak::pkg_install(c("skimr", "tidyverse", "tidymodels", "corrplot", "ranger", "treeshap", "shapviz"))

# Load required libraries
library(tidyverse)
library(skimr)
library(tidymodels)
library(corrplot)
library(ranger)
library(treeshap)
library(shapviz)

########################## READ THE DATA ############################

# Original source https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
# Read the data from web
url <- "https://raw.githubusercontent.com/AshNumpy/Sleep-Health-ML-Project/main/Datasets/cleaned-dataset.csv"
data <- read_csv(url)

# Preprocess the data
glimpse(data)

# Summary of the data
skim(data)

# Show unique values in text columns
data %>%
  select(where(is.character)) %>%
  map(unique)

########################## Data encoding ############################
# Create a recipe for data pre-processing
rec <- recipe(~., data = data) %>%
  # Drop Person ID column
  step_rm(`Person ID`) %>%
  # One hot encode Occupation, Gender and Sleep Disorder
  step_dummy(Occupation, Gender, `Sleep Disorder`, one_hot = TRUE)

# Prepare the recipe
data_encoded <- prep(rec) %>%
  # Apply the recipe to the data
  bake(new_data = NULL)

# Recode the BMI
data_encoded$`BMI Category` <- as.integer(recode_factor(data_encoded$`BMI Category`,
  "Normal" = 1, "Normal Weight" = 1,
  "Overweight" = 2, "Obese" = 3
))

# Summary after encoding
skim(data_encoded)

######################### Visualization ############################

# HEATMAP of the data
data_encoded %>%
  scale() %>%
  as.matrix() %>%
  heatmap(col = viridis::viridis(100))

# Correlation plot of the variables
data_encoded %>%
  cor(method = "spearman") %>%
  corrplot(
    method = "color", type = "upper", order = "hclust",
    tl.col = "black", tl.srt = 45
  )

######################## Clustering ###############################

# K-means clustering
set.seed(123)
kmeans_model <- kmeans(data_encoded, centers = 3, nstart = 25)

# PCA of the data
pca <- prcomp(data_encoded, scale = TRUE)

# Plot the PCA with ggplot
pca_df <- as.data.frame(pca$x)
pca_df$cluster <- as.factor(kmeans_model$cluster)

pca_df %>%
  ggplot(aes(x = PC1, y = PC2, color = cluster)) +
  geom_point() +
  labs(
    title = "PCA of the data with K-means Clusters",
    x = "PC1", y = "PC2"
  ) +
  theme_minimal()

####################### Supervised learning #####################

# Split the data into training and testing excluding the Quality of Sleep
set.seed(123)
data_split <- initial_split(data_encoded, prop = 0.8, strata = `Quality of Sleep`)
data_train <- training(data_split)
data_test <- testing(data_split)

# Create a model specification
model_spec <- rand_forest(mode = "regression", trees = 100) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# Create a recipe for the model
rec_model <- recipe(`Quality of Sleep` ~ ., data = data_train)

# DEFINE WORKFLOW
wkflow <- workflow() |>
  add_model(model_spec) |>
  add_recipe(rec_model)

# TRAINING MODEL
model_fit <- wkflow |>
  fit(data = data_train)

# ASSESS MODEL ON TEST DATA
preds <- model_fit |>
  augment(new_data = data_test)

# Evaluate the model
preds %>%
  metrics(truth = `Quality of Sleep`, estimate = .pred) %>%
  summary()

# Plot the predictions
preds %>%
  ggplot(aes(x = `Quality of Sleep`, y = .pred)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  labs(
    title = "Predictions vs Truth",
    x = "Truth", y = "Prediction"
  ) +
  theme_minimal()

########################## MODEL EXPLAINABILITY ###################

rfo <- model_fit %>%
  extract_fit_engine()

treeshp <- treeshap(unify(rfo, data_encoded),
  x = data_encoded %>% select(-`Quality of Sleep`),
  interactions = TRUE
)

shp <- shapviz(treeshp, X = data_encoded)

### Global feature importance
sv_importance(shp, kind = "beeswarm")

x <- c("Age", "Daily Steps")
sv_dependence(shp, v = "Stress Level", color_var = x, interactions = TRUE)