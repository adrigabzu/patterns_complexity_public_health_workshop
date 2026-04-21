# Student Guide: Discovering Patterns in Public Health Data

Go through the code and the questions below to discover the hidden, complex patterns in a synthetic public health dataset. This guide is designed to lead you through the exploratory data analysis, unsupervised learning, supervised learning, and explainability sections of the workshop. Reflect on the questions, discuss with your peers and by the end the rules used to generate the data wll be revealed.

---

## Part 1: Exploratory Data Analysis & Complex Relationships (Sections 2–5)


**1. Baseline Population Health**
Look at the output of the `skim(df)` function in **Section 2**.
* What is the overall baseline prevalence (mean) of `Depressive symptoms` and `Sleep problems` in this population?
* *Hint:* Find the mean value for these binary (0/1) columns.

**2. Non-Linear Health Trajectories**
Health behaviors and biology change dynamically across our lifespan. Examine the pair plots in **Section 3**.
* Look at the scatter plot comparing `Age` (x-axis) and `Stress Levels` (y-axis). Does stress increase linearly with age forever? At approximately what age does stress appear to peak before declining?
* Now look at `Age` vs. `Physical Activity Level`. At what age does physical activity seem to peak?

**3. Socio-Economic Segregation**
Look at the Spearman Correlation Matrix in **Section 5** and the categorical variable outputs in **Section 3**.
* How does `Family Size` correlate with the `Average Income` of the location they reside in? What does this imply about the spatial pressures on larger families?
* *Code challenge:* Using the pair plots or by adding a simple `df.groupby('Occupation')['Average wealth'].mean()` cell, can you identify which occupation exclusively lives in high-wealth locations, and which lives in low-wealth locations? What about Nurses?

**4. The Phenotype of Depression**
Depression doesn't manifest identically in everyone. Look at the pair plots again for individuals with `Depressive symptoms`.
* Do people with depressive symptoms simply sleep less? What is the unique shape of the `Sleep duration` distribution for this specific group compared to the rest of the population?

---

## Part 2: Unsupervised Learning - Finding Hidden Subpopulations (Section 6)
*Clustering allows us to see health profiles—rather than single variables—group together in the wild.*

**5. Principal Drivers of Variation**
Look at the K-means PCA plot with the red loading arrows.
* Which features (the longest arrows) seem to be the strongest drivers pulling individuals apart in this 2D space? Do demographic factors (Age) or lifestyle factors (Activity, BMI, Stress) dominate?

**6. Profiling "At-Risk" Clusters**
HDBSCAN finds natural, irregularly shaped groupings in the data. Look at the text output for "**Summary statistics by cluster**".
* Identify a cluster that represents a "High-Risk" health profile (e.g., higher average BMI, highest Stress Levels, and lowest Physical Activity). 
* Compare this cluster's location data (`Average Income`, `Population Density`) to a "Healthy" cluster. Are the high-risk individuals concentrated in specific types of neighborhoods?

---

## Part 3: Supervised Learning & The Limits of Linearity (Sections 7–8)
*Many traditional epidemiological models assume risks add up linearly. Machine learning trees (LightGBM) allow for non-linear, compounding risks.*

**7. Evaluating the Model**
Look at the classification report and confusion matrix in **Section 7**.
* What is your baseline accuracy (if the model just guessed "No" for everyone)? 
* Does the model successfully outperform this baseline? Why is the **PR AUC** (Precision-Recall Area Under Curve) a better metric to look at here than raw accuracy?

**8. Linear vs. Non-Linear Complexity**
Look at the side-by-side bar chart comparing Logistic Regression (Linear) and LightGBM (Non-Linear/Tree-based) in **Section 8**.
* Does the non-linear model perform significantly better? 
* If there is a noticeable gap in performance, what does that tell you about how variables like BMI, Stress, and Activity interact to cause sleep problems? (Hint: Does a BMI of 31 carry just a tiny bit more risk than a BMI of 29, or is there a threshold effect?)

---

## Part 4: Explainability & Synergistic Risks (Section 9)
*SHAP values let us peek inside the "black box" to see exactly how individual risk factors interact to predict sleep problems.*

**9. Conflicting Feature Importances**
Look at the side-by-side horizontal bar chart comparing *Logistic Regression Coefficients* to *LightGBM SHAP values*.
* Identify a feature that ranks highly in the SHAP (LightGBM) chart but very low in the Logistic Regression chart. 
* Why might the linear model have missed this variable's importance? (Think about thresholds, like `BMI > 30`, or specific subsets of people, like students).

**10. The Protective Shield of Activity**
Look at the **SHAP Beeswarm plot**. 
* Find `Physical Activity Level` on the y-axis. Where do the red dots (high physical activity) land on the x-axis (SHAP value)? 
* What does this tell you about the protective nature of physical activity against sleep problems?

**11. Compounding Interaction Risks**
Scroll down to the SHAP dependence plots. Use the provided function `plot_shap_dependence()` to investigate specific interactions.
* Run `plot_shap_dependence("BMI", "Stress Levels")`. 
* Look at the right side of the plot where BMI is high (> 30). What happens to the risk (y-axis) when an individual *also* has high stress (red dots)? Do these risks just add together slowly, or do they trigger a massive jump in sleep problem risk?

**12. Intersectional Demographics**
Finally, use the dependence plot to look for a highly specific demographic interaction hidden in the data.
* Run `plot_shap_dependence("Age", "Sex_M")` (or `Sex_F` depending on your dummy columns) and keep in mind how stress affects female sleep duration in the generated rules.
* Can you spot a unique non-linear pattern or cluster of dots for young females (Age 15–35) that separates them from males of the same age or older females?