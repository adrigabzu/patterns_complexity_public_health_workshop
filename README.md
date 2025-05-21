# Pattern Recognition in Public Health

Workshop for the course [Introduction to Complex Systems Approaches in Public Health](https://phdcourses.ku.dk/detailkursus.aspx?id=112103&sitepath=SUND)

## Introduction
By the end of this session, participants will be exposed to practical examples in R and Python for pattern recognition on a synthetic dataset. No prior programming experience is required, but R (+ RStudio recommended) or Python (+ VS Code recommended) will be needed to run the necessary code.

The instructions are simple, download this repository and try to run the provided code in the script of your programming language of choice. Both R and Python scripts are provided and have the same steps and similar methods. Once you achieve this, explore which interesting patterns you recognise with the different methods implemented.

Take this opportunity to change variables or parameters to explore the impact of different choices in the methods. If you are not familiar with the methods, you can check them online or ask about it. If you are familiar with the methods, take the time to explore new variables and how that changes your question and results.

## Data
These datasets are entirely artificial and were generated using Large Language Models (LLMs) to support hands-on learning in pattern recognition without exposing any real personal data.

- `data/families.csv`: This dataset contains information about family income and the individuals belonging to each family. The file has three columns: `Family ID`, `Family Income`, and `Person ID`. Each row represents an individual, and the `Family ID` and `Family Income` are repeated for each individual in the same family.

- `data/individuals.csv`: This dataset contains information about individuals, including their sex, age, occupation, location, physical activity level, stress levels, BMI, depressive symptoms, sleep duration, and sleep problems. Each row represents an individual, and the columns provide various attributes for each individual.

- `data/locations.csv`: This dataset contains information about different locations, including their population density, average income, and average wealth. Each row represents a location, and the columns provide various attributes for each location.

## Setting up your environment

### R users
To work with your current installation of R (and Rstudio), uncomment the first lines in the scripts to install the necessary packages. I suggest to install the [pak library](https://pak.r-lib.org/) first to manage the installation of the other packages. You can install it with the following command:

```R
install.packages("pak")
pak::pkg_install(c(
  "skimr", "tidyverse", "tidymodels", "corrplot", "lightgbm", "bonsai",
  "treeshap", "shapviz", "GGally", "umap", "dbscan", "viridis", "ggrepel"
))
```
Alternatively you can use:

```R
install.packages(c(
  "skimr", "tidyverse", "tidymodels", "corrplot", "lightgbm", "bonsai",
  "treeshap", "shapviz", "GGally", "umap", "dbscan", "viridis", "ggrepel"
))
```

### Python users
Instructions to create a conda environment are in the [conda environment file](scripts/patterns_env.yml) and necessary packages in the [requirements file](scripts/requirements.txt). You can create the environment with the following commands in your terminal (assuming you are at the root directory of this repository):

```bash
conda env create -f ./scripts/patterns_env.yml
conda activate patterns_env
uv pip install -r ./scripts/requirements.txt
```

Execution of the Python scripts, cell by cell, has been tested in [Visual Studio Code](https://code.visualstudio.com/). The script can also be exported as a Jupyter notebook.

You can delete later the conda environment with the following command:

```bash
conda remove -n patterns_env --all
```