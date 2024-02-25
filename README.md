# Workshop on Pattern Recognition in a dataset

Held in the afternoon of the 27th of February 2024 as part of the PhD course: [Introduction to Complex Systems Approaches in Public Health](https://phdcourses.dk/Course/106362)

## Introduction
By the end of the afternoon session, participants will have hands-on experience using R or Python to apply simple approaches for pattern recognition on a Public Health dataset. No prior programming experience is required but R (+ Rstudio recommended) or Python (+ VScode recommended) will be needed to run the necessary code.

The instructions are simple, download this repository and try to run the provided code. Once you achieve this, explore which patterns you notice at the different steps. 

## Data
The data is a synthetic dataset from a Kaggle competitions with the title [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset). Follow the link for information on the variables.

## Setting up your environment

### R users
To work with your current installation of R (and Rstudio), uncomment the first lines in the scripts to install the necessary packages. I suggest to install the [pak library](https://pak.r-lib.org/) first to manage the installation of the other packages. You can install it with the following command:

```R
install.packages("pak", repos = "https://r-lib.github.io/p/pak/devel/")
pak::pkg_install(c("skimr", "tidyverse", "tidymodels", "corrplot", "ranger", "treeshap", "shapviz"))
```

### Python users
Instructions to create a conda environment are in the [conda environment file](scripts/patterns_env.yml) and necessary packages in the `pip` [requirements file](scripts/requirements.txt). You can create the environment with the necessary packages executing the following commands in your terminal (assuming you are at the root directory of this repository):

```bash
  conda env create -f ./scripts/patterns_env.yml
  conda activate patterns_env
  pip install -r ./scripts/requirements.txt
```

Execution of the Python scripts, cell by cell, has been tested in [Visual Studio Code](https://code.visualstudio.com/). The script can also be exported as a Jupyter notebook.
