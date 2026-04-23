# Pattern Recognition in Public Health

Workshop for the course [Introduction to Complex Systems Approaches in Public Health](https://phdcourses.ku.dk/detailkursus.aspx?id=113504&sitepath=SUND)

## What you will learn

By the end of this session you will have worked through practical examples in both R and Python covering:

1. **Exploratory Data Analysis** — understanding variable distributions and correlations
2. **Unsupervised learning** — finding subgroups with K-means and HDBSCAN clustering
3. **Supervised learning** — predicting sleep problems with a LightGBM model
4. **Model explainability** — interpreting predictions with SHAP values

No prior programming experience is required. Both R and Python scripts follow the similar steps, so you can work in whichever language you prefer.

## Quick start

1. **Download** this repository (green *Code* button → *Download ZIP*, then unzip) or clone it with `git clone`.
2. **Install an IDE** — see recommendations below.
3. **Set up your environment** — see the language-specific instructions below.
4. **Open the script** for your language and run it cell by cell. Read the comments, change parameters, and explore! We recommend keeping `reflection_guide.md` open alongside the code to guide your analysis.

## Interactive development environments (IDEs)

[**Positron**](https://positron.posit.co/) is a nice choice for this workshop. It supports both R and Python in the same application and has a clean notebook-style interface for running code cell by cell.

Alternatives that also work well:

- **R users:** [RStudio](https://posit.co/download/rstudio-desktop/) — open the `.Rproj` file at the project root and paths will be set automatically.
- **Python users:** [VS Code](https://code.visualstudio.com/) with the Python extension — open the `scripts/` folder and run cells interactively with `Shift+Enter`.

## Files

```
├── data/
│   ├── individuals.csv   # 1000 synthetic individuals 
│   ├── families.csv      # Links individuals to families and family income
│   └── locations.csv     # 10 synthetic locations and their parameters
├── scripts/
│   ├── discovering_patterns.qmd    # R Quarto notebook (recommended for R users to run cell by cell)
│   ├── discovering_patterns.py     # Python script (run cell-by-cell in VS Code or Positron)
│   ├── discovering_patterns.ipynb  # Jupyter notebook version of the Python script
│   ├── patterns_env.yml            # Conda environment specification for Python dependencies
│   └── requirements.txt            # pip dependencies
└── reflection_guide.md    # Guide for students to follow the coding exercise

```

### Dataset descriptions

| File | Contents |
|------|----------|
| `individuals.csv` | One row per person: sex, age, occupation, location, physical activity level, stress levels, BMI, depressive symptoms, sleep duration, sleep problems |
| `families.csv` | `Family ID`, `Family Income`, `Person ID` — links individuals to family-level income |
| `locations.csv` | One row per location: population density, average income, average wealth |

These datasets are entirely synthetic and were generated using a script with particular patterns encoded in the data. Your task is to discover these patterns using the techniques covered in the workshop!

## Setting up your environment

### R

Open the `.Rproj` file at the project root in RStudio or Positron. This sets the working directory automatically so all relative paths work.

Then open `scripts/discovering_patterns.qmd` (Quarto notebook, recommended) and run cells one at a time. The first code cell in both files will automatically install any missing packages.

### Python

Create and activate a conda environment (run once in your terminal from the project root):

```bash
conda env create -f ./scripts/patterns_env.yml
conda activate patterns_env
uv pip install -r ./scripts/requirements.txt
```

Then open either:

- `scripts/discovering_patterns.ipynb` a Jupyter notebook
- `scripts/discovering_patterns.py` in VS Code or Positron can also be run cells by cell.

The first code cell in both files will automatically install any missing packages.


> [!TIP]
> If you cannot set up R or Python to run in your computer, you can also work with the Python notebook in Google Colab by clicking the link below:
[discovering_patterns_collab.ipynb](https://colab.research.google.com/github/adrigabzu/patterns_complexity_public_health_workshop/blob/main/scripts/discovering_patterns_collab.ipynb)
