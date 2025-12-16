# ITU BDS MLOPS'25 – Project

## About

This repository contains the group project for the course **Data Science in Production: MLOps and Software Engineering** at the IT University of Copenhagen.

The project is based on a provided machine learning notebook and focuses on applying **MLOps principles** to structure, refactor, and automate a machine learning workflow. The repository demonstrates:

- Refactoring notebook-based ML code into maintainable Python modules
- Using a **cookiecutter MLOps template** as a project foundation
- Experiment and data management with **DVC**
- Automation and reproducibility using **Dagger pipelines** and **CI workflows**

The end goal is to support reproducible training and evaluation of machine learning models in a structured software engineering setup.

## Project structure

```bash
itu-sdse-project/
├── README.md                     # Project overview and description
├── dagger.json                   # Dagger project configuration
├── .dvc/                         # DVC metadata and configuration
├── .github/
│   └── workflows/                # CI/CD workflows (GitHub Actions)
├── cookiecutter-mlops-template/  # Cookiecutter MLOps template used for refactoring
├── dagger/                       # Dagger-based pipeline orchestration
├── docs/                         # Project documentation
└── mlops_refactor/               # Refactored ML project (Cookiecutter-based)
    ├── data/                     # Data directory (tracked via DVC)
    ├── src/                      # Main Python source code
    │   ├── data/                 # Data loading and preprocessing modules
    │   ├── models/               # Model training and definition code
    │   └── pipeline/             # ML pipeline logic
    ├── model_inference.py        # Script for running model inference
    └── requirements.txt          # Python dependencies for ML code
```

## How to run

### On GitHub

The project uses **GitHub Actions** to automatically run checks and pipeline steps defined in the workflows located in `.github/workflows/`.

These workflows are triggered on pushes to the repository and ensure the project remains reproducible and executable in a clean environment.

### Locally

To run the project locally:

1. Clone the repository
2. Create and activate a Python virtual environment
3. Install required dependencies:

```bash
pip install -r mlops_refactor/requirements.txt
```

4. Navigate to the `dagger` directory and run the pipeline:

```bash
go run main.go
```

This command will:
- Start the Dagger engine and connect to it
- Install Python dependencies in the container
- Load and preprocess data from DVC (`mlops_refactor/data/raw/raw_data.csv`)
- Train machine learning models (Logistic Regression, XGBoost) with cross-validation
- Select the best model based on F1-score
- Save the final model to `dagger/artifacts/model`
- Export artifacts to `dagger/artifacts-out/`

After the pipeline finishes, the final model can be found at:

```
dagger/artifacts/model
```

or locally exported at:

```
dagger/artifacts-out/model
```

**Note:** The `dagger.json` file contains only project metadata (name and engine version) and is **not the pipeline configuration** itself. The pipeline logic is defined entirely in `main.go`.

## Authors

- Tetiana Tretiak
- Mariia Zviahintseva
- Mihael Stoyanov

## Materials provided by course

- Original project repository: https://github.com/lasselundstenjensen/itu-sdse-project
- Model validation GitHub Action: https://github.com/lasselundstenjensen/itu-sdse-project-model-validator

