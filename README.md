# AI Research Lab

A Python-based platform for historical-data analysis, statistical research, and machine-learning experimentation.

AI Research Lab provides an interactive environment for exploring datasets, running experiments, evaluating machine-learning models, analyzing predictions, and generating research reports.

**Project focus:** Data analysis, research, experimentation, and predictive modeling. This project is not a live trading or automated trading system.

## Features

- Dataset upload, inspection, and preview
- Data quality validation
- Numerical and categorical data preprocessing
- Regression model training and evaluation
- PyTorch neural-network experiments
- K-fold cross-validation
- Model performance comparison
- Prediction analysis and residual diagnostics
- Experiment and model checkpoint persistence
- SQLite-based experiment storage
- Research report generation
- Reproducible experiment configuration

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Main programming language |
| Streamlit | Interactive application |
| pandas | Data manipulation |
| NumPy | Numerical computing |
| scikit-learn | Classical machine learning |
| PyTorch | Neural networks |
| SciPy | Scientific computing |
| Plotly | Interactive visualization |
| Matplotlib | Data visualization |
| SQLite | Experiment persistence |
| pytest | Testing |

## Machine Learning Models

### Train/Test Experiments

- Linear Regression
- Random Forest Regressor
- Mean Baseline
- PyTorch Neural Network

### Cross-Validation

The current cross-validation implementation supports:

- Linear Regression
- Random Forest Regressor
- Mean Baseline

Neural-network cross-validation is planned for a future development stage.

## Evaluation Metrics

The platform evaluates regression models using:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of Determination

These metrics help compare model performance and understand prediction errors.

## Data Processing

The preprocessing pipeline supports:

**Numerical features**
- Median imputation
- Standard scaling

**Categorical features**
- Most-frequent-value imputation
- One-hot encoding
- Unknown-category handling

Cross-validation uses fold-specific preprocessing to avoid fitting preprocessing transformations on validation-fold data.

## Project Structure

```text
ai-research-lab/
├── app.py
├── components/
├── config/
├── pages/
├── services/
├── src/
│   ├── data/
│   ├── experiments/
│   ├── ml/
│   ├── reports/
│   └── storage/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### Requirements

- Python 3.12
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/bellapacx3/ai-research-lab.git
cd ai-research-lab
```

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
```

Activate it:

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows**

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency file is a snapshot of the development environment. It includes PyTorch and its installed CUDA-related packages, so compatibility may vary across operating systems and hardware.

### 4. Start the application

```bash
streamlit run app.py
```

Open the local URL provided by Streamlit, normally:

```text
http://localhost:8501
```

## Running Tests

Run the test suite:

```bash
pytest -q
```

Compile the application modules:

```bash
python -m compileall -q app.py src services pages components
```

## Experiment Workflow

1. Upload a dataset.
2. Inspect the dataset and its quality.
3. Select the target variable.
4. Configure the input features.
5. Choose an evaluation mode.
6. Select the machine-learning models.
7. Configure model parameters.
8. Run the experiment.
9. Review model metrics and predictions.
10. Inspect saved results and reports.

## Experiment Persistence

The platform uses SQLite to store experiment information and evaluation results.

Model checkpoints and generated research reports are maintained separately from the database.

Experiment metadata and random seeds support reproducibility and comparison between experiments.

## Research Reports

Cross-validation reports support:

- Markdown
- CSV
- JSON

Reports include experiment metadata, model performance summaries, fold-level metrics, and model comparisons.

## Development Status

The project is actively evolving.

| Component | Status |
|---|---|
| Dataset ingestion and inspection | Implemented |
| Data preprocessing | Implemented |
| Classical regression models | Implemented |
| PyTorch neural network | Implemented |
| Train/test experiment runner | Implemented |
| Experiment persistence | Implemented |
| Model checkpoints | Implemented |
| Classical model cross-validation | Implemented |
| Cross-validation reports | Implemented |
| Neural-network cross-validation | Planned |
| Additional research and diagnostics capabilities | In development |

## Future Development

Planned improvements include:

- Neural-network cross-validation
- Additional machine-learning algorithms
- Expanded statistical analysis
- Advanced dataset diagnostics
- More experiment comparison tools
- Enhanced research reporting
- Additional model architectures

## License

All rights reserved unless a license is added to this repository.
