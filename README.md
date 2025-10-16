# Bone Marrow Transplantation — ML Pipeline

End-to-end scikit-learn pipeline to predict patient survival from the Bone Marrow Transplantation dataset (UCI). It demonstrates clean preprocessing, dimensionality reduction, and model selection—all reproducibly wired into a single Pipeline.

## Highlights

- Load ARFF data with `scipy.io.arff` → pandas DataFrame
- Clean and type-cast columns, encode binary features to 0/1
- Column-wise preprocessing: categorical (impute + OneHotEncode) and numeric (impute + scale)
- Dimensionality reduction with PCA
- Classification with Logistic Regression
- Hyperparameter tuning via GridSearchCV

## Project layout

- `bone-marrow.arff` — dataset in ARFF format (expected at repo root)
- `script.py` — standalone Python script that builds, trains, and tunes a classifier
- `bone_marrow_pipeline.ipynb` — interactive Jupyter notebook with EDA, visualizations, and detailed explanations
- `project_overview.md` — brief project background
- `requirements.txt` — Python dependencies
- `LICENSE` — MIT License
- `README.md` — this guide

## Quickstart

1) Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2) Install dependencies:

```bash
pip install -r requirements.txt  # or: pip install numpy pandas scikit-learn scipy
```

3) Run the pipeline:

**Option A: Python Script** (quick, command-line)
```bash
python script.py
```

**Option B: Jupyter Notebook** (interactive, with visualizations)
```bash
jupyter notebook bone_marrow_pipeline.ipynb
# or use: jupyter lab bone_marrow_pipeline.ipynb
```

You should see output like:

- Unique-value counts per column
- Names of columns with missing values
- Baseline pipeline accuracy on the test set
- The best model (after GridSearchCV) and its hyperparameters
- Final test-set accuracy of the best model

## Reproducibility and configuration

Key settings you can tweak in `script.py`:

- Train/test split: `test_size=0.2`, `random_state=42`
- PCA components in the search space: `pca__n_components`
- Logistic Regression strength: `clf__C`
- Preprocessing choices inside `cat_vals` and `num_vals`

To add more models, extend `search_space` with alternative estimators (e.g., `RandomForestClassifier`) and corresponding hyperparameters, and swap the pipeline’s `clf` as needed.

## How it works (under the hood)

1. Load: `bone-marrow.arff` → DataFrame
2. Drop `Disease` column (dataset-specific cleanup)
3. Coerce columns to numeric (`errors='coerce'`), encode binary columns to 0/1
4. Split to `X` (features) and `y` (`survival_status`), drop `survival_time` from `X`
5. Identify categorical vs numeric columns by cardinality (≤7 unique → categorical)
6. Build preprocessing with `ColumnTransformer`
7. Fit a Pipeline: preprocess → PCA → LogisticRegression
8. Tune PCA and C with GridSearchCV (5-fold CV)

## Troubleshooting

- File not found: Ensure `bone-marrow.arff` exists at the repository root.
- Different schema: If your ARFF doesn’t have `survival_status`/`survival_time` or includes/omits `Disease`, adjust the column operations in `script.py`.
- Convergence warnings: Consider increasing `max_iter` in `LogisticRegression()` (e.g., `max_iter=1000`) or scaling features (already done) and checking class balance.
- SciPy/NumPy mismatches: Upgrade both packages to compatible versions.

## Extending the project

- **Interactive exploration**: Use `bone_marrow_pipeline.ipynb` for detailed EDA, visualizations, and step-by-step experimentation
- Add metrics: precision/recall/F1, ROC AUC, confusion matrix (already in notebook!)
- Persist models: `joblib.dump(best_model, 'models/best_model.joblib')` (code included in notebook)
- Inference script: load the saved pipeline and run predictions on new CSV/JSON
- Experiment tracking: log results to CSV, MLflow, or Weights & Biases
- Testing: smoke test for pipeline fit and basic assertions on output shapes

## Dataset notes

This repo assumes columns `survival_status` (target) and `survival_time` exist, and removes `Disease`. If your ARFF differs, please update the column selection and preprocessing accordingly.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
