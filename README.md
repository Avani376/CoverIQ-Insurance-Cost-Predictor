# CoverIQ — Insurance Cost Intelligence

CoverIQ is a Streamlit machine-learning application that explores a public insurance dataset, compares regression models, estimates annual charges for an entered profile, provides beginner-friendly sensitivity explanations, exports PDF reports, and saves user-specific prediction history in SQLite.

> **Important:** This is an educational demonstration using a sample dataset. It does not provide an insurance quote, financial advice, or medical advice.

## Features

- Premium dark dashboard with responsive Streamlit layout
- Dataset exploration and summary statistics
- Linear Regression, Random Forest, and Gradient Boosting
- Train/test evaluation: R², MAE, RMSE, and cross-validation R²
- Actual-versus-predicted visualisation
- Feature importance / coefficient analysis
- One-feature-at-a-time sensitivity explanation against a typical profile
- Downloadable PDF report with profile, estimate, explanation, and model metrics
- Sign-up/sign-in demo with PBKDF2 password hashing
- SQLite account storage and persistent per-user prediction history
- CSV export for history, cleaned dataset, and model comparison

## Project structure

```text
CoverIQ_Full_Project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

The `insurewise.db` SQLite database is created automatically when the app runs. It is intentionally ignored by Git because it contains account and prediction data.

## Run locally (Windows PowerShell)

1. Install Python 3.10 or later.
2. Extract this project folder.
3. Open PowerShell in the extracted folder.
4. (Optional but recommended) create a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

5. Install packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

6. Start the app:

   ```powershell
   python -m streamlit run app.py
   ```

7. Open the local address Streamlit prints (usually `http://localhost:8501`).
8. Create an account in the app, sign in, enter a profile, and click **Calculate my estimate**.

## Dataset

The app first looks for `data/insurance.csv` or `insurance.csv` in the project folder. If it does not find a valid local file, it attempts to download the sample dataset from:

`https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv`

Expected columns:

- `age`
- `sex`
- `bmi`
- `children`
- `smoker`
- `region`
- `charges`

Internet access is needed when no local dataset is available.

## Methodology

1. Numeric and categorical columns are separated.
2. Missing numeric values are imputed with the median; categorical values are imputed with the most frequent category.
3. Categorical values are one-hot encoded; numeric values are standardised.
4. The dataset is split into training and test partitions.
5. Three regression models are trained and compared:
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
6. Metrics are calculated on the held-out test partition. Cross-validation R² is also calculated.
7. The highest test-set R² model is marked as the current best model.

### What the explanation means

The app compares a submitted profile with a typical reference profile (median numeric values and most frequent categories). It changes one input at a time and shows how the model output changes. This is a simple **sensitivity analysis**, not a causal explanation. These changes can overlap and should not be added together.

## PDF reports

After making a prediction, select **Download personalized PDF report**. The report includes the entered profile, annual and monthly estimates, sensitivity comparisons, model metrics, and limitations.

## Accounts and data storage

This is a learning/demo implementation. Passwords are stored as salted PBKDF2-HMAC-SHA256 hashes rather than plain text. Prediction history is associated with the signed-in user in a local SQLite database.

**Before real public use:**
- Use a production authentication provider or carefully reviewed authentication flow.
- Use a managed database such as PostgreSQL for hosted persistence.
- Add HTTPS, secrets management, rate limiting, privacy disclosures, data retention/deletion controls, and security review.
- Do not collect real sensitive personal information in this educational demo.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `README.md`, `.gitignore`, and `.streamlit/config.toml`.
2. Do **not** commit `insurewise.db`, `.venv`, or personal data.
3. Open Streamlit Community Cloud, connect GitHub, choose the repository and branch, and set the main file to `app.py`.
4. Deploy and test the app using a new account and a test prediction.

### Important hosting limitation

A local SQLite file on a hosted app may not be durable across restarts, redeploys, or infrastructure changes. For reliable account and prediction persistence on a public deployment, connect a managed PostgreSQL database and store credentials as deployment secrets. This project defaults to SQLite for local development and classroom demonstration.

## Screenshots for the portfolio

Capture screenshots after running the app:
1. `screenshots/01-dashboard.png` — overview and KPI cards
2. `screenshots/02-prediction.png` — prediction result and explanation
3. `screenshots/03-model-evaluation.png` — model comparison
4. `screenshots/04-history.png` — saved prediction history
5. `screenshots/05-pdf-report.png` — generated PDF report

Create a `screenshots/` directory and add your actual screenshots. Do not use placeholder screenshots or claim a deployment is live until it has been tested.

## Model metrics interpretation

- **R²:** fraction of target variation explained on the held-out test set; closer to 1 is generally better.
- **MAE:** average absolute prediction error in the target's units; lower is better.
- **RMSE:** square root of mean squared error; larger errors have more influence; lower is better.
- **Cross-validation R²:** average R² across validation folds; it may differ from the held-out test result.

A score is not the same as individual prediction accuracy. Performance depends on the dataset, split, and model settings.

## Limitations and future improvements

- Public sample data may not reflect current insurance pricing or all populations.
- Several important factors such as policy details and provider-specific rules are not represented.
- Sensitivity analysis is not causal inference.
- SQLite is appropriate for local demos, not concurrent public production use.
- Potential future work: managed database, SHAP-based explanations, automated tests, model versioning, data drift monitoring, and a CI deployment workflow.

## License

Add a license file if you intend to distribute or publish this project. Check the dataset source terms before redistribution.


## CoverIQ theme update
The interface uses a dark navy, violet, and electric-blue visual theme with subtle CSS motion and hover effects. The database filename remains `insurewise.db` for compatibility with an existing local account database.
