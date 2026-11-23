# 💳 Credit Card Fraud Detection & Risk Monitoring

Detects fraudulent credit card transactions in a highly imbalanced
dataset (0.172% fraud), compares several models, tunes the decision
threshold, estimates business value, and serves the model through a
**Streamlit dashboard** and a **FastAPI** service.

> Educational project built on the public Kaggle dataset. Not a real
> bank system. Dollar figures are assumptions.

## What this project shows

-   **Imbalanced classification:** why accuracy is useless here (a model
    that always says "legitimate" scores 99.83% and catches nothing).
-   **Models:** Dummy baseline, Logistic Regression, Decision Tree,
    Random Forest, optional XGBoost.
-   **Imbalance methods:** class weights, SMOTE (training data only),
    threshold tuning.
-   **Honest evaluation:** train / validation / test split, threshold
    chosen on validation, test set used once; optional time-based split.
-   **Metrics:** precision, recall, F1, PR-AUC, ROC-AUC, confusion
    matrix, top-K prioritisation.
-   **Business simulation:** net benefit from prevented fraud minus
    investigation cost.
-   **Deployment:** Streamlit dashboard + FastAPI REST API, tests,
    Docker files.

## Project structure

``` text
credit-card-fraud-detection/
├── data/
│   ├── README.md                  # how to get creditcard.csv
│   └── creditcard.csv             # (you add this; not committed)
├── notebooks/
│   └── fraud_detection_experiments.ipynb   # the standalone Google Colab notebook
├── src/
│   ├── config.py                  # paths, columns, risk bands, business defaults
│   ├── data_loader.py             # load_data(), split_data() (random or time-based)
│   ├── preprocessing.py           # features, scaling, validation, risk levels (shared by app + API)
│   ├── train.py                   # trains all experiments, picks the best, saves everything
│   ├── evaluate.py                # metrics, threshold table, plots
│   ├── predict.py                 # score new transactions with the saved model
│   ├── utils.py                   # business simulation, top-K
│   └── eda.py                     # exploratory charts
├── models/
│   └── best_model.pkl             # created by src.train
├── reports/
│   ├── experiment_results.csv     # created by src.train
│   ├── threshold_analysis.csv
│   ├── final_report.json
│   └── figures/                   # PR/ROC curves, confusion matrix, EDA charts
├── api/
│   ├── main.py                    # FastAPI app
│   └── schemas.py                 # request / response models
├── tests/                         # pytest suite (preprocessing, training, API)
├── app.py                         # Streamlit dashboard
├── requirements.txt
├── requirements-dev.txt
├── requests.http                  # ready-made API calls for the VS Code REST Client
├── Dockerfile, docker-compose.yml
├── pytest.ini
├── .gitignore
└── .vscode/                       # run/debug configs, settings
```

## Setup in VS Code

1.  Install **Python 3.10+** and VS Code with the extensions **Python**,
    **Jupyter** (and optionally **REST Client**).

2.  Open this folder in VS Code (`File → Open Folder`).

3.  Create a virtual environment and install packages. In the VS Code
    terminal:

    Windows (PowerShell):

    ``` powershell
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements-dev.txt
    ```

    macOS / Linux:

    ``` bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements-dev.txt
    ```

    If PowerShell blocks activation, run
    `Set-ExecutionPolicy -Scope Process RemoteSigned` first.

4.  Press `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the
    `.venv` one.

5.  Download the data (see `data/README.md`) and place it at
    `data/creditcard.csv`.

## Run it (always from the project root)

  ------------------------------------------------------------------------------------------
  Step                    Command                           Result
  ----------------------- --------------------------------- --------------------------------
  EDA charts              `python -m src.eda`               figures in `reports/figures/`

  Train                   `python -m src.train`             `models/best_model.pkl`,
                                                            `reports/*`,
                                                            `data/sample_transactions.csv`

  Dashboard               `streamlit run app.py`            http://localhost:8501
  Dashboard Deployed      `streamlit run app.py`            https://credit-card-fraud-detection-rpyhskylxy9iwwqo4vw5m4.streamlit.app/

  API                     `uvicorn api.main:app --reload`   http://127.0.0.1:8000/docs

  Tests                   `pytest`                          all tests pass
  ------------------------------------------------------------------------------------------

Training options: `--full` (bigger forests, slower), `--split time`
(time-based split), `--no-xgb`, `--save-all`. You can also use the **Run
and Debug** panel (F5): configs 1--5 are already set up in
`.vscode/launch.json`.

## API

  -----------------------------------------------------------------------
  Method                  Path                    Purpose
  ----------------------- ----------------------- -----------------------
  GET                     `/health`               service and model
                                                  status

  GET                     `/model-info`           chosen model,
                                                  threshold, test metrics

  GET                     `/sample?fraud=true`    a real example
                                                  transaction (with
                                                  `fraud=true`, the fraud
                                                  case the model is most
                                                  sure about)

  POST                    `/predict`              score one transaction
                                                  (JSON with `Time`,
                                                  `V1`..`V28`, `Amount`)

  POST                    `/predict/batch`        score a list of
                                                  transactions

  POST                    `/predict/csv`          upload a CSV in
                                                  `creditcard.csv` format
  -----------------------------------------------------------------------

Optional query parameters: `threshold` (0--1) on all predict endpoints,
`only_flagged` on `/predict/csv`.

Example (PowerShell or bash, after training):

``` bash
curl http://127.0.0.1:8000/sample?fraud=true            # copy the JSON it returns
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '<paste JSON here>'
```

Response:

``` json
{"fraud_probability": 0.97, "risk_level": "Critical", "decision": "Flag for review", "threshold": 0.5}
```

The numbers above are an illustration of the format, not real output.
You can also use the interactive page at `/docs` or `requests.http`.

## Results

`<small>`{=html}Best experiment: **XGBoost (Experiment 8)** · Random
train/validation/test split · Decision threshold:
**0.20**`</small>`{=html}

The threshold of 0.20 was selected using validation data. The following
metrics and business results are from the held-out test set.

### Test-set model performance

  Metric          Result
  ----------- ----------
  Accuracy      99.9508%
  Precision       90.70%
  Recall          79.59%
  F1-score        84.78%
  PR-AUC          0.8576
  ROC-AUC         0.9795

### Confusion matrix

                            Predicted fraud   Predicted legitimate
  ----------------------- ----------------- ----------------------
  **Actual fraud**              **78 (TP)**            **20 (FN)**
  **Actual legitimate**          **8 (FP)**        **56,856 (TN)**

At the selected threshold, the model correctly detected **78 fraudulent
transactions**, missed **20**, and raised **8 false alarms**. It
generated **86 alerts** in total.

### Business simulation

  Measure                             Result
  --------------------------- --------------
  Alerts raised                           86
  Fraud detected                          78
  Fraud missed                            20
  Assumed losses prevented          \$12,480
  Investigation cost                   \$430
  **Estimated net benefit**     **\$12,050**
  Loss with no model                \$19,600

The estimated net benefit is calculated as assumed losses prevented
minus investigation costs. These figures are illustrative and depend on
the project's business assumptions; they are not actual bank savings or
measured real-world outcomes.

### Top-K fraud prioritisation

Top-K evaluation ranks transactions by predicted fraud probability and
considers the first K transactions for investigation.

  -----------------------------------------------------------------------
              Top K      Frauds found       Precision@K Share of all test
       transactions                                           fraud found
  ----------------- ----------------- ----------------- -----------------
                 50                50           100.00%            51.02%

                100                82            82.00%            83.67%

                200                85            42.50%            86.73%

                500                87            17.40%            88.78%
  -----------------------------------------------------------------------

For example, reviewing the 100 highest-risk transactions would identify
82 fraud cases in this test set, corresponding to 83.67% of all test-set
fraud cases.

## Limitations

-   Features `V1`--`V28` are anonymised PCA components, so individual
    predictions cannot be explained in business terms.
-   Only two days of European cardholder transactions from 2013; fraud
    patterns change over time.
-   Business figures (average loss, alert cost, prevention rate) are
    assumptions.
-   The threshold is tuned for F1 on validation data; a real bank would
    tune it to its own costs.

## Notebook

`notebooks/fraud_detection_experiments.ipynb` is a self-contained
version of the whole workflow for Google Colab or Jupyter (it downloads
the data itself and builds its own copy of the app).
