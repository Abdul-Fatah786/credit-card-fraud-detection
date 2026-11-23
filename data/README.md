# Data

Put the Kaggle file here as **`data/creditcard.csv`** (about 150 MB, not committed to git).

Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  (free Kaggle account needed)

Or with the Kaggle CLI:

    pip install kaggle
    kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip

Facts: 284,807 transactions over two days, 492 fraud (0.172%), columns `Time`, `V1`-`V28` (anonymised PCA features),
`Amount`, and the target `Class` (1 = fraud).

`sample_transactions.csv` is a small demo file created by `python -m src.train` (every test-set fraud plus 1,500
random legitimate transactions). The Streamlit app and the API `/sample` endpoint use it.
