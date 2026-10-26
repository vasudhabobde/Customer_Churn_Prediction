# Customer Churn Prediction & Analytics

## Files
- `Customer_Churn_Prediction_All_Practicals.ipynb` — Google Colab notebook combining the uploaded SI practical concepts.
- `app.py` — Streamlit application.
- `Telco-Customer-Churn.csv` — supplied dataset.
- `requirements.txt` — Python dependencies.

## Run the notebook in Google Colab
1. Upload `Customer_Churn_Prediction_All_Practicals.ipynb` to Google Colab.
2. Run cells from top to bottom.
3. When the CSV upload cell appears, upload `Telco-Customer-Churn.csv`.
4. The notebook covers web-scraping basics, data inspection/cleaning, distributions, sampling distributions, point estimates, confidence intervals, IQR outliers, hypothesis testing, covariance/correlation, encoding/scaling, churn classification, model evaluation and a separate numeric regression comparison.
5. At the end, download the exported model bundle and cleaned dataset if needed.

## Run Streamlit locally (Windows)
Open Command Prompt in this folder:
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
The app opens in your browser. The CSV is included by default; you can also upload another compatible CSV from the sidebar. If your Python command is different, use the Python installation where Streamlit was installed.

## Important notes
- `Churn` is the target column and should contain `Yes` / `No`.
- `customerID` is excluded from model features because it is an identifier.
- The classification section predicts churn. Regression models are used separately to predict numeric `TotalCharges`, because churn itself is categorical.
- The app trains models when opened and can take a short time on first load.
- Probabilities and risk bands are estimates, not guarantees.
