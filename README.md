# Phishing Email Detection (Synthetic Enron + Nazario-like dataset)

This project is a self-contained phishing email detection demo. It uses a preprocessed which is the
combined dataset (Enron-style ham emails + Nazario-style phishing samples), trains classification models (Naive Bayes and SVM), selects the best model, and provides a simple Streamlit UI to paste an email and classify it.

## Files
`data/combined.csv` - combined synthetic dataset (ham + phishing)
 `src/preprocess.py` - text cleaning function
 `src/train.py` - script to retrain models and save artifacts
 `src/app.py` - Streamlit app to classify pasted emails
 `models/vectorizer.pkl` - trained TF-IDF vectorizer
 `models/model.pkl` - trained model (best of Naive Bayes / SVM)
 `requirements.txt` - Python packages to install

## Quickstart (run locally)
1. Clone or extract the zip.
2. Create a virtual environment and activate it:
 python -m venv venv 
# Windows
venv\\Scripts\\activate
# macOS / Linux
source venv/bin/activate

3. Install requirements:
pip install -r requirements.txt

4. (Optional) Retrain models (will overwrite `models/` artifacts)
python src/train.py

5. Launch the Streamlit app:python -m streamlit run src/app.py --server.headless true

6. In the app, paste an email and click **Classify**.


