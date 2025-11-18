# Fake News Detection

A modern web app for AI-powered content verification. Paste news articles, headlines, or statements and analyze them for credibility using multiple machine learning models.

## Features
- **Multiple Models:**
  - Smart 6-Way (SVM, Fakeddit)
  - Fast True/False (Political, Scikit-learn)
  - Deep Reasoning (Logistic Regression)
- **Beautiful UI:** Responsive, dark-themed interface with Tailwind CSS.
- **Clipboard Tools:** Paste and clear content easily.
- **Confidence Score & Key Indicators:** Visual feedback on results.

## Project Structure
```
├── index.html         # Main web app (UI + JS)
├── main.py            # FastAPI backend (serves ML models)
├── requirement.txt    # Python dependencies
├── script.js          # (Unused, logic is in index.html)
├── style.css          # Custom styles (mostly handled by Tailwind)
└── models/
    ├── fakeddit_svm_model.joblib
    └── logreg_fakenews.joblib
```

## Setup Instructions

### 1. Clone the Repository
```sh
git clone https://github.com/yourusername/fake_news_detection_model.git
cd fake_news_detection_model
```

### 2. Install Python Dependencies
```sh
pip install -r requirement.txt
```

### 3. Start the Backend Server
```sh
uvicorn main:app --reload
```

### 4. Open the Web App
Just open `index.html` in your browser. Make sure the backend is running at `http://127.0.0.1:8000`.

## Usage
1. Paste or type the content you want to analyze.
2. Select the analysis model.
3. Click **Run Analysis**.
4. View the verdict, confidence score, and key indicators.

## Requirements
- Python 3.8+
- FastAPI
- Uvicorn
- scikit-learn
- joblib
- (See `requirement.txt` for full list)

## Model Files
Model files are stored in the `models/` directory. You can replace them with your own trained models if needed.

## Dataset Link
1. https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets
2. https://www.kaggle.com/datasets/vanshikavmittal/fakeddit-dataset?select=multimodal_only_samples
   
## Customization
- All frontend logic is in `index.html`.
- Backend logic is in `main.py`.
- You can add more models or endpoints by editing `main.py` and updating the UI accordingly.


---
