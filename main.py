import joblib
import numpy as np
import os
import re
import sys 
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager


class LogisticRegression:
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        # Add a small epsilon to prevent overflow/underflow
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped) + 1e-15)

    def fit(self, X, y):
        # Not used for prediction, but part of the class
        pass 

    def predict_proba(self, X):
        if self.weights is None or self.bias is None:
            raise Exception("Model not loaded properly.")
        if hasattr(X, "toarray"):
            X = X.toarray()
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.array([1 if i > 0.5 else 0 for i in probabilities])

def preprocess_text(text):
    
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'https?:\/\/\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

setattr(sys.modules['__main__'], 'LogisticRegression', LogisticRegression)
setattr(sys.modules['__main__'], 'preprocess_text', preprocess_text)
MODELS_DIR = "models"
FAKEDDIT_MODEL_PATH = os.path.join(MODELS_DIR, "fakeddit_svm_model.joblib")
POLITICAL_SVM_PATH = os.path.join(MODELS_DIR, "svm_fake_news_model.pkl")
POLITICAL_TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
LOGREG_MODEL_PATH = os.path.join(MODELS_DIR, "logreg_fakenews.joblib")

# Fakeddit 6-way labels
FAKEDDIT_LABEL_MAP = {
    0: "True / Factual",
    1: "Sarcasm / Satire",
    2: "Misleading / Exaggeration",
    3: "Imposter Content",
    4: "False Context",
    5: "Manipulated Content"
}
# Scratch-LR labels
LOGREG_LABEL_MAP = {0: 'FALSE (FAKE)', 1: 'TRUE (REAL)'}

model_store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- This code runs ON STARTUP ---
    print("--- Athena AI Server Booting Up ---")
    
    # --- Model 1: Fakeddit ---
    try:
        model_store["FAKEDDIT_MODEL"] = joblib.load(FAKEDDIT_MODEL_PATH)
        print(f"✅ Fakeddit model loaded from '{FAKEDDIT_MODEL_PATH}'")
    except Exception as e:
        print(f"❌ ERROR loading Fakeddit model: {e}")
        model_store["FAKEDDIT_MODEL"] = None

    # --- Model 2: Political SVM ---
    try:
        model_store["POLITICAL_SVM"] = joblib.load(POLITICAL_SVM_PATH)
        model_store["POLITICAL_TFIDF"] = joblib.load(POLITICAL_TFIDF_PATH)
        print(f"✅ Political SVM & TFIDF loaded.")
    except Exception as e:
        print(f"❌ ERROR loading political models: {e}")
        model_store["POLITICAL_SVM"] = None
        model_store["POLITICAL_TFIDF"] = None

    # --- Model 3: Scratch-built LR ---
    try:
        pipeline = joblib.load(LOGREG_MODEL_PATH)
        model_store["LOGREG_VECTORIZER"] = pipeline['vectorizer']
        model_store["LOGREG_MODEL"] = pipeline['model']
        print(f"✅ Scratch LR model loaded from '{LOGREG_MODEL_PATH}'.")
    except Exception as e:
        print(f"❌ ERROR loading Scratch LR model: {e}")
        model_store["LOGREG_VECTORIZER"] = None
        model_store["LOGREG_MODEL"] = None
    
    print("--- Server is running. Ready for analysis. ---")
    yield

    model_store.clear()
    print("--- Models cleared. Server shutting down. ---")


app = FastAPI(
    title="Athena AI News Analyzer",
    description="Serves THREE different fake news models.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str


# --- Homepage Endpoint ---
@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    try:
        with open("index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found.</h1>", status_code=404)

# --- Fakeddit Model Endpoint ---
@app.post("/analyze/fakeddit")
async def analyze_fakeddit(request: TextRequest):
    if not model_store.get("FAKEDDIT_MODEL"):
        raise HTTPException(status_code=500, detail="Fakeddit model is not loaded. Check server logs.")
    
    try:
        predicted_id = model_store["FAKEDDIT_MODEL"].predict([request.text])[0]
        key_indicator = FAKEDDIT_LABEL_MAP[predicted_id]

        if predicted_id == 0:
            result = "True"
            indicators = ["True / Factual"]
        else:
            result = "False / Suspicious"
            indicators = [key_indicator]

        return {
            "model_used": "Fakeddit (6-Way SVM)",
            "result": result,
            "key_indicators": indicators
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

# --- Political SVM Model Endpoint ---
@app.post("/analyze/political")
async def analyze_political(request: TextRequest):
    if not model_store.get("POLITICAL_SVM") or not model_store.get("POLITICAL_TFIDF"):
        raise HTTPException(status_code=500, detail="Political model is not loaded. Check server logs.")
        
    try:
        vec = model_store["POLITICAL_TFIDF"].transform([request.text])
        pred = model_store["POLITICAL_SVM"].predict(vec)
        result = "True" if pred[0] == 1 else "False"

        return {
            "model_used": "Political (Scikit-learn SVM)",
            "result": result,
            "key_indicators": [result]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

# --- Scratch-LR Model Endpoint ---
@app.post("/analyze/logreg")
async def analyze_logreg(request: TextRequest):
    if not model_store.get("LOGREG_MODEL") or not model_store.get("LOGREG_VECTORIZER"):
        raise HTTPException(status_code=500, detail="Scratch LR model is not loaded. Check server logs.")
        
    try:
        # 1. Clean the text
        cleaned_text = preprocess_text(request.text)
        
        # 2. Vectorize
        vec = model_store["LOGREG_VECTORIZER"].transform([cleaned_text])
        
        # 3. Predict
        pred = model_store["LOGREG_MODEL"].predict(vec)[0]
        label = LOGREG_LABEL_MAP.get(pred, "Unknown")
        result = "True" if pred == 1 else "False / Suspicious"
        
        return {
            "model_used": "Logistic Regression (Scratch Trained)",
            "result": result,
            "key_indicators": [label] # e.g., "FALSE (FAKE)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

# --- Uvicorn server (if run directly) ---
if __name__ == "__main__":
    import uvicorn
    print("--- Starting Uvicorn server at http://127.0.0.1:8000 ---")
    uvicorn.run(app, host="127.0.0.1", port=8000)