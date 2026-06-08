import tkinter as tk
from tkinter import ttk, messagebox
import json
import time
import joblib
import numpy as np
import re
import tensorflow as tf
from pathlib import Path

# NOTE: Ensure these local imports still work in your folder structure
try:
    from utils.tokenizer import tokenize_text
    from model.cnn_dqa_classifier import ZipfAttentionLayer
    from src.utils.zipf_weightage import compute_zipf_weights
except ImportError:
    # If you moved files, you might need to adjust these paths
    pass

BASE_DIR = Path(__file__).resolve().parent.parent

# -------------------------
# Backend Logic
# -------------------------
def basic_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class PredictionEngine:
    """Class to handle model loading and prediction logic"""
    def __init__(self):
        self.model_path = BASE_DIR / "analytics/results/models"
        self.vocab_path = BASE_DIR / "analytics/results/vocabulary.json"
        self.cached_models = {}
        self.vectorizer = None

    def get_nb_rf(self, name):
        if name not in self.cached_models:
            if not self.vectorizer:
                self.vectorizer = joblib.load(self.model_path / "tfidf_vectorizer.joblib")
            
            file_name = "naive_bayes.joblib" if name == "nb" else "random_forest.joblib"
            self.cached_models[name] = joblib.load(self.model_path / file_name)
        return self.cached_models[name], self.vectorizer

    def get_cnn(self):
        if "cnn" not in self.cached_models:
            model_file = self.model_path / "cnn_dqa_model.keras"
            self.cached_models["cnn"] = tf.keras.models.load_model(
                model_file,
                custom_objects={"ZipfAttentionLayer": ZipfAttentionLayer}
            )
        return self.cached_models["cnn"]

    def predict(self, text, model_name):
        start = time.time()
        
        if model_name in ["nb", "rf"]:
            model, vectorizer = self.get_nb_rf(model_name)
            X = vectorizer.transform([text])
            prob = model.predict_proba(X)[0][1]
        else:
            model = self.get_cnn()
            with open(self.vocab_path) as f:
                vocab = json.load(f)["word_to_idx"]
            
            token_seq = np.expand_dims(tokenize_text(text, vocab), axis=0)
            zipf_seq = np.expand_dims(compute_zipf_weights(token_seq[0]), axis=0)
            prob = model.predict([token_seq, zipf_seq], verbose=0)[0][0]
            
        elapsed = (time.time() - start) * 1000
        return float(prob), elapsed

# -------------------------
# GUI Interface
# -------------------------
class PhishingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentinel: Phishing Email Detector")
        self.root.geometry("700x650")
        self.engine = PredictionEngine() # Init backend
        
        # Styles
        self.style = ttk.Style()
        self.style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"))
        self.style.configure("Result.TLabel", font=("Segoe UI", 14, "bold"))
        
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill="both", expand=True)

        # Inputs
        ttk.Label(main_frame, text="📧 Email Security Scanner", style="Header.TLabel").grid(row=0, column=0, pady=(0, 20), sticky="w")
        
        ttk.Label(main_frame, text="Subject Line:").grid(row=1, column=0, sticky="w")
        self.subject_entry = ttk.Entry(main_frame, width=60)
        self.subject_entry.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Label(main_frame, text="Email Body:").grid(row=3, column=0, sticky="w")
        self.body_text = tk.Text(main_frame, height=10, font=("Segoe UI", 10))
        self.body_text.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(0, 15))

        # Controls
        ctrls = ttk.Frame(main_frame)
        ctrls.grid(row=5, column=0, columnspan=2, sticky="ew")
        
        self.model_var = tk.StringVar(value="cnn")
        ttk.Label(ctrls, text="Model:").pack(side="left")
        ttk.Combobox(ctrls, textvariable=self.model_var, values=("nb", "rf", "cnn"), state="readonly").pack(side="left", padx=10)
        
        ttk.Button(ctrls, text="Clear", command=self.clear_fields).pack(side="right", padx=5)
        ttk.Button(ctrls, text="Analyze Email", command=self.run_analysis).pack(side="right")

        # Visual Result Box
        self.res_box = tk.Frame(main_frame, relief="flat", bg="#f0f0f0", pady=20)
        self.res_box.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(20, 0))
        
        self.status_lbl = tk.Label(self.res_box, text="READY TO SCAN", font=("Segoe UI", 14, "bold"), bg="#f0f0f0")
        self.status_lbl.pack()
        
        self.stats_lbl = tk.Label(self.res_box, text="Enter email details to begin", bg="#f0f0f0")
        self.stats_lbl.pack()

    def clear_fields(self):
        self.subject_entry.delete(0, tk.END)
        self.body_text.delete("1.0", tk.END)
        self.status_lbl.config(text="READY TO SCAN", fg="black")

    def run_analysis(self):
        text = f"{self.subject_entry.get()} {self.body_text.get('1.0', tk.END)}".strip()
        if len(text) < 5:
            messagebox.showwarning("Warning", "Email content is too short to analyze.")
            return

        self.status_lbl.config(text="SCANNING...", fg="blue")
        self.root.update_idletasks()

        try:
            prob, speed = self.engine.predict(basic_clean(text), self.model_var.get())
            
            # Update UI based on result
            is_phish = prob >= 0.5
            color = "#cc0000" if is_phish else "#228b22"
            label = "PHISHING DETECTED" if is_phish else "LEGITIMATE"
            
            self.status_lbl.config(text=label, fg=color)
            self.stats_lbl.config(text=f"Phishing Probability: {prob:.2%} | Time: {speed:.2f}ms")
            self.res_box.config(highlightbackground=color, highlightthickness=2, relief="solid")

        except Exception as e:
            messagebox.showerror("System Error", f"Model Error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PhishingApp(root)
    root.mainloop()
