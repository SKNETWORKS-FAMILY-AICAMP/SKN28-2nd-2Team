import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

model = joblib.load(MODEL_DIR / "best_model.pkl")
feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

x_test_sample = pd.read_csv(PROCESSED_DIR / "x_test_sample.csv")
x_test_sample = x_test_sample[feature_names]

explainer = shap.Explainer(model, x_test_sample)
shap_values = explainer(x_test_sample)

# Summary plot
plt.figure()
shap.summary_plot(shap_values, x_test_sample, show=False)
plt.tight_layout()
plt.savefig(PROCESSED_DIR / "shap_summary.png", bbox_inches="tight")
plt.close()

# Bar plot
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.tight_layout()
plt.savefig(PROCESSED_DIR / "shap_bar.png", bbox_inches="tight")
plt.close()

print("SHAP 시각화 저장 완료")