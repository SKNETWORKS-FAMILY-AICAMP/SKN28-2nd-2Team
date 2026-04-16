import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

model = joblib.load(MODEL_DIR / "best_model.pkl")

# Label Encoding 버전 테스트 데이터 사용
X_test = pd.read_csv(PROCESSED_DIR / "X_test_label.csv")

# 샘플 100개로 SHAP 계산 (속도)
x_sample = X_test.head(100)

explainer = shap.Explainer(model, x_sample)
shap_values = explainer(x_sample)

# Summary plot
plt.figure()
shap.summary_plot(shap_values, x_sample, show=False)
plt.tight_layout()
plt.savefig(PROCESSED_DIR / "shap_summary.png", bbox_inches="tight", dpi=150)
plt.close()

# Bar plot
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.tight_layout()
plt.savefig(PROCESSED_DIR / "shap_bar.png", bbox_inches="tight", dpi=150)
plt.close()

print("SHAP 시각화 저장 완료")
print(f"  - {PROCESSED_DIR / 'shap_summary.png'}")
print(f"  - {PROCESSED_DIR / 'shap_bar.png'}")
