#!/bin/bash

cd "$(dirname "$0")"

echo "=============================="
echo "1. 라이브러리 설치"
echo "=============================="
python3 -m pip install -r requirements.txt

echo "=============================="
echo "2. 모델 학습 및 processed 파일 생성"
echo "=============================="
python3 src/train.py

echo "=============================="
echo "3. SHAP 시각화 생성"
echo "=============================="
python3 src/shap_analysis.py

echo "=============================="
echo "4. Streamlit 실행"
echo "=============================="
streamlit run app/app.py