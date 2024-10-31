import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 학습된 모델 불러오기
with open("trained_model.pkl", "rb") as file:
    model = pickle.load(file)

# 인코더 및 스케일러 초기화 (모델 학습 시 사용된 인코딩을 동일하게 적용)
categorical_columns = ['구분', '주심', '요일', '날씨', '경기장', 'pre_h_승패여부']
label_encoders = {col: LabelEncoder() for col in categorical_columns}
scaler = StandardScaler()

st.title("스틸러스 관중 예측 애플리케이션")
st.write("정해진 형식의 CSV 파일을 업로드하여 경기 관중 수를 예측합니다.")

# CSV 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")

if uploaded_file:
    # 파일 읽기
    input_data = pd.read_csv(uploaded_file)
    st.write("업로드된 데이터 미리보기:")
    st.write(input_data.head())
    
    # 데이터 전처리
    input_data['일자'] = pd.to_datetime(input_data['일자'], errors='coerce')
    input_data['연도'] = input_data['일자'].dt.year
    input_data['월'] = input_data['일자'].dt.month
    input_data['일'] = input_data['일자'].dt.day
    input_data = input_data.drop(columns=['일자'])

    input_data['시간'] = pd.to_datetime(input_data['시간'], format='%H:%M:%S').dt.hour

    # 범주형 변수 인코딩
    for col, le in label_encoders.items():
        input_data[col] = le.fit_transform(input_data[col])

    input_data['Bteam'] = LabelEncoder().fit_transform(input_data['Bteam'])
    
    # 타겟 변수 제외
    X_input = input_data.drop(columns=['관중'])
    
    # 스케일링 적용
    X_input_scaled = scaler.fit_transform(X_input)
    
    # 예측
    predictions = model.predict(X_input_scaled)

    # 예측 결과 표시
    input_data['예측 관중'] = predictions
    st.write("예측 결과:")
    st.write(input_data[['예측 관중']])  # 원하는 컬럼과 함께 표시
