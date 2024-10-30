import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# 저장된 모델 로드
with open("rf_model.pkl", "rb") as file:
    rf_model = pickle.load(file)

# 범주형 변수를 학습한 인코더들 로드
with open("label_encoders.pkl", "rb") as file:
    label_encoders = pickle.load(file)

# Streamlit 앱 설정
st.title("관중수 예측 앱")
st.write("경기 정보를 입력하고 예상 관중 수를 확인하세요.")

# 사용자 입력을 위한 인터페이스
구분 = st.selectbox("구분", label_encoders['구분'].classes_)
주심 = st.selectbox("주심", label_encoders['주심'].classes_)
요일 = st.selectbox("요일", label_encoders['요일'].classes_)
날씨 = st.selectbox("날씨", label_encoders['날씨'].classes_)
경기장 = st.selectbox("경기장", label_encoders['경기장'].classes_)
상대승리_5G = st.number_input("상대팀 최근 5경기 승리 수", min_value=0, max_value=5)
상대패배_5G = st.number_input("상대팀 최근 5경기 패배 수", min_value=0, max_value=5)
상대득실_5G = st.number_input("상대팀 최근 5경기 득실", min_value=-20, max_value=20)
Bteam = st.selectbox("상대팀", label_encoders['Bteam'].classes_)
Bteam_followers = st.number_input("상대팀 팔로워 수", min_value=0)
Bteam_follower_rank = st.number_input("상대팀 팔로워 순위", min_value=0)
온도 = st.number_input("온도 (°C)")
상대_습도 = st.number_input("상대 습도 (%)")
이슬점 = st.number_input("이슬점 (°C)")
체감_온도 = st.number_input("체감 온도 (°C)")
강수량 = st.number_input("강수량 (mm)")
날씨_코드 = st.number_input("날씨 코드 (WMO 코드)", min_value=0)
구름량 = st.number_input("구름량 (%)", min_value=0, max_value=100)
풍속_10m = st.number_input("10m 풍속 (km/h)")
풍속_100m = st.number_input("100m 풍속 (km/h)")
토양온도_0_7cm = st.number_input("0~7cm 토양 온도 (°C)")
토양온도_7_28cm = st.number_input("7~28cm 토양 온도 (°C)")
일조_시간 = st.number_input("일조 시간 (초)")
best11_sum = st.number_input("선발 베스트11 합계", min_value=0)
mvp_YN = st.number_input("MVP 여부", min_value=0, max_value=1)
득실차 = st.number_input("득실차", min_value=-10, max_value=10)
관중수 = st.number_input("예측에 활용할 이전 관중수", min_value=0)
승패여부 = st.selectbox("승패 여부", label_encoders['pre_h_승패여부'].classes_)

# 입력 데이터를 DataFrame 형태로 준비
input_data = pd.DataFrame({
    '구분': [구분], '주심': [주심], '요일': [요일], '날씨': [날씨], '경기장': [경기장], '상대승리(5G)': [상대승리_5G],
    '상대패배(5G)': [상대패배_5G], '상대득실(5G)': [상대득실_5G], 'Bteam': [Bteam],
    'Bteam_followers': [Bteam_followers], 'Bteam_follower_rank': [Bteam_follower_rank], '온도 (°C)': [온도],
    '상대 습도 (%)': [상대_습도], '이슬점 (°C)': [이슬점], '체감 온도 (°C)': [체감_온도], '강수량 (mm)': [강수량],
    '날씨 코드 (WMO 코드)': [날씨_코드], '구름량 (%)': [구름량], '10m 풍속 (km/h)': [풍속_10m],
    '100m 풍속 (km/h)': [풍속_100m], '0~7cm 토양 온도 (°C)': [토양온도_0_7cm],
    '7~28cm 토양 온도 (°C)': [토양온도_7_28cm], '일조 시간 (초)': [일조_시간], 'pre_r_best11_sum': [best11_sum],
    'pre_r_mvp_YN': [mvp_YN], 'pre_h_득실차': [득실차], 'pre_h_관중수': [관중수], 'pre_h_승패여부': [승패여부]
})

# 인코딩 적용
for col in categorical_columns + ['Bteam']:
    input_data[col] = label_encoders[col].transform(input_data[col])

# 예측 버튼과 결과 표시
if st.button("관중 수 예측하기"):
    prediction = rf_model.predict(input_data)
    st.write("예측된 관중 수:", int(prediction[0]))
