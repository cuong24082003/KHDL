import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv("Housing.csv")

categorical_cols = [
    'mainroad', 'guestroom', 'basement', 'hotwaterheating',
    'airconditioning', 'prefarea', 'furnishingstatus'
]
numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

X = df[categorical_cols + numerical_cols]
y = df['price']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical_cols)
], remainder='passthrough')

lr_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

st.title("Dự đoán giá nhà (Đơn vị: VND)")
st.markdown("Nhập thông tin về căn nhà để dự đoán giá")

area = st.number_input("Diện tích (m2)", min_value=100, max_value=10000, value=5000, step=100)
bedrooms = st.number_input("Số phòng ngủ", min_value=1, max_value=10, value=3, step=1)
bathrooms = st.number_input("Số phòng tắm", min_value=1, max_value=10, value=2, step=1)
stories = st.number_input("Số tầng", min_value=1, max_value=5, value=2, step=1)
parking = st.number_input("Chỗ đậu xe", min_value=0, max_value=5, value=1, step=1)

mainroad = st.selectbox("Mặt đường chính?", ['yes', 'no'])
guestroom = st.selectbox("Phòng khách riêng?", ['yes', 'no'])
basement = st.selectbox("Tầng hầm?", ['yes', 'no'])
hotwaterheating = st.selectbox("Hệ thống nước nóng?", ['yes', 'no'])
airconditioning = st.selectbox("Điều hòa?", ['yes', 'no'])
prefarea = st.selectbox("Khu vực ưu tiên?", ['yes', 'no'])
furnishingstatus = st.selectbox("Tình trạng nội thất", ['furnished', 'semi-furnished', 'unfurnished'])

input_df = pd.DataFrame([{
    'mainroad': mainroad,
    'guestroom': guestroom,
    'basement': basement,
    'hotwaterheating': hotwaterheating,
    'airconditioning': airconditioning,
    'prefarea': prefarea,
    'furnishingstatus': furnishingstatus,
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'parking': parking
}])

if st.button("Dự đoán giá"):
    pred_lr = lr_model.predict(input_df)[0]
    pred_rf = rf_model.predict(input_df)[0]

    st.success(f"Giá dự đoán (Linear Regression): {int(pred_lr):,} VND")
    st.success(f"Giá dự đoán (Random Forest): {int(pred_rf):,} VND")

    st.subheader("Biểu đồ phân phối giá nhà")
    fig, ax = plt.subplots()
    sns.histplot(df['price'], kde=True, ax=ax, color='skyblue')
    ax.set_xlabel('Giá nhà (VND)')
    st.pyplot(fig)

    st.subheader("Giá nhà theo diện tích")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x='area', y='price', data=df, ax=ax2, alpha=0.6)
    ax2.set_xlabel('Diện tích (m2)')
    ax2.set_ylabel('Giá nhà (VND)')
    st.pyplot(fig2)
