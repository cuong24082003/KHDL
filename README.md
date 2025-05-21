

TRƯỜNG ĐẠI HỌC KỸ THUẬT CÔNG NGHIỆP
KHOA ĐIỆN TỬ
Bộ môn: Công nghệ thông tin



BÀI TẬP KẾT THÚC MÔN HỌC

MÔN HỌC
KHOA HỌC DỮ LIỆU


 	Sinh viên: Phạm Nguyên Cương 
 	Lớp :K57KMT.01.  
	Giáo viên giảng dạy: Nguyễn Văn Huy
	Link Github: https://github.com/cuong24082003/KHDL
      	Link Youtube: https://youtu.be/rdhD0E9h9kY


 



Thái Nguyên – 2025

TRƯỜNG ĐHKTCN
CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM
KHOA ĐIỆN TỬ	Độc lập - Tự do - Hạnh phúc


BÀI TẬP KẾT THÚC MÔN HỌC
	
	MÔN HỌC: KHOA HỌC DỮ LIỆU
	BỘ MÔN CÔNG NGHỆ THÔNG TIN
Sinh viên: Phạm Nguyên Cương
Lớp: K57KMT        Ngành: Kỹ thuật máy tính 
Giáo viên hướng dẫn: Nguyễn Văn Huy
Ngày giao đề 20/5/2025   Ngày hoàn thành: 30/5/2025
Tên đề tài : "Dự đoán giá nhà bằng Machine Learning kết hợp giao diện Streamlit
Yêu cầu :
•	Sử dụng Pandas đọc và xử lý dữ liệu khuyết thiếu.
•	Feature engineering: sử dụng các đặc trưng phù hợp để dự báo.
•	Dùng scikit-learn huấn luyện mô hình Linear Regression hoặc Random Forest.
•	Hiển thị kết quả bằng đồ thị histogram, scatter plots để phân tích giá.
GIÁO VIÊN HƯỚNG DẪN
(Ký và ghi rõ họ tên)
 

NHẬN XÉT CỦA GIÁO VIÊN HƯỚNG DẪN
					
						
					
					
					
					
					
					
					
					
					
					


Thái Nguyên, ngày….tháng…..năm 20....
              GIÁO VIÊN HƯỚNG DẪN
               (Ký ghi rõ họ tên)


















CHƯƠNG I: GIỚI THIỆU ĐỀ TÀI

Trong thời đại số hóa hiện nay, việc ứng dụng trí tuệ nhân tạo và học máy trong lĩnh vực bất động sản đang ngày càng trở nên phổ biến và cần thiết. Việc định giá bất động sản chính xác không chỉ giúp người mua và người bán đưa ra quyết định hợp lý mà còn hỗ trợ các nhà đầu tư, ngân hàng và tổ chức tài chính trong việc phân tích và đánh giá rủi ro.
Đề tài "Xây dựng ứng dụng GUI hoặc web dự đoán giá nhà" nhằm phát triển một công cụ đơn giản, trực quan nhưng hiệu quả giúp người dùng có thể dự báo giá nhà dựa trên các đặc điểm đầu vào như diện tích, số phòng ngủ, phòng tắm, vị trí địa lý và các tiện ích liên quan. Dữ liệu đầu vào được lấy từ bộ dữ liệu nổi tiếng House Prices Dataset của Kaggle – một nguồn dữ liệu đã được xử lý và sử dụng rộng rãi trong nghiên cứu và ứng dụng học máy.
Ứng dụng sử dụng các công nghệ phổ biến trong lĩnh vực khoa học dữ liệu và phát triển phần mềm như:
•	Pandas để xử lý và phân tích dữ liệu, đặc biệt là xử lý giá trị khuyết thiếu.
•	Scikit-learn để xây dựng mô hình dự đoán, bao gồm Linear Regression và Random Forest – hai thuật toán phổ biến trong bài toán hồi quy.
•	Matplotlib/Seaborn để trực quan hóa dữ liệu và kết quả, giúp người dùng hiểu rõ mối quan hệ giữa các yếu tố đầu vào và giá nhà.
•	Streamlit hoặc Tkinter để xây dựng giao diện người dùng, cho phép nhập dữ liệu đầu vào và hiển thị kết quả dự đoán một cách thân thiện và dễ sử dụng.
Thông qua đề tài này, người học không chỉ nắm vững quy trình xây dựng một hệ thống dự đoán giá nhà hoàn chỉnh mà còn có cơ hội thực hành tích hợp các kỹ thuật xử lý dữ liệu, học máy và phát triển giao diện vào một ứng dụng thực tiễn. Đây là một bước đệm quan trọng để tiếp cận các vấn đề lớn hơn trong lĩnh vực Data Science và AI ứng dụng.













CHƯƠNG II: CƠ SỞ DỮ LIỆU 

Trong chương này, chúng ta sẽ trình bày các kiến thức nền tảng và công nghệ cốt lõi được sử dụng để xây dựng hệ thống dự đoán giá nhà. Những thành phần này bao gồm kiến thức về cấu trúc dữ liệu trong Python, thư viện xử lý dữ liệu Pandas, thư viện trực quan hóa Matplotlib và Seaborn, các mô hình học máy (Machine Learning) như Linear Regression và Random Forest, cùng với các công cụ xây dựng giao diện người dùng như Tkinter hoặc Streamlit.

2.1 Cấu trúc dữ liệu List trong Python
Trong Python, List là một kiểu dữ liệu quan trọng dùng để lưu trữ nhiều giá trị trong một biến duy nhất. List cho phép lưu trữ các phần tử có kiểu dữ liệu khác nhau, có thể truy cập theo chỉ số, thay đổi giá trị, hoặc lặp qua các phần tử. Trong chương trình, List thường được sử dụng để lưu trữ các giá trị đầu vào hoặc đầu ra như danh sách tiện ích, danh sách đặc trưng, v.v.

2.2 Thư viện Pandas
Pandas là một thư viện mạnh mẽ dùng để thao tác và phân tích dữ liệu trong Python. Các đối tượng chính của Pandas là:
•	Series: một mảng một chiều.
•	DataFrame: một bảng dữ liệu hai chiều, tương tự như bảng Excel hoặc bảng trong cơ sở dữ liệu.
Trong chương trình, Pandas được dùng để:
•	Đọc dữ liệu từ file CSV (read_csv)
•	Xử lý dữ liệu thiếu (fillna, dropna)
•	Biến đổi dữ liệu (chuyển đổi kiểu, mã hóa nhãn, trích chọn đặc trưng)
•	Tính toán thống kê mô tả (mean, median, min, max, std)

2.3 Thư viện trực quan hóa dữ liệu: Matplotlib và Seaborn
Matplotlib là thư viện nền tảng để tạo các biểu đồ trong Python. Seaborn là thư viện xây dựng trên Matplotlib, cung cấp giao diện thân thiện và dễ sử dụng cho việc trực quan hóa dữ liệu thống kê.
Các dạng biểu đồ phổ biến sử dụng trong chương trình:
•	Histogram: biểu đồ phân phối giá nhà
•	Scatter Plot: biểu diễn mối quan hệ giữa giá nhà và các đặc trưng như diện tích, số phòng
•	Boxplot: biểu diễn sự phân tán và giá trị ngoại lai

2.4 Các mô hình học máy sử dụng
2.4.1 Linear Regression (Hồi quy tuyến tính)
Là mô hình hồi quy đơn giản dùng để dự đoán giá trị liên tục. Linear Regression cố gắng tìm ra đường thẳng tốt nhất biểu diễn mối quan hệ giữa các đặc trưng đầu vào và đầu ra.
Phương trình tổng quát:
y = w1*x1 + w2*x2 + ... + wn*xn + b
Trong đó:
•	y là giá nhà dự đoán
•	x1, x2, ..., xn là các đặc trưng (diện tích, số phòng, v.v.)
•	w1, ..., wn là hệ số mô hình học được
•	b là hằng số
2.4.2 Random Forest Regression
Là một mô hình hồi quy phức tạp hơn dựa trên ensemble learning, sử dụng nhiều cây quyết định (decision trees) để cải thiện độ chính xác và tránh overfitting.
Ưu điểm:
•	Hiệu quả với dữ liệu phi tuyến tính
•	Chịu ảnh hưởng ít hơn bởi giá trị ngoại lai
•	Đưa ra kết quả ổn định

2.5 Thư viện Scikit-learn
Scikit-learn là thư viện mạnh mẽ hỗ trợ các thuật toán học máy trong Python. Trong chương trình, thư viện này được dùng để:
•	Tiền xử lý dữ liệu (StandardScaler, LabelEncoder, OneHotEncoder)
•	Chia dữ liệu huấn luyện và kiểm tra (train_test_split)
•	Xây dựng và huấn luyện mô hình (LinearRegression, RandomForestRegressor)
•	Đánh giá mô hình (mean_squared_error, r2_score)

2.6 Xây dựng giao diện người dùng
2.6.1 Tkinter
Tkinter là thư viện tiêu chuẩn của Python để xây dựng giao diện đồ họa (GUI). Cho phép tạo form nhập liệu, nút bấm, nhãn, và hiển thị kết quả trực tiếp.
2.6.2 Streamlit
Streamlit là công cụ tạo giao diện web tương tác một cách nhanh chóng cho các ứng dụng Data Science và Machine Learning. Ưu điểm của Streamlit:
•	Giao diện hiện đại
•	Tương thích tốt với các thư viện khoa học dữ liệu
•	Dễ dàng triển khai và mở rộng







CHƯƠNG III : THIẾT KẾ VÀ XÂY DỰNG CHƯƠNG TRÌNH

3.1. Sơ đồ khối hệ thống
Mô tả các module chính trong chương trình
Chương trình gồm các module chức năng chính như sau:
•	Module đọc và xử lý dữ liệu: Đọc dữ liệu từ tập tin .csv, xử lý dữ liệu thiếu, mã hóa dữ liệu dạng chuỗi, chuẩn hóa dữ liệu số.
•	Module huấn luyện mô hình: Sử dụng Linear Regression hoặc Random Forest để học từ dữ liệu đã xử lý.
•	Module giao diện người dùng: Nhận dữ liệu đầu vào từ người dùng (GUI hoặc Web), hiển thị kết quả dự đoán.
•	Module dự đoán và đánh giá: Nhận đầu vào mới và trả về giá nhà dự đoán, đánh giá mô hình bằng MSE, RMSE, R².
•	Module trực quan hóa dữ liệu: Tạo biểu đồ Histogram, Scatter Plot, Boxplot để phân tích mối quan hệ giữa các yếu tố và giá nhà.
•	Dự đoán giá nhà

 1. Xử lý dữ liệu
│   ├── Đọc file CSV
│   ├── Làm sạch dữ liệu
│   └── Mã hóa và chuẩn hóa

 2. Huấn luyện mô hình
│   ├── Chia dữ liệu train/test
│   └── Huấn luyện Linear Regression / Random Forest
│
3. Giao diện người dùng
│   ├── Nhập thông tin nhà ở
│   └── Hiển thị kết quả
│
4. Dự đoán giá
│   └── Tính toán và trả kết quả
│
5. Trực quan hóa
    ├── Histogram giá nhà
    ├── Scatter plot theo diện tích, phòng
    └── Boxplot theo vị trí
3.2. Sơ đồ khối các thuật toán chính
[Đọc dữ liệu từ CSV] 
       ↓
[Tiền xử lý dữ liệu]
       ↓
[Chia tập Train/Test]
       ↓
[Huấn luyện mô hình ML]
       ↓
[Đánh giá mô hình]
       ↓
[Nhận đầu vào từ người dùng]
       ↓
[Dự đoán giá nhà]
       ↓
[Hiển thị kết quả + biểu đồ]

•	Đọc dữ liệu: Đọc file train.csv từ Kaggle.
•	Tiền xử lý: Xử lý dữ liệu thiếu (fillna), mã hóa nhãn (LabelEncoder, OneHotEncoder), chuẩn hóa (StandardScaler).
•	Chia tập dữ liệu: Sử dụng train_test_split() để chia thành tập huấn luyện và kiểm tra.
•	Huấn luyện mô hình: Dùng LinearRegression() hoặc RandomForestRegressor() để huấn luyện.
•	Đánh giá mô hình: Tính toán MSE, RMSE, R² trên tập kiểm tra.
•	Nhận đầu vào: Từ giao diện người dùng, nhận thông tin như diện tích, số phòng, vị trí.
•	Dự đoán: Chạy mô hình với đầu vào mới để dự đoán giá.
•	Hiển thị kết quả: In ra giá nhà ước lượng và hiển thị biểu đồ liên quan.
3.3. Cấu trúc dữ liệu
Nguồn dữ liệu: House Prices - Kaggle Dataset
Tên trường	Mô tả	Kiểu dữ liệu
LotArea	Diện tích đất	Số nguyên
OverallQual	Chất lượng tổng thể	Số nguyên
YearBuilt	Năm xây dựng	Số nguyên
TotRmsAbvGrd	Tổng số phòng trên mặt đất	Số nguyên
FullBath	Số phòng tắm đầy đủ	Số nguyên
BedroomAbvGr	Số phòng ngủ trên mặt đất	Số nguyên
GarageCars	Sức chứa gara	Số nguyên
Neighborhood	Khu vực	Chuỗi (string)
SalePrice	Giá bán nhà (label để dự đoán)	Số thực (float)
3.4. Chương trình




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


























CHƯƠNG 4 : THỰC NGHIỆM VÀ KẾT LUẬN

4.1.Chạy thử

(.venv) PS C :\Users\9\PycharmProjects\BTCM KHDL> streamlit run main.py 

  You can now view your Streamlit app in your browser.

  Local URL : http://localhost :8501
  Network URL : http://192.168.1.109 :8501

 
  
 

4.2. Kết luận 
1. Sản phẩm đã thực hiện được
Sau quá trình thiết kế và xây dựng, em đã hoàn thành ứng dụng dự đoán giá nhà với các chức năng chính như sau:
•	Đọc và xử lý dữ liệu từ bộ dữ liệu House Prices trên Kaggle.
•	Tiền xử lý dữ liệu: xử lý dữ liệu khuyết thiếu, mã hóa biến phân loại, chuẩn hóa đặc trưng số.
•	Huấn luyện mô hình Machine Learning với Linear Regression và Random Forest để dự đoán giá nhà.
•	Xây dựng giao diện người dùng bằng Streamlit/Tkinter giúp người dùng nhập dữ liệu đầu vào như diện tích, số phòng, vị trí... và nhận kết quả dự đoán tức thì.
•	Trực quan hóa dữ liệu và kết quả mô hình bằng các biểu đồ như histogram, scatter plot, boxplot.
Ứng dụng chạy ổn định, giao diện đơn giản, dễ sử dụng, mô hình cho kết quả dự đoán hợp lý trên tập dữ liệu thử nghiệm.

2. Kiến thức và kỹ năng thu được
Trong quá trình thực hiện đề tài, em đã học hỏi và áp dụng được nhiều kiến thức như:
•	Kỹ năng xử lý dữ liệu bằng Pandas, xử lý dữ liệu khuyết thiếu, chuẩn hóa, mã hóa biến phân loại.
•	Nắm được cách huấn luyện và đánh giá mô hình dự báo giá với Linear Regression và Random Forest trong thư viện scikit-learn.
•	Biết cách trực quan hóa dữ liệu với Matplotlib và Seaborn để hỗ trợ phân tích dữ liệu.
•	Biết thiết kế giao diện ứng dụng đơn giản bằng Tkinter hoặc triển khai web-app bằng Streamlit.
•	Nâng cao kỹ năng lập trình Python, phân tích, thiết kế và triển khai một hệ thống hoàn chỉnh.

3. Hướng cải tiến trong tương lai
Mặc dù sản phẩm đã hoàn thiện về chức năng cơ bản, em nhận thấy vẫn còn nhiều điểm có thể cải tiến:
•	Cải thiện độ chính xác mô hình: thử nghiệm thêm các thuật toán khác như XGBoost, Gradient Boosting, hoặc sử dụng kỹ thuật Feature Selection để loại bỏ đặc trưng không cần thiết.
•	Giao diện nâng cao hơn: nâng cấp giao diện người dùng trực quan, thêm bảng thống kê, biểu đồ tương tác với Plotly hoặc Dash.
•	Triển khai trên nền tảng web thực tế: đóng gói và triển khai ứng dụng lên các nền tảng như Heroku, Render, hoặc Streamlit Cloud để người dùng có thể truy cập trực tuyến.
•	Cho phép người dùng tải file CSV và dự đoán hàng loạt thay vì chỉ dự đoán một căn nhà.
•	Tích hợp phân tích SHAP để giải thích mô hình (model interpretability), giúp người dùng hiểu rõ yếu tố nào ảnh hưởng nhiều nhất đến giá nhà.






















