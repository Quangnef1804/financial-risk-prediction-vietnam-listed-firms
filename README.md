# Financial Distress Prediction

Project này xây dựng pipeline xử lý dữ liệu tài chính doanh nghiệp niêm yết trên HNX và HOSE, tạo bộ đặc trưng tài chính, huấn luyện mô hình dự báo `target`, và đánh giá kết quả bằng các metric phân loại. Ngoài các mô hình machine learning, bước đánh giá còn có baseline theo công thức Altman Z-score proxy để so sánh.

## Tổng Quan

Mục tiêu của bài toán là dự báo doanh nghiệp có rủi ro hiệu quả sinh lời thấp ở năm tiếp theo. Biến mục tiêu `target` được tạo từ `ROS_next`:

```text
target = 1 nếu ROS_next < 0.05
target = 0 nếu ROS_next >= 0.05
```

Các feature chính:

```text
CR, ROS, DS, TAT, SIZE, is_HNX
```

Trong đó:

- `CR`: current ratio.
- `ROS`: return on sales.
- `DS`: debt-to-sales.
- `TAT`: total asset turnover.
- `SIZE`: log tổng tài sản.
- `is_HNX`: biến sàn, HNX = 1 và HOSE = 0.

Các mô hình đang dùng:

- Logistic Regression.
- Random Forest.
- XGBoost.
- Altman Z-score Proxy baseline.

## Cài Đặt

Tạo môi trường ảo và cài thư viện:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Nếu dùng file `.env`, có thể copy từ `.env.example`:

```powershell
Copy-Item .env.example .env
```

Các biến đường dẫn mặc định:

```text
RAW_HNX_FILE=Data/raw/HNX_V2.xlsx
RAW_HOSE_FILE=Data/raw/HOSE_V2.xlsx
FEATURED_DATA_FILE=Data/train/DATA_FEATURED.xlsx
SUPERVISED_DATA_FILE=Data/train/DATA_SUPERVISED.xlsx
PREPROCESSING_OUTPUT_DIR=src/output/preprocessing
MODELS_OUTPUT_DIR=src/output/models
```

## Cách Chạy Pipeline

### 1. Tổng quan dữ liệu raw

```powershell
python src/raw_data_overview.py
```

Output mặc định nằm ở:

```text
Data/raw/overview/
```

### 2. Tạo dữ liệu feature và supervised

```powershell
python src/process_financial_data.py
```

Output:

```text
Data/train/DATA_FEATURED.xlsx
Data/train/DATA_SUPERVISED.xlsx
```

Nếu muốn loại công ty tài chính khỏi `DATA_SUPERVISED`:

```powershell
python src/process_financial_data.py --exclude-finance
```

### 3. Chạy EDA và phân tích thống kê

Phân phối feature:

```powershell
python src/eda_feature_distribution.py
```

So sánh feature theo target:

```powershell
python src/analyze_features_by_target.py
```

Phân tích tương quan feature:

```powershell
python src/analyze_feature_correlations.py
```

So sánh HNX và HOSE:

```powershell
python src/analyze_exchange_difference.py
```

Các biểu đồ và file phân tích mặc định được lưu ở:

```text
src/output/
```

### 4. Preprocessing cho mô hình

```powershell
python src/preprocessing.py
```

Mặc định pipeline dùng time-based split:

```text
train: nam <= 2023
test : nam == 2024
```

Output:

```text
src/output/preprocessing/
  X_train.csv
  X_test.csv
  X_train_scaled.csv
  X_test_scaled.csv
  y_train.csv
  y_test.csv
  meta_train.csv
  meta_test.csv
  preprocessing_config.json
```

Có thể đổi năm split:

```powershell
python src/preprocessing.py --train-cutoff 2023 --test-year 2024
```

Giữ lại công ty tài chính:

```powershell
python src/preprocessing.py --keep-finance
```

### 5. Train models

Train tất cả mô hình:

```powershell
python src/train_models.py
```

Train một số mô hình cụ thể:

```powershell
python src/train_models.py --models lr rf xgb
```

Kiểm tra setup trước khi train:

```powershell
python src/train_models.py --check-only
```

Output:

```text
src/output/models/
  logistic_regression_model.pkl
  random_forest_model.pkl
  xgboost_model.pkl
  model_comparison.csv
  model_comparison.xlsx
  predictions_*.csv
```

### 6. Evaluate saved models

```powershell
python src/evaluate_models.py
```

Bước này đọc model đã train trong `src/output/models/`, đọc test set từ `src/output/preprocessing/`, sau đó tạo bảng đánh giá cuối cùng.

Output chính:

```text
src/output/models/model_evaluation.csv
src/output/models/model_evaluation.xlsx
src/output/models/evaluation_predictions_*.csv
src/output/models/evaluation_predictions_altman_z_score.csv
src/output/models/altman_z_score_components.csv
src/output/models/plots/
```

Bỏ qua audit train/test:

```powershell
python src/evaluate_models.py --skip-audit
```

Evaluate một số model cụ thể:

```powershell
python src/evaluate_models.py --models lr rf
```

Lưu ý: Altman Z-score Proxy vẫn được tính như baseline rule-based trong bước evaluate để so sánh với các model ML.

## Altman Z-score Proxy

Do dữ liệu hiện tại không có đầy đủ các thành phần gốc của Altman Z-score như retained earnings và market value equity, project dùng proxy dựa trên các feature đã tạo:

```text
TL/TA ~= DS * TAT
WC/TA ~= (CR - 1) * TL/TA
EBIT/TA ~= ROS * TAT
Book Equity/TL ~= (1 - TL/TA) / TL/TA

Z ~= 1.2*WC/TA + 3.3*EBIT/TA + 0.6*BookEquity/TL + 1.0*TAT
```

Ngưỡng distress mặc định:

```text
Z <= 1.81
```

Altman baseline được lưu cùng bảng evaluation để dễ so sánh các metric như AUC, F1, F2, precision, recall, specificity và accuracy.

## Thứ Tự Chạy Nhanh

Nếu chạy lại toàn bộ từ đầu:

```powershell
python src/raw_data_overview.py
python src/process_financial_data.py
python src/preprocessing.py
python src/train_models.py
python src/evaluate_models.py
```

Chạy thêm phân tích EDA:

```powershell
python src/eda_feature_distribution.py
python src/analyze_features_by_target.py
python src/analyze_feature_correlations.py
python src/analyze_exchange_difference.py
```
