"""Cấu hình tập trung: đường dẫn, seed và tên cột của dataset (BigMart Sales)."""
from pathlib import Path

# --- Đường dẫn ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "bigmart_sales.csv"  # CSV gốc tải từ Kaggle (BigMart Sales, tên gốc Train.csv), không sửa tay
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "save_models"  # thư mục lưu mô hình (.npy) + preprocessor + best_model.txt
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Tham số chung ---
SEED = 42          # seed cố định cho mọi lần chạy
TEST_SIZE = 0.2    # tỉ lệ tập test
K_FOLDS = 5        # số fold cho cross-validation

# --- Cột của dataset BigMart Sales (Kaggle) ---
TARGET_COL = "Item_Outlet_Sales"  # doanh số của sản phẩm tại outlet (biến mục tiêu)

CATEGORICAL_COLS = [
    "Item_Fat_Content",      # hàm lượng chất béo
    "Item_Type",             # loại sản phẩm
    "Outlet_Identifier",     # mã cửa hàng
    "Outlet_Size",           # cỡ cửa hàng (có giá trị thiếu)
    "Outlet_Location_Type",  # hạng vị trí cửa hàng
    "Outlet_Type",           # loại cửa hàng
]

NUMERIC_COLS = [
    "Item_Weight",              # trọng lượng sản phẩm (có giá trị thiếu)
    "Item_Visibility",          # độ hiển thị sản phẩm tại cửa hàng
    "Item_MRP",                 # giá niêm yết (Maximum Retail Price)
    "Outlet_Establishment_Year",  # năm thành lập cửa hàng
]

# --- Các cột tùy chọn: dataset này KHÔNG có → đặt None để bỏ qua ---
DATE_COL = None          # không có cột ngày → chia train/test ngẫu nhiên
HOLIDAY_COL = None       # không có cột ngày lễ
PRODUCT_ID_COL = "Item_Identifier"  # mã sản phẩm (độ phổ biến)
CUSTOMER_ID_COL = None   # không có mã khách hàng → bỏ qua đặc trưng CLV
PRICE_COL = "Item_MRP"   # đơn giá
QUANTITY_COL = None      # không có số lượng
REVIEW_COL = None        # không có số review
RATING_COL = None        # không có rating
AD_SPEND_COL = None      # không có chi phí quảng cáo
