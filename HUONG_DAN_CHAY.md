# Hướng dẫn chạy Q-FraudGuard

Chạy các lệnh bằng **PowerShell trên Windows**, từ thư mục chứa `pyproject.toml`. Để xem sản phẩm lần đầu, chỉ cần thực hiện hai mục đầu. Các mục sau dành cho người muốn phát giao dịch mới, thử API hoặc đổi bộ giải.

## Cài đặt

Dùng Python 3.11–3.13. Tạo môi trường riêng cho dự án và gọi trực tiếp Python trong môi trường đó. Các lệnh tiếp theo dùng cùng đường dẫn `.\.venv\Scripts\python.exe`.

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

## Chạy phiên minh họa

Bản ghi Haituo và dữ liệu phát lại đi kèm repo. Các biến bên dưới dùng một sổ SQLite riêng cho demo.

```powershell
$env:QFG_SOURCE = "recorded"
$env:QFG_STREAM = "demo-haituo"
$env:QFG_CASE_DB = "artifacts/demo-haituo.sqlite3"
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8765
```

Mở `http://127.0.0.1:8765/login` và dùng tài khoản demo. Sau khi đăng nhập, ứng dụng chuyển đến `/dashboard`. Trên sổ mới, **Bàn điều tra** hiển thị **1 hồ sơ chờ và 3 đề xuất bổ sung**. Mở `AML-004060379`, trỏ vào một giao dịch trong **Transaction Graph** để xem chi tiết. Tiếp nhận hồ sơ rồi mở **Nhật ký** để kiểm tra người xử lý và lý do. Nhấn `Ctrl+C` trong terminal để dừng. Mở lại với cùng tên luồng và đường dẫn SQLite để tiếp tục từ trạng thái đã lưu. [Tài khoản demo](README.md#tài-khoản-demo)

## Phát lặp toàn bộ AMLworld

Chế độ `loop` chấm điểm giao dịch khi phát, tạo hồ sơ mới và quay lại đầu nguồn khi hết dữ liệu. Chuẩn bị ZIP IBM tại `data/downloads/ibm_amlworld.zip`, bên trong có `HI-Small_Trans.csv`. Nếu máy đã có `artifacts/models/amlworld_full_scorer.joblib` cho luồng này, bỏ qua lệnh huấn luyện. [Nguồn tải và cách tạo dữ liệu](data/DATASETS.md#tái-lập-các-luồng-dữ-liệu)

```powershell
.\.venv\Scripts\python.exe scripts/train_amlworld_full.py --target-recall 0.96 --stream-output data/amlworld_hi_small_full_stream.csv --save-test-scores
```

Dừng dashboard demo, mở PowerShell mới ở thư mục dự án và đặt cấu hình cho luồng phát lặp.

```powershell
$env:QFG_SOURCE = "loop"
$env:QFG_DATASET = "amlworld_full"
$env:QFG_MODEL_PATH = "artifacts/models/amlworld_full_scorer.joblib"
$env:QFG_STREAM = "loop-hi-small-cases"
$env:QFG_CASE_DB = "artifacts/live-casework.sqlite3"
$env:QFG_PRIORITY_MODE = "qubo"
$env:QFG_SOLVER = "exact"
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8765
```

Lần đầu, ứng dụng tạo chỉ mục phát lại trong `artifacts/`. Giữ trang mở để xem giao dịch và hồ sơ được cập nhật. Để chạy liên tục cả khi đóng trình duyệt, dừng tiến trình này, mở ba terminal ở thư mục dự án và **chạy khối cấu hình luồng phát lặp phía trên trong cả ba**. Mỗi terminal sau đó chạy lệnh tương ứng.

| Terminal | Lệnh |
|---|---|
| 1 · Phát giao dịch | `.\.venv\Scripts\python.exe scripts/stream.py replay-loop` |
| 2 · Chọn hồ sơ | `.\.venv\Scripts\python.exe scripts/stream.py prioritize --window-size 18` |
| 3 · Dashboard | Đặt `$env:QFG_REPLAY_EXTERNAL = "1"` và `$env:QFG_EXTERNAL_PRIORITY = "1"` trên hai dòng, rồi chạy lại lệnh Streamlit ở trên. |

### Chuyển bước ưu tiên sang QPU

Trên máy xử lý, chuẩn bị SDK và thông tin truy cập Quafu trong **cùng môi trường Python** mà worker sử dụng. Khi worker đang chạy, mở terminal quản trị với cùng cấu hình luồng. Lệnh đầu đọc lựa chọn hiện tại. Lệnh thứ hai chuyển **các cửa sổ kế tiếp** sang phần cứng Haituo khi backend đã được cấu hình cho chip này. Để quay về CPU, thay `quafu-fast-qpu` bằng `exact`. [Cài SDK, cấu hình token và chạy trên chip](docs/TRIEN_KHAI_QPU.md#chạy-trên-quafu)

```powershell
.\.venv\Scripts\python.exe scripts/stream.py backend
.\.venv\Scripts\python.exe scripts/stream.py set-backend quafu-fast-qpu
```

## Chạy Core Banking sandbox

Để thử tích hợp trên máy cá nhân, dùng mô hình PaySim đi kèm. Dừng dashboard cũ nếu đang dùng cổng 8765. Mở **hai terminal** tại thư mục dự án và đặt các biến sau trong **cả hai** để API và dashboard nhìn thấy cùng hồ sơ.

```powershell
$env:QFG_SOURCE = "live"
$env:QFG_DATASET = "paysim"
$env:QFG_MODEL_PATH = "artifacts/models/paysim_lightgbm_scorer.joblib"
$env:QFG_CASE_DB = "artifacts/core-sandbox.sqlite3"
$env:QFG_STREAM = "core-paysim-sandbox"
$env:QFG_PRIORITY_MODE = "qubo"
$env:QFG_SOLVER = "exact"
```

**Terminal 1** chạy adapter. Các lệnh dưới tạo ba token ngẫu nhiên riêng cho phiên thử cục bộ. Khi tích hợp tại ngân hàng, cấp token qua kho bí mật cho từng ứng dụng theo quyền đọc, gửi giao dịch hoặc ghi biên nhận.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[core-banking]"
$env:QFG_CORE_INGEST_TOKEN = .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
$env:QFG_CORE_RECEIPTS_TOKEN = .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
$env:QFG_CORE_READ_TOKEN = .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
.\.venv\Scripts\python.exe -m uvicorn qf_guard.core_banking:create_app --factory --host 127.0.0.1 --port 8766 --workers 1 --no-access-log
```

**Terminal 2** kiểm tra API rồi mở dashboard. Lệnh đầu phải trả về `ready`. Hợp đồng máy đọc được tại `http://127.0.0.1:8766/openapi.json`. Hàng chờ bắt đầu có hồ sơ khi ứng dụng nguồn gửi giao dịch và giao dịch qua ngưỡng sàng lọc. [Payload giao dịch mẫu, biên nhận và nhật ký](docs/CORE_BANKING.md#hợp-đồng-http-v1)

```powershell
(Invoke-RestMethod http://127.0.0.1:8766/healthz).status
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8765
```

## Kiểm chứng kết quả

Có thể kiểm lại kết quả phần cứng ngoại tuyến bằng thư viện chuẩn Python. Script dựng lại QUBO từ dữ liệu đã lưu và đối chiếu với mẫu chip trả về. Kết quả mong đợi là **40/40** cửa sổ chọn trước và **4/4** cửa sổ thử thách đạt giá trị tối ưu. [Phương pháp kiểm chứng](docs/TRIEN_KHAI_QPU.md#kiểm-chứng-ngoại-tuyến)

```powershell
.\.venv\Scripts\python.exe scripts/verify_quafu_evidence.py
```

Để chạy toàn bộ bộ kiểm thử, cài thêm phụ thuộc cho API, TGNN và các bộ giải rồi chạy pytest.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,quantum,core-banking,tgnn]" qiskit==2.4.2
.\.venv\Scripts\python.exe -m pytest -q
```

Nếu cổng 8765 đang bận, đổi `--server.port` sang một cổng trống và mở địa chỉ tương ứng. Nếu dashboard và API cho kết quả khác nhau khi tra cùng mã giao dịch, kiểm tra `QFG_STREAM` và đường dẫn tuyệt đối của `QFG_CASE_DB` ở hai terminal. Mỗi terminal PowerShell mới cần đặt lại biến môi trường cho chế độ đang chạy. [Danh mục cấu hình](.env.example)
