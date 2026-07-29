# Q-FraudGuard

Q-FraudGuard là prototype hỗ trợ phát hiện, ưu tiên và xử lý giao dịch gian lận
theo luồng thời gian thực. Pipeline chính kết hợp AI để lọc ứng viên, TGNN để
phân tích quan hệ giao dịch theo thời gian, sau đó mô hình hóa nhóm ứng viên
thành QUBO và giải bằng Simulated Quantum Annealing.

## Ý tưởng hệ thống

Mỗi giao dịch đi qua ba tầng xử lý:

```text
Transaction stream
    → AI scoring và candidate filtering
    → TGNN temporal relational scoring
    → QUBO candidate grouping
    → OpenJij SQA
    → Investigation queue
    → Operator action
```

### 1. AI scoring và lọc ứng viên

Dữ liệu đầu vào được kiểm tra schema, chuẩn hóa và biến đổi thành đặc trưng.
Với giao dịch `i`, risk score được tổng hợp theo:

```text
s_i = αp_i + βa_i + γg_i + δr_i
```

Trong đó:

- `p_i`: điểm supervised từ LightGBM.
- `a_i`: điểm bất thường từ Isolation Forest.
- `g_i`: điểm liên quan trên graph giao dịch.
- `r_i`: điểm từ rule engine.
- `α, β, γ, δ`: trọng số lựa chọn trên validation set.

LightGBM, Isolation Forest, calibrator, trọng số fusion và ngưỡng ứng viên được
huấn luyện offline. Dashboard chỉ tải model artifact và chạy inference.

### 2. TGNN và graph thời gian

TGNN biểu diễn tài khoản và giao dịch trên graph có timestamp. Khi chấm điểm
giao dịch tại thời điểm `t`, graph chỉ sử dụng lịch sử quan sát trước `t`.

Đầu ra graph gồm điểm liên quan, embedding giao dịch và trọng số giữa các ứng
viên. Các tín hiệu này được kết hợp với điểm AI để mô tả mức rủi ro và quan hệ
giữa các giao dịch trong cùng cửa sổ xử lý.

### 3. QUBO và SQA

Các ứng viên được đưa vào một bài toán tối ưu nhị phân:

```text
x_i = 1  nếu giao dịch i được chọn vào nhóm ưu tiên
x_i = 0  nếu giao dịch i không thuộc nhóm ưu tiên
```

Hàm mục tiêu QUBO kết hợp risk score, quan hệ graph và giới hạn số hồ sơ cần ưu
tiên. Prototype giải QUBO bằng OpenJij Simulated Quantum Annealing trên CPU,
không sử dụng QPU hoặc API D-Wave.

Cột `SQA chọn` thể hiện giao dịch được solver chọn vào nhóm ưu tiên trong cửa
sổ hiện tại. Mọi ứng viên vẫn nằm trong hàng đợi xử lý dù có được SQA chọn hay
không.

## Luồng demo PaySim

Dashboard sử dụng phần test của PaySim để mô phỏng transaction stream:

```text
PaySim
    → đọc batch tiếp theo
    → tạo timestamp và mã giao dịch mới
    → inference
    → lọc ứng viên
    → tối ưu QUBO/SQA
    → cập nhật bảng và panel cảnh báo
```

Khi đọc hết dữ liệu, con trỏ quay lại đầu phần test để tiếp tục mô phỏng. Việc
lặp lại chỉ tạo luồng demo trong bộ nhớ và không ghi thay đổi vào file PaySim.

## Giao diện vận hành

Dashboard hiển thị:

- Số giao dịch đã xử lý.
- Số giao dịch vượt ngưỡng ứng viên.
- Tỷ lệ fraud thực tế được phát hiện.
- Latency inference trung bình.
- Bảng giao dịch realtime với risk score, trạng thái ứng viên, kết quả SQA và
  nhãn thật.

Panel bên phải cập nhật cùng nhịp với bảng chính và sắp xếp cảnh báo chưa xử lý
theo mức ưu tiên. Người vận hành có thể chọn hành động được khuyến nghị hoặc
một hành động khác. Sau khi thực thi thành công, cảnh báo được xóa khỏi hàng đợi
và giao diện hiển thị xác nhận.

## Dữ liệu và model runtime

```text
data/paysim.csv
artifacts/models/paysim_lightgbm_scorer.joblib
artifacts/models/paysim_realtime_metrics.json
```

Thông tin về PaySim, Elliptic, Bank Account Fraud Suite và IBM AMLworld được lưu
trong `data/DATASETS.md`.

## Cấu trúc project

```text
Q-FraudGuard/
├── dashboard/
│   └── app.py
├── src/
│   └── qf_guard/
├── data/
│   ├── paysim.csv
│   └── DATASETS.md
├── artifacts/
│   └── models/
├── .streamlit/
│   └── config.toml
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Chạy local

Yêu cầu Python 3.11 trở lên.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
Copy-Item .env.example .env
```

Nạp các biến trong `.env` vào phiên PowerShell:

```powershell
Get-Content .env |
  Where-Object { $_ -and -not $_.StartsWith("#") } |
  ForEach-Object {
    $name, $value = $_.Split("=", 2)
    Set-Item -Path "Env:$name" -Value $value
  }
```

Khởi động dashboard:

```powershell
python -m streamlit run dashboard/app.py `
  --server.address $env:STREAMLIT_SERVER_ADDRESS `
  --server.port $env:STREAMLIT_SERVER_PORT `
  --server.headless $env:STREAMLIT_SERVER_HEADLESS
```

Truy cập `http://localhost:8765`.
