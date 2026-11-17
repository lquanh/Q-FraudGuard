# Q-FraudGuard

Q-FraudGuard là prototype phát hiện, ưu tiên và xử lý giao dịch gian lận theo
luồng thời gian thực. AI chấm điểm và lọc ứng viên; các ứng viên trong cùng cửa
sổ được mô hình hóa thành bài toán QUBO "chọn K hồ sơ điều tra trước", giải bằng
Simulated Quantum Annealing trên CPU **hoặc trên máy lượng tử annealing D-Wave
thật** qua Leap.

```text
Transaction stream
    → AI scoring (LightGBM + Isolation Forest + graph features + rules)
    → Candidate filtering (ngưỡng học trên validation)
    → Cửa sổ ứng viên → QUBO exact-K
    → Solver: SQA (CPU) | D-Wave QPU | D-Wave Hybrid CQM   (fallback tự động)
    → Investigation queue → Operator action
```

## 1. AI scoring và lọc ứng viên

```text
s_i = α·p_i + β·a_i + γ·g_i + δ·r_i
```

- `p_i`: xác suất LightGBM (huấn luyện với trọng số cân bằng lớp, hiệu chỉnh
  bằng logistic calibration trên validation).
- `a_i`: điểm bất thường Isolation Forest, chuẩn hóa theo phân vị 5%–98% của train.
- `g_i`: điểm graph nhân quả: fan-in, fan-out, thiết bị dùng chung, dòng tiền
  đi qua, tần suất — chỉ dùng lịch sử trước thời điểm `t`.
- `r_i`: rule engine (số tiền vượt phân vị 98,5%, rút cạn số dư, lệch số dư…).
- `α, β, γ, δ`: grid search bước 0,05 trên validation, tối đa hóa tỷ lệ giữ
  fraud ở mức 5% ứng viên, phá hòa bằng PR-AUC.

> **Lưu ý:** phần graph là *đặc trưng đồ thị theo thời gian* được thiết kế thủ
> công, **không phải TGNN** (chưa có mạng nơ-ron đồ thị trong project).

## 2. QUBO chọn hồ sơ ưu tiên

Với `N` ứng viên trong một cửa sổ, `x_i = 1` nếu giao dịch `i` được ưu tiên:

```text
min  −Σ r_i x_i − c·Σ w_ij x_i x_j + P·(Σ x_i − K)²,   K = round(0,25·N)
```

- `w_ij = 0,5` khi hai ứng viên dùng chung tài khoản gửi/nhận, `c = 0,25`.
- `P` được chọn **chặt**: `P > max_i (r_i + c·Σ_j w_ij)`. Điều kiện này đủ để mọi
  nghiệm tối ưu toàn cục đều thỏa `Σx = K` (có chứng minh trong
  `qubo.exact_k_penalty_bound` và kiểm thử brute-force). P nhỏ giúp QPU — vốn
  chuẩn hóa hệ số về dải analog hẹp — vẫn phân biệt được chênh lệch risk.
- Ràng buộc exact-K làm mọi cặp biến liên kết với nhau (đồ thị đầy đủ), nên trên
  QPU dùng `DWaveCliqueSampler` với clique embedding (tối đa ~177 biến trên
  Advantage).
- Hậu xử lý chung cho mọi backend: chọn mẫu khả thi có năng lượng thấp nhất,
  sửa mẫu lệch K (nếu không mẫu nào khả thi), rồi local search hoán đổi cặp giữ
  nguyên K. Năng lượng **trước** hậu xử lý được lưu trong `raw_objective` để
  đánh giá riêng chất lượng phần cứng.

## 3. Backend giải QUBO

| `QFG_SOLVER` | Chạy ở đâu | Ghi chú |
|---|---|---|
| `sqa` (mặc định) | OpenJij SQA, CPU cục bộ | Không cần mạng |
| `exact` | Duyệt toàn bộ C(N,K), CPU | Ground truth cho benchmark |
| `dwave-qpu` | **QPU D-Wave thật** (Advantage/Advantage2) | Cần `DWAVE_API_TOKEN` |
| `dwave-hybrid` | Leap Hybrid CQM (cổ điển + QPU) | Ràng buộc K dạng native, ≥ 5 s/job |
| `dwave-mock` | `MockDWaveSampler` offline | Chạy đúng code path QPU, **không phải phần cứng lượng tử** |

Với `dwave-qpu`/`dwave-hybrid`, dashboard tự chuyển về SQA khi lỗi mạng, sai
token, quá kích thước clique hoặc khi vượt `QFG_QPU_BUDGET_SECONDS` (mặc định
10 s thời gian truy cập QPU mỗi tiến trình). Thẻ **Bộ giải QUBO** hiển thị
backend thực sự đã dùng, thời gian QPU, ngân sách đã tiêu và lý do fallback.

Dashboard gom ứng viên thành cửa sổ (`QFG_QUANTUM_MIN_BATCH`=8 hoặc sau
`QFG_QUANTUM_WINDOW_SECONDS`=10 s, tối đa 40) và giải trong luồng nền, nên độ trễ
mạng của QPU không làm chậm nhịp 1 giây của giao diện. Cột `Ưu tiên QUBO` hiển
thị trạng thái: chờ batch → đang giải → ưu tiên / không chọn. Cảnh báo được QUBO
chọn xếp trước trong cùng mức hành động.

## 4. Triển khai lên D-Wave

1. Tạo tài khoản tại <https://cloud.dwavesys.com/leap>. Tài khoản mới có gói
   trial giới hạn thời gian QPU (kiểm tra điều khoản hiện hành trên Leap).
2. Cài SDK: `python -m pip install -r requirements-quantum.txt`
3. Lấy API token trong Leap dashboard, đặt vào `.env`:
   ```text
   QFG_SOLVER=dwave-qpu
   DWAVE_API_TOKEN=<token của bạn>
   ```
4. Kiểm tra kết nối và chạy một bài QUBO 8 biến (~10–30 ms QPU):
   ```powershell
   python scripts/quantum_check.py --no-run   # chỉ liệt kê solver
   python scripts/quantum_check.py            # submit 1 job lên QPU
   ```
5. Benchmark trên cửa sổ ứng viên PaySim thật (mỗi cửa sổ ~1 lần gọi QPU):
   ```powershell
   python scripts/quantum_benchmark.py --solvers exact,sqa,dwave-qpu --windows 10
   ```
   Báo cáo JSON lưu trong `artifacts/benchmarks/`, gồm `problem_id` của từng job
   (xem lại trên D-Wave Inspector / Leap).
6. Chạy dashboard như mục 7; thẻ **Bộ giải QUBO** phải hiển thị `D-Wave QPU` và
   tên chip.

Chạy thử toàn bộ đường đi QPU mà không cần token: `QFG_SOLVER=dwave-mock`.

## 5. Kết quả hiện tại và giới hạn

Benchmark 20 cửa sổ × 16 ứng viên từ tập test PaySim (`exact,sqa,dwave-mock`):

| Solver | Tối ưu sau hậu xử lý | Tối ưu thô (trước local search) | Thời gian TB |
|---|---|---|---|
| exact | 100% | 100% | 4 ms |
| sqa | 100% | 10% | 68 ms |
| dwave-mock | 100% | 0% | 19 ms |

- **Trên PaySim, QUBO không tạo khác biệt so với chọn top-K theo risk**: các ứng
  viên gần như không bao giờ dùng chung tài khoản (0 cạnh trong cả 20 cửa sổ),
  nên bài toán suy biến thành sắp xếp. Phần cứng lượng tử ở đây chứng minh khả
  năng triển khai, không mang lại lợi thế về chất lượng hay tốc độ. Để QUBO có ý
  nghĩa cần dữ liệu có cấu trúc quan hệ (ví dụ IBM AMLworld, Elliptic).
- Annealer (SQA, SA, QPU) lật từng bit nên khó di chuyển giữa các nghiệm cùng K;
  tỷ lệ tối ưu thô thấp là đặc điểm của bài toán, local search khắc phục điều đó.
- Số liệu `dwave-mock` là mô phỏng; chất lượng trên QPU thật (nhiễu, chain
  break) cần đo bằng `quantum_benchmark.py` với token thật.

Model hiện tại (test 19.884 giao dịch): PR-AUC 0,876, ROC-AUC 0,968, bắt 59/61
fraud, precision 6,9% ở 4,3% ứng viên.

## 6. Huấn luyện lại model

```powershell
python scripts/train.py --output artifacts/models/paysim_lightgbm_scorer.joblib
```

Chia theo thời gian 60/20/20, ghi kèm `paysim_lightgbm_metrics.json`. Lần chạy
kiểm chứng: PR-AUC 0,881, ROC-AUC 0,978, bắt 59/61 fraud, ~9 giây.

## 7. Chạy local

Yêu cầu Python 3.11 trở lên.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-quantum.txt
python -m pip install --no-deps -e .
Copy-Item .env.example .env
```

Nạp biến môi trường và khởi động dashboard:

```powershell
Get-Content .env |
  Where-Object { $_ -and -not $_.StartsWith("#") } |
  ForEach-Object {
    $name, $value = $_.Split("=", 2)
    Set-Item -Path "Env:$name" -Value $value
  }

python -m streamlit run dashboard/app.py `
  --server.address $env:STREAMLIT_SERVER_ADDRESS `
  --server.port $env:STREAMLIT_SERVER_PORT `
  --server.headless $env:STREAMLIT_SERVER_HEADLESS
```

Truy cập `http://localhost:8765`. Kiểm thử: `python -m pytest`.

## 8. Cấu trúc project

```text
Q-FraudGuard/
├── dashboard/app.py              # Streamlit: stream, cảnh báo, cửa sổ QUBO nền
├── src/qf_guard/
│   ├── data.py, features.py      # chuẩn hóa, đặc trưng nhân quả + graph
│   ├── scoring.py, training.py   # RiskScorer (fit/score), huấn luyện + đánh giá
│   ├── qubo.py                   # QUBOModel, builder, penalty chặt, to_bqm
│   ├── prioritization.py         # ứng viên → cạnh → QUBO → lựa chọn
│   ├── solvers.py                # exact, SQA, D-Wave QPU/hybrid/mock, fallback
│   ├── postprocess.py            # chọn mẫu, sửa exact-K, swap local search
│   └── explainability.py         # mã lý do, khuyến nghị xử lý
├── scripts/                      # train.py, quantum_check.py, quantum_benchmark.py
├── tests/                        # pytest
├── data/                         # paysim.csv, DATASETS.md
└── artifacts/models/             # model đã huấn luyện + metrics
```

Thông tin về PaySim, Elliptic, Bank Account Fraud Suite và IBM AMLworld được lưu
trong `data/DATASETS.md`.
