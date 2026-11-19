# Q-FraudGuard

Q-FraudGuard chấm điểm giao dịch trên CPU rồi dùng máy tính lượng tử (QPU) để chọn các hồ sơ cần điều tra trước. **Bước chọn hồ sơ đã chạy trên chip siêu dẫn Quafu Haituo. Trên 44/44 cửa sổ thử nghiệm gồm 8–18 ứng viên từ dữ liệu tổng hợp IBM AMLworld, chip đo được phương án có cùng giá trị tối ưu với CPU.** [Kết quả đo trên phần cứng](docs/TRIEN_KHAI_QPU.md)

Trong một phiên chạy qua dashboard, hệ thống đọc 64 giao dịch, gom 18 cảnh báo và lưu 4 hồ sơ ưu tiên vào hàng đợi điều tra. Hai trong bốn hồ sơ đó khác với lựa chọn nếu chỉ xếp theo điểm rủi ro. CPU duyệt đủ 3.060 tổ hợp và xác nhận giá trị của phương án chip đo được. [Bản ghi phiên chạy](artifacts/deploy/quafu_dashboard_18_20261118T220101Z.json)

## So sánh QPU với CPU

Mỗi cửa sổ là một nhóm cảnh báo cần xét cùng lúc. Bài toán tối ưu nhị phân bậc hai (QUBO) chọn đúng K hồ sơ để tối đa hóa tổng điểm rủi ro và phần thưởng cho các hồ sơ chung tài khoản. CPU tính nghiệm chính xác bằng duyệt hết và đối chiếu bằng quy hoạch nguyên hỗn hợp (MILP). QPU đạt cùng chất lượng khi ít nhất một mẫu đo trực tiếp trên chip chọn đúng K và có cùng giá trị mục tiêu, trước bước hậu xử lý trên CPU.

| Cửa sổ AMLworld | QPU đạt tối ưu | CPU chọn K điểm cao nhất | CPU chọn theo điểm rồi đổi chỗ |
|---|---:|---:|---:|
| 40 cửa sổ có liên kết, chọn trước khi chạy | 40/40 | 40/40 | 40/40 |
| 4 cửa sổ chọn riêng vì xếp theo điểm bỏ lỡ nghiệm tối ưu | 4/4 | 0/4 | 4/4 |

Chip đạt tối ưu ngay lô đầu ở cả 40 cửa sổ chọn trước. Bốn cửa sổ thử thách lần lượt cần 1, 1, 7 và 2 lô ở 8, 12, 15 và 18 qubit. Mỗi lô có 1.024 lần đo. Tổng cộng, 51 tác vụ đo trên các cửa sổ này trả về 52.224 mẫu. [Bản ghi 40 cửa sổ](artifacts/deploy/quafu_verified_hits_8_12_15_18_20261118T212255Z.json) · [Bản ghi 4 cửa sổ thử thách](artifacts/deploy/quafu_ranking_misses_8_12_15_18_20261118T212601Z.json)

Ở 18 qubit, trên 10 cửa sổ chọn trước, trung vị thời gian từ lúc bắt đầu xử lý đến khi máy khách xác minh nghiệm QPU là **2,83 giây**. CPU lập bài toán và duyệt hết mất **7,73 mili giây** ở trung vị. Mạch dùng một qubit cho mỗi ứng viên. Mục tiêu tiếp theo là thử 24 và 32 ứng viên trên phần cứng phù hợp, giữ cùng giới hạn số lần đo và so sánh thời gian ở cùng chất lượng nghiệm. [Thời gian theo kích thước và thiết kế phép thử tiếp theo](docs/TRIEN_KHAI_QPU.md)

## Kiểm chứng kết quả trên máy cá nhân

Từ thư mục gốc, chạy một lệnh với Python 3.11 trở lên. Chương trình dùng thư viện chuẩn, đọc điểm ứng viên và tần suất từng chuỗi bit trong các báo cáo đã lưu, rồi tự duyệt mọi cách chọn đúng K. Có thể chạy ngoại tuyến mà không cần tài khoản Quafu.

```powershell
python scripts/verify_quafu_evidence.py
```

Hai dòng kết quả cho biết số cửa sổ đạt tối ưu, kết quả của từng cách chọn trên CPU và số mẫu tối ưu trong mỗi tập. [Mã kiểm chứng](scripts/verify_quafu_evidence.py)

## Chạy dashboard

Mẫu và mô hình PaySim đi kèm đủ để chạy dashboard trên Windows. Các lệnh dưới dùng PowerShell với Python 3.13. Chế độ `replay` ghi vị trí đã xử lý vào SQLite để lần mở sau tiếp tục cùng luồng.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
$env:QFG_DATASET = "paysim"
$env:QFG_SOURCE = "replay"
$env:QFG_PRIORITY_MODE = "risk"
python -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8765
```

Mở `http://localhost:8765`. Để chạy trên chip, chuẩn bị luồng AMLworld, môi trường Quafu và token theo hướng dẫn, rồi dùng lệnh dưới đây. Script kiểm tra backend trên hai cửa sổ trước khi mở dashboard với kích thước cửa sổ mục tiêu là 18 ứng viên. [Chuẩn bị dữ liệu AMLworld](data/DATASETS.md) · [Chạy trên Quafu](docs/TRIEN_KHAI_QPU.md)

```powershell
.\scripts\start_qpu_dashboard.ps1 -Solver quafu-fast-qpu -WindowSize 18
```

## Từ giao dịch đến hàng đợi điều tra

| Bước | Kết quả trên tập kiểm tra IBM AMLworld HI-Small |
|---|---|
| **Sàng lọc trên CPU** | Trong 1.015.448 giao dịch, mô hình gắn cờ 129.766 giao dịch (12,8%) và giữ được 1.089/1.143 giao dịch rửa tiền (95,3%). [Kết quả theo ngưỡng sàng lọc](artifacts/training/amlworld_full_operating_points.json) |
| **Chuyển chuyên viên** | Ngưỡng điều tra được chọn để chuyển 2% số giao dịch trên tập xác thực. Khi áp dụng cho tập kiểm tra, 31.407 hồ sơ qua ngưỡng, gồm 925 giao dịch rửa tiền. [Ngưỡng của mô hình](artifacts/models/amlworld_full_scorer.review.json) |

Hai ngưỡng trên thuộc tầng chấm điểm. Ở bước ưu tiên, `QFG_PRIORITY_MODE=risk` xếp hồ sơ theo điểm, còn `QFG_PRIORITY_MODE=qubo` gửi từng cửa sổ ứng viên cho bộ giải ở luồng nền. Dashboard ghi bộ giải thực sự đã trả kết quả. Chuyên viên có thể xem cả ứng viên dưới ngưỡng điều tra và ghi quyết định cho từng hồ sơ. [Luồng sự kiện và hàng đợi](src/qf_guard/operations.py)

### Nhận sự kiện JSONL

Sau khi đã chuẩn bị mô hình AMLworld, có thể nhập giao dịch theo thứ tự thời gian. Mỗi dòng chứa `transaction_id`, `timestamp`, `sender_id`, `receiver_id` và `amount`. Gửi lại cùng mã với cùng nội dung chỉ ghi một giao dịch. Khi đổi mô hình hoặc ngưỡng, dùng tên luồng mới.

```powershell
$env:QFG_DATASET = "amlworld_full"
$env:QFG_SOURCE = "live"
$env:QFG_STREAM = "pilot-aml-v1"
$env:QFG_MODEL_PATH = "artifacts/models/amlworld_full_scorer.joblib"
python scripts/stream.py ingest --input events.jsonl
python scripts/stream.py decisions --after 0 --limit 100
```

Dashboard và tiến trình nhập dùng chung `QFG_STREAM` và `QFG_CASE_DB`. SQLite mặc định nằm ở `artifacts/operations.sqlite3`. Nhật ký quyết định lưu người xử lý, hành động và thời gian UTC. Hệ thống nghiệp vụ có thể đọc tiếp từ ID cuối cùng đã nhận. [Cấu hình](.env.example)

## Tài liệu và mã nguồn

- Kết quả huấn luyện trên mẫu AMLworld bốn ngày ghi lại cách chọn mô hình chấm điểm trên CPU. [Báo cáo huấn luyện](artifacts/training/amlworld_training_report.md)
- Hàm mục tiêu và chứng nhận MILP nằm trong [prioritization.py](src/qf_guard/prioritization.py). Mạch và ánh xạ qubit nằm trong [quafu.py](src/qf_guard/quafu.py).
- Nguồn của bộ dữ liệu và điều khoản phân phối bản trích nằm trong [danh mục giấy phép](data/licenses/LICENSES.md).

Để chạy toàn bộ bộ kiểm thử trong môi trường đã kích hoạt, cài thêm Qiskit cho mô phỏng Quafu và D-Wave Ocean cho mô phỏng annealer. Các kiểm thử chạy cục bộ.

```powershell
python -m pip install -e ".[dev,quantum]" qiskit==2.4.2
python -m pytest -q
```
