# Q-FraudGuard

Q-FraudGuard chấm điểm giao dịch trên CPU và hỗ trợ chọn hồ sơ điều tra bằng CPU hoặc máy tính lượng tử (QPU). **Bước chọn hồ sơ đã chạy trên chip siêu dẫn Quafu Haituo. Trên 44/44 cửa sổ thử nghiệm gồm 8–18 ứng viên từ dữ liệu tổng hợp IBM AMLworld, chip đo được phương án có cùng giá trị tối ưu với CPU.** [Kết quả đo trên phần cứng](docs/TRIEN_KHAI_QPU.md)

Trong một phiên chạy qua dashboard, hệ thống đọc 64 giao dịch, gom 18 cảnh báo và lưu 4 hồ sơ ưu tiên vào hàng đợi điều tra. Hai hồ sơ khác với lựa chọn theo điểm rủi ro. Theo nhãn AMLworld, cả hai cách chọn đều tìm được 0 giao dịch rửa tiền trong 4 hồ sơ. CPU duyệt đủ 3.060 tổ hợp và xác nhận giá trị QUBO của phương án chip đo được. [Bản ghi phiên chạy](artifacts/deploy/quafu_dashboard_18_20261118T220101Z.json)

## So sánh QPU với CPU

Mỗi cửa sổ là một nhóm cảnh báo cần xét cùng lúc. Bài toán tối ưu nhị phân bậc hai (QUBO) chọn đúng K hồ sơ để tối đa hóa tổng điểm rủi ro và phần thưởng cho các hồ sơ chung tài khoản. CPU tính nghiệm chính xác bằng duyệt hết và đối chiếu bằng quy hoạch nguyên hỗn hợp (MILP). QPU đạt cùng chất lượng khi ít nhất một mẫu đo trực tiếp trên chip chọn đúng K và có cùng giá trị mục tiêu, trước bước hậu xử lý trên CPU.

| Cửa sổ AMLworld | QPU đạt tối ưu | CPU chọn K điểm cao nhất | CPU chọn theo điểm rồi đổi chỗ |
|---|---:|---:|---:|
| 40 cửa sổ có liên kết, chọn trước khi chạy | 40/40 | 40/40 | 40/40 |
| 4 cửa sổ chọn riêng vì xếp theo điểm bỏ lỡ nghiệm tối ưu | 4/4 | 0/4 | 4/4 |

Chip đạt tối ưu ngay lô đầu ở cả 40 cửa sổ chọn trước. Bốn cửa sổ thử thách lần lượt cần 1, 1, 7 và 2 lô ở 8, 12, 15 và 18 qubit. Mỗi lô có 1.024 lần đo. Tổng cộng, 51 tác vụ đo trên các cửa sổ này trả về 52.224 mẫu. [Bản ghi 40 cửa sổ](artifacts/deploy/quafu_verified_hits_8_12_15_18_20261118T212255Z.json) · [Bản ghi 4 cửa sổ thử thách](artifacts/deploy/quafu_ranking_misses_8_12_15_18_20261118T212601Z.json)

Ở 18 qubit, trên 10 cửa sổ chọn trước, máy khách mất **2,83 giây** ở trung vị để nhận và xác minh nghiệm QPU. CPU lập bài toán và duyệt hết mất **7,73 mili giây** ở trung vị. Thời gian qua dịch vụ QPU dài gấp khoảng **366 lần** theo tỷ số hai trung vị. Riêng khoảng từ lúc bắt đầu gửi tác vụ đến khi nhận tần suất mẫu đã chiếm **2,78 giây** ở trung vị. Bản ghi gộp truyền dữ liệu, chờ dịch vụ và xử lý tác vụ nên chưa xác định được riêng thời gian tải lên hoặc thời gian trên chip. [Các mốc thời gian trên Haituo](docs/TRIEN_KHAI_QPU.md)

**CPU là lựa chọn thực tế cho bước chọn hồ sơ hiện tại.** Heuristic CPU chọn 10 trong 40 cảnh báo trong **1,64 ms** ở trung vị. Bài toán này có ánh xạ hợp lệ dùng 160 qubit vật lý trên Zephyr Z12 lý tưởng. Nếu một QPU tương lai trả kết quả tốt ngang heuristic trong **5 ms từ lúc gửi đến lúc nhận**, ở nhóm 160 cảnh báo nó sẽ tiết kiệm khoảng **4,5 ms** so với trung vị CPU 9,53 ms. [So sánh CPU và QPU ở 40, 80 và 160 cảnh báo](docs/UOC_TINH_THOI_GIAN_DWAVE.md)

## Năm điểm còn hạn chế của prototype

1. **Hàm mục tiêu gần với xếp hạng theo điểm.** Benchmark và dashboard ở chế độ QUBO mặc định chia điểm cho điểm cao nhất cửa sổ và dùng `connectivity_strength=0.01`. Mỗi cặp chung tài khoản đóng góp 0,005. CPU xếp theo điểm rồi đổi chỗ đạt tối ưu trên 44/44 cửa sổ, với trung vị 0,35 ms ở tập 40 và 0,52 ms ở tập 4. Thư viện khi gọi trực tiếp mặc định dùng điểm thô và hệ số 0,25. Tính lại trên CPU cho thấy top-K thuần đạt tối ưu trên 40/40 cửa sổ chọn trước ở cấu hình benchmark, 31/40 khi chỉ tăng hệ số lên 0,25 và 2/40 ở mặc định thư viện. Kết quả trên chip chỉ bao phủ cấu hình đầu. [Hàm mục tiêu](src/qf_guard/prioritization.py) · [Cấu hình dashboard](dashboard/app.py) · [Phép kiểm lại](scripts/verify_quafu_evidence.py)
2. **Chưa thấy lợi ích về phát hiện rửa tiền.** Benchmark chấm theo giá trị QUBO. Khi đối chiếu sau chạy với nhãn AMLworld, QPU và top-K chọn đúng cùng các hồ sơ trên 40 cửa sổ và đều tìm được **3 giao dịch rửa tiền trong 130 lượt hồ sơ được chọn**. Ở bốn cửa sổ thử thách, hai cách đều đạt **0/13** vì toàn bộ ứng viên trong bốn cửa sổ đều mang nhãn âm. Các số này mô tả dữ liệu tổng hợp đã chạy. Hiệu quả điều tra tại ngân hàng vẫn cần đo. [Dữ liệu và cách đối chiếu nhãn](data/DATASETS.md)
3. **CPU có thể mô phỏng hiệu quả mạch đã benchmark.** Ở lượt đầu, mạch `local` ưu tiên các bit của top-K với xác suất khởi tạo 0,9 cho từng bit được ưu tiên. Cổng hai qubit chỉ nối các cặp rời nhau. Khi đã có tham số mạch, CPU chỉ cần O(N) phép tính để dựng các phân phối của từng cặp hoặc sinh một mẫu từ mạch lý tưởng. Phiên dashboard cần 29,07 giây để nhận lô chứa mẫu tối ưu duy nhất trong 8.192 lần đo. Số tổ hợp tăng theo cấp số nhân khi K xấp xỉ N/4. Cấu trúc mạch này và số đo hiện có chưa cho cơ sở kết luận QPU nhanh hơn CPU theo cấp số nhân. [Cấu trúc mạch và phạm vi phép đo](docs/TRIEN_KHAI_QPU.md)
4. **Cảnh báo sai tạo tải lớn cho chuyên viên.** Với ngưỡng nhắm giữ 95% giao dịch rửa tiền trên tập xác thực, tập kiểm tra có 120.645 cảnh báo sai và precision 0,89%. Ngưỡng chuyển chuyên viên nhắm 2% lưu lượng trên tập xác thực đạt precision 2,95% trên tập kiểm tra. Chọn 4 trong 18 hồ sơ diễn ra sau khi các cảnh báo này đã phát sinh. [Số liệu theo ngưỡng](#từ-giao-dịch-đến-hàng-đợi-điều-tra)
5. **Hai thí nghiệm dùng hai tập kiểm tra.** Mô hình được chọn trong báo cáo bốn ngày đạt PR-AUC **0,518** trên 275.911 giao dịch. Mô hình của luồng `amlworld_full` đạt **0,434** trên 1.015.448 giao dịch kiểm tra của bộ mười ngày. Chênh lệch giữa hai số không đo được mức cải thiện của mô hình. Thư mục mô hình cục bộ có sáu tệp `.joblib`. Dashboard mặc định nạp `amlworld_full_scorer.joblib` nếu chưa đặt biến môi trường. [Mô hình, cấu hình và tập dữ liệu tương ứng](data/DATASETS.md#mô-hình-nào-được-dùng)

## Kiểm chứng kết quả trên máy cá nhân

Từ thư mục gốc, chạy một lệnh với Python 3.11 trở lên. Chương trình dùng thư viện chuẩn, đọc điểm ứng viên và tần suất từng chuỗi bit trong các báo cáo đã lưu, rồi tự duyệt mọi cách chọn đúng K. Có thể chạy ngoại tuyến mà không cần tài khoản Quafu.

```powershell
python scripts/verify_quafu_evidence.py
```

Với mỗi tập, chương trình in số cửa sổ và số mẫu đạt tối ưu, số giao dịch rửa tiền trong lựa chọn của QPU và top-K, cùng số cửa sổ top-K còn đạt tối ưu khi tăng hệ số liên kết trên CPU. [Mã kiểm chứng](scripts/verify_quafu_evidence.py)

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

Mỗi dòng áp dụng một ngưỡng của luồng `amlworld_full` lên cùng **1.015.448 giao dịch kiểm tra**, gồm **1.143** giao dịch rửa tiền. Recall là tỷ lệ giao dịch rửa tiền được giữ lại. Precision là tỷ lệ giao dịch thực sự rửa tiền trong số hồ sơ qua ngưỡng. [Các ngưỡng sàng lọc](artifacts/training/amlworld_full_operating_points.json) · [Ngưỡng chuyển chuyên viên](artifacts/models/amlworld_full_scorer.review.json)

| Mục tiêu đặt trên tập xác thực | Hồ sơ qua ngưỡng trên tập kiểm tra | Rửa tiền tìm thấy | Cảnh báo sai | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| Sàng lọc, nhắm recall 95% | 121.728 (11,99%) | 1.083 | 120.645 | 94,75% | 0,89% |
| Sàng lọc hiện dùng, nhắm recall 96% | 129.766 (12,78%) | 1.089 | 128.677 | 95,28% | 0,84% |
| Chuyển chuyên viên, nhắm 2% lưu lượng | 31.407 (3,09%) | 925 | 30.482 | 80,93% | 2,95% |

Ở bước ưu tiên sau sàng lọc, `QFG_PRIORITY_MODE=risk` xếp hồ sơ theo điểm, còn `QFG_PRIORITY_MODE=qubo` gửi từng cửa sổ ứng viên cho bộ giải ở luồng nền. Dashboard ghi bộ giải thực sự đã trả kết quả. Chuyên viên có thể xem cả ứng viên dưới ngưỡng điều tra và ghi quyết định cho từng hồ sơ. [Luồng sự kiện và hàng đợi](src/qf_guard/operations.py)

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
