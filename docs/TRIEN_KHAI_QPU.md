# Triển khai và kiểm chứng trên Quafu Haituo

Q-FraudGuard đã chạy bước chọn hồ sơ trên chip siêu dẫn **Quafu Haituo**. Trong **44/44 cửa sổ** từ dữ liệu tổng hợp IBM AMLworld, chip đo được ít nhất một danh sách chọn đúng số hồ sơ và có giá trị QUBO bằng nghiệm tối ưu do CPU xác nhận. Các cửa sổ gồm 8, 12, 15 hoặc 18 cảnh báo. [Bản ghi 40 cửa sổ chọn trước](../artifacts/deploy/quafu_verified_hits_8_12_15_18_20261118T212255Z.json) · [Bản ghi 4 cửa sổ thử thách](../artifacts/deploy/quafu_ranking_misses_8_12_15_18_20261118T212601Z.json)

## Kết quả trên cùng một bài toán

Mỗi cảnh báo ứng với một qubit. Hệ thống chọn đúng `K = max(1, round(0.25 × N))` hồ sơ. Điểm rủi ro được chia cho điểm cao nhất trong cửa sổ. Với cấu hình đã chạy (`risk_normalization=max`, `connectivity_strength=0.01`), mỗi cặp hồ sơ được chọn có chung tài khoản được cộng 0,005 điểm. CPU dùng toàn bộ hàm QUBO để chấm các chuỗi bit do chip trả về. Một mẫu đạt khi chọn đúng K và có năng lượng bằng nghiệm tối ưu trong sai số `10⁻⁹ × max(1, |E*|)`. CPU đối chiếu bằng duyệt hết và quy hoạch nguyên hỗn hợp (MILP). [Hàm chọn hồ sơ](../src/qf_guard/prioritization.py) · [Mã kiểm chứng trên cùng cửa sổ](../scripts/quafu_first_hit.py)

Backend `quafu-fast-qpu` dùng mạch `local` một lớp với cổng hai qubit trên các cặp rời nhau. Ở lượt thử đầu, các bit thuộc Top-K được khởi tạo với xác suất 0,9 cho trạng thái 1. Mỗi lô có **1.024 lần đo**. Tập chọn trước gồm 10 cửa sổ ở mỗi kích thước, lấy từ các giao dịch có điểm từ `0.00016` và có ít nhất một cặp chung tài khoản (seed `20261120`). Tập thử thách gồm một cửa sổ ở mỗi kích thước mà xếp theo điểm đơn thuần bỏ lỡ nghiệm tối ưu (seed `20261121`). Benchmark dừng từng cửa sổ khi nhận được lô chứa mẫu tối ưu đầu tiên. Khi gọi trực tiếp `PriorityConfig()`, thư viện dùng điểm thô và hệ số liên kết 0,25. [Mạch Quafu](../src/qf_guard/quafu.py) · [Cách tạo cửa sổ](../scripts/qpu_check.py) · [Cấu hình thư viện](../src/qf_guard/prioritization.py)

| Cảnh báo / qubit | K hồ sơ | Cửa sổ chọn trước đạt tối ưu ngay lô đầu | Lô có mẫu tối ưu đầu tiên ở cửa sổ thử thách |
|---:|---:|---:|---:|
| 8 | 2 | 10/10 | 1 |
| 12 | 3 | 10/10 | 1 |
| 15 | 4 | 10/10 | 7 |
| 18 | 4 | 10/10 | 2 |

Hai báo cáo ghi **51 lô, 52.224 lần đo**. Trên tập chọn trước, 3.776/40.960 mẫu đạt tối ưu. Trên bốn cửa sổ thử thách, tỷ lệ là 51/11.264 mẫu. CPU xếp theo điểm rồi đổi chỗ cũng đạt tối ưu ở cả 44 cửa sổ. Mốc 44/44 xác nhận chất lượng mẫu tốt nhất trên từng cửa sổ với hệ số 0,01. Chất lượng điều tra được đối chiếu riêng theo nhãn AMLworld cùng số suất K. [Bản ghi chọn trước](../artifacts/deploy/quafu_verified_hits_8_12_15_18_20261118T212255Z.json) · [Bản ghi tập thử thách](../artifacts/deploy/quafu_ranking_misses_8_12_15_18_20261118T212601Z.json) · [Đối chiếu nhãn](../data/DATASETS.md#kiểm-lại-44-cửa-sổ-đã-chạy-trên-chip)

Bản ghi của mỗi tác vụ cung cấp cấu hình mạch, mã tác vụ và số lần xuất hiện của từng chuỗi bit. Có thể dùng các trường này để kiểm tra số hồ sơ được chọn và giá trị QUBO. Để lập kế hoạch thời gian cho phương án D-Wave, dùng số lượt đọc dự kiến và thông số của máy tham chiếu. [Giả định và bảng ước tính D-Wave](UOC_TINH_THOI_GIAN_DWAVE.md)

## Hai phiên triển khai đã lưu kết quả

- **Dashboard 18 qubit.** Hệ thống đọc **64 giao dịch**, gom **18 cảnh báo** và lưu **4 hồ sơ chip chọn**. Trong Bàn điều tra, một hồ sơ đạt ngưỡng và ba hồ sơ còn lại xuất hiện ở mục đề xuất bổ sung. CPU duyệt đủ 3.060 cách chọn và xác nhận danh sách chip đo được có giá trị QUBO tối ưu `−1,067874098782923`. Bản ghi lưu chuỗi bit gốc, thứ tự qubit và bốn mã giao dịch. [Bản ghi dashboard](../artifacts/deploy/quafu_dashboard_18_20261118T220101Z.json)
- **Ủ lượng tử số hóa.** Mạch 8 qubit với 3 bước Trotter dùng toàn bộ hệ số QUBO. Trong 2.048 lần đo, 15 mẫu đạt nghiệm tối ưu do MILP xác nhận. Mã tác vụ là `2611190440279930041`. [Bản ghi mạch ủ số](../artifacts/deploy/qpu_check_quafu-anneal-qpu_20261118T204034Z.json)

## Kiểm chứng ngoại tuyến

Từ thư mục gốc, chạy lệnh sau để đọc hai báo cáo và 559 ứng viên đã lưu. Chương trình dùng thư viện chuẩn Python để khôi phục từng bài QUBO, duyệt các cách chọn đúng K và đếm lại số cửa sổ có mẫu tối ưu. Hai tập cho kết quả **40/40** và **4/4**. [Dữ liệu ứng viên](../data/quafu_benchmark_candidates.csv) · [Mã kiểm chứng](../scripts/verify_quafu_evidence.py)

```powershell
python scripts/verify_quafu_evidence.py
```

<details>
<summary>Đối chiếu chuỗi bit với 4 hồ sơ của phiên dashboard</summary>

Bản ghi dashboard lưu cả chuỗi bit do nhà cung cấp trả về và thứ tự ứng viên trên mạch. Đoạn Python dưới chạy từ thư mục gốc, áp dụng chiều đọc của chip rồi so sánh trực tiếp bốn mã giao dịch với danh sách đã ghi vào hàng đợi. [Bản ghi phiên 18 qubit](../artifacts/deploy/quafu_dashboard_18_20261118T220101Z.json)

```python
import json
from pathlib import Path

record = json.loads(Path("artifacts/deploy/quafu_dashboard_18_20261118T220101Z.json").read_bytes())
audit = record["audit"]
bits = audit["winning_provider_bitstring"]
if audit["reverse_provider_bits"]:
    bits = bits[::-1]
order = audit["winning_circuit_candidate_order"]
selected = [audit["candidate_ids"][i] for bit, i in zip(bits, order, strict=True) if bit == "1"]
assert len(selected) == record["dashboard"]["k"] and sorted(selected) == sorted(record["selected_ids"])
print(*sorted(selected), sep="\n")
```

</details>

## Chạy trên Quafu

Các lệnh dưới đây chạy từ thư mục gốc dự án trên Windows. Môi trường đã chạy benchmark dùng Qiskit 2.4.2, quarkstudio 7.3.9 và quarkcircuit 0.5.13. Đặt token trong biến `QPU_API_TOKEN` hoặc trong file `.env` cục bộ. [Quafu-SQC](https://quafu-sqc.baqis.ac.cn)

```powershell
python -m venv .venv-quafu
.\.venv-quafu\Scripts\python.exe -m pip install -e . qiskit==2.4.2 quarkstudio==7.3.9 quarkcircuit==0.5.13
```

Cả benchmark và bước kiểm tra trước khi mở dashboard đọc `artifacts/training/full_scores_test.pkl`. Trên máy mới, tạo luồng và điểm từ ZIP IBM AMLworld bằng một lần huấn luyện. Có thể chỉ định nguồn điểm khác cho benchmark bằng `--scores`. [Vị trí dữ liệu nguồn](../data/DATASETS.md)

```powershell
.\.venv-quafu\Scripts\python.exe scripts/train_amlworld_full.py --target-recall 0.96 --stream-output data/amlworld_hi_small_full_stream.csv --save-test-scores
```

Để đo với ngân sách cố định, lệnh đầu gửi tối đa một lô 1.024 lần đo cho mỗi cửa sổ. Một phiên mới có thể gửi thêm mạch hiệu chuẩn thứ tự bit. Hai lệnh tiếp theo tái tạo thiết kế của hai báo cáo với `--until-hit` và lưu kết quả vào tệp mới. Để tiếp tục một lần chạy dở, dùng lại cùng tham số và đường dẫn `--report`. Các tác vụ phần cứng sử dụng hạn mức của tài khoản.

```powershell
.\.venv-quafu\Scripts\python.exe scripts/quafu_first_hit.py --sizes 8 12 15 18 --windows 1 --max-jobs 1 --report artifacts/deploy/fixed_budget_demo.json
.\.venv-quafu\Scripts\python.exe scripts/quafu_first_hit.py --sizes 8 12 15 18 --windows 10 --seed 20261120 --until-hit --report artifacts/deploy/replay_40.json
.\.venv-quafu\Scripts\python.exe scripts/quafu_first_hit.py --sizes 8 12 15 18 --windows 1 --seed 20261121 --where-ranking-misses --until-hit --report artifacts/deploy/replay_challenge.json
.\scripts\start_qpu_dashboard.ps1 -Solver quafu-fast-qpu -WindowSize 18
```

Lệnh mở dashboard kiểm tra backend trên hai cửa sổ rồi mở giao diện ở chế độ QUBO. `-WindowSize 18` đặt mục tiêu gom 18 ứng viên mỗi lần. Khi hết thời gian chờ hoặc phát hết dữ liệu, dashboard có thể gửi một cửa sổ nhỏ hơn. [Script khởi chạy dashboard](../scripts/start_qpu_dashboard.ps1)

Trên host đang chạy luồng phát lặp, người vận hành dùng cùng `QFG_STREAM` và `QFG_CASE_DB` với worker. Lệnh `python scripts/stream.py backend` đọc cấu hình. Lệnh `python scripts/stream.py set-backend quafu-fast-qpu` chọn Quafu cho cửa sổ kế tiếp sau khi kiểm tra môi trường. Dùng `set-backend exact` để chọn lại CPU hoặc `set-backend dwave-qpu` khi đã cấu hình Leap. Sổ vận hành lưu người đổi và thời điểm đổi. [CLI quản trị](../scripts/stream.py) · [Cấu hình phát lặp](../README.md#phát-lặp-toàn-bộ-amlworld)

| Backend | Đường chạy |
|---|---|
| `quafu-fast-qpu` / `quafu-fast-mock` | Mạch một lớp khởi tạo theo rủi ro trên chip / mô phỏng cục bộ |
| `quafu-qpu` / `quafu-mock` | QAOA với bộ trộn XY trên chip / mô phỏng cục bộ |
| `quafu-anneal-qpu` / `quafu-anneal-mock` | Ủ số hóa trên chip / mô phỏng cục bộ |
| `exact` / `sqa` | Duyệt hết hoặc mô phỏng ủ lượng tử trên CPU |
| `dwave-qpu` / `dwave-hybrid` / `dwave-mock` | Annealer qua Leap / dịch vụ hybrid CQM / kiểm tra tích hợp cục bộ |

## Mở rộng cửa sổ

Với backend Quafu đang dùng, mỗi ứng viên cần một qubit. Bảng cho biết số danh sách khả thi khi chọn đúng K.

| Ứng viên / qubit | K hồ sơ | Số cách chọn đúng K | Trạng thái |
|---:|---:|---:|---|
| 18 | 4 | 3.060 | Đã đo trên Haituo |
| 24 | 6 | 134.596 | Đề xuất chạy trên chip lớn hơn |
| 32 | 8 | 10.518.300 | Đề xuất chạy trên chip lớn hơn |

Để thử cửa sổ 24 hoặc 32 trên Quafu, chọn chip có đủ qubit kết nối và đặt `QFG_QUAFU_FAST_MAX_QUBITS` theo kích thước được hỗ trợ. [Cấu hình bộ giải Quafu](../src/qf_guard/solvers.py)

Với D-Wave, cài các phụ thuộc bằng `python -m pip install -r requirements-quantum.txt`. Dùng `python scripts/qpu_check.py --solver dwave-mock --windows 1` để kiểm tra tích hợp cục bộ. Khi đã cấu hình `DWAVE_API_TOKEN`, lệnh `python scripts/quantum_check.py --no-run` liệt kê các solver tài khoản có thể truy cập. Báo cáo D-Wave trình bày riêng các ánh xạ trên Zephyr Z12 lý tưởng và ước tính thời gian lấy mẫu theo số lượt đọc cùng một bài toán. [Ước tính thời gian D-Wave theo số lượt đọc](UOC_TINH_THOI_GIAN_DWAVE.md)
