# Triển khai và kiểm chứng trên Quafu Haituo

Trên chip siêu dẫn Haituo của Quafu-SQC, Q-FraudGuard đã đo được nghiệm tối ưu cho **44/44 cửa sổ cảnh báo** của IBM AMLworld tổng hợp. CPU chấm điểm giao dịch và lập mạch cho từng cửa sổ. QPU chạy mạch và trả tần suất các phương án đã đo. CPU dùng kết quả đó để chọn hồ sơ và kiểm chứng giá trị tối ưu. Một phiên dashboard riêng đã ghi các hồ sơ được chọn vào hàng đợi điều tra. [Bản ghi dashboard](../artifacts/deploy/quafu_dashboard_18_20261118T220101Z.json)

## So sánh trên cùng một bài toán

Mỗi cửa sổ có N ứng viên, mỗi ứng viên ứng với một qubit. Hệ thống chọn đúng `K = max(1, round(0.25 × N))` hồ sơ. Điểm rủi ro được chia cho điểm cao nhất trong cửa sổ. Mỗi cặp hồ sơ có chung ít nhất một tài khoản gửi hoặc nhận được cộng 0,005 nếu cả hai cùng được chọn. Với đúng K hồ sơ, năng lượng QUBO là `E = −(tổng điểm hồ sơ + tổng phần thưởng liên kết)`. Phương án có năng lượng thấp nhất là nghiệm tối ưu.

Backend `quafu-fast-qpu` khởi tạo mạch một lớp theo điểm rủi ro. Mạch giữ tương tác giữa các cặp qubit lân cận và xấp xỉ các tương tác còn lại quanh trạng thái khởi tạo. Mỗi lô trả về tần suất các chuỗi bit sau 1.024 lần đo. Máy khách chấm từng mẫu theo **hàm QUBO đầy đủ**. CPU duyệt mọi cách chọn đúng K và đối chiếu với chứng nhận MILP. Một cửa sổ đạt nếu chip đo được ít nhất một mẫu khả thi có năng lượng bằng nghiệm tối ưu trong sai số `10⁻⁹ × max(1, |E*|)`. Các phương án đồng hạng đều được tính. [Mạch Quafu](../src/qf_guard/quafu.py) · [Mã benchmark](../scripts/quafu_first_hit.py)

### 40 cửa sổ có liên kết, chọn trước khi giải

Benchmark lấy các giao dịch có điểm từ `0.00016` trở lên trong tập kiểm tra, xếp theo thời gian rồi chia thành nhóm liên tiếp. Ở từng kích thước, script chọn trước 10 nhóm rời nhau có ít nhất một cặp chung tài khoản, với hạt giống ngẫu nhiên (seed) `20261120`. Cả 40 cửa sổ đều có mẫu tối ưu ngay ở lô đầu. Hai đối chứng CPU, xếp theo điểm và xếp theo điểm rồi đổi chỗ, cũng đạt tối ưu trên cả 40. [Cách tạo cửa sổ](../scripts/qpu_check.py) · [Bản ghi gốc](../artifacts/deploy/quafu_verified_hits_8_12_15_18_20261118T212255Z.json)

| N qubit | K hồ sơ | Mẫu tối ưu / tổng phép đo | Máy khách xác minh nghiệm QPU | CPU duyệt hết | CPU xếp theo điểm rồi đổi chỗ |
|---:|---:|---:|---:|---:|---:|
| 8 | 2 | 2.234 / 10.240 | 2,63 s | 0,43 ms | 0,29 ms |
| 12 | 3 | 1.020 / 10.240 | 2,99 s | 0,85 ms | 0,31 ms |
| 15 | 4 | 235 / 10.240 | 3,78 s | 3,44 ms | 0,36 ms |
| 18 | 4 | 287 / 10.240 | 2,83 s | 7,73 ms | 0,48 ms |

Các cột thời gian đều là trung vị của 10 cửa sổ ở từng kích thước, bắt đầu khi ứng viên đã có điểm. Cột QPU dùng `time_to_verified_hit_s`, gồm chuẩn bị, gửi việc, chờ dịch vụ, nhận trọn lô kết quả và kiểm chứng trên CPU. Hai cột CPU dùng `cpu.exact.seconds` và `cpu.top_k_swap.seconds`, mỗi giá trị trên một cửa sổ là trung vị của bảy lượt. Ở N = 18, phương pháp duyệt hết xét đủ 3.060 cách chọn 4 hồ sơ.

Gộp 40 cửa sổ chọn trước, thời gian máy khách xác minh nghiệm QPU có trung vị **3,00 s** và phân vị 95 (P95) **6,42 s**. P95 ở đây là quan sát thứ 38 khi xếp 40 thời gian tăng dần. [Số đo từng cửa sổ](../artifacts/deploy/quafu_verified_hits_8_12_15_18_20261118T212255Z.json)

### 4 cửa sổ mà xếp hạng theo điểm bỏ lỡ nghiệm tối ưu

Tập thử thách dùng seed `20261121`. Ở mỗi kích thước, CPU duyệt các cách chọn đúng K và giữ một cửa sổ mà K hồ sơ có điểm cao nhất chưa đạt tối ưu QUBO. CPU xây mạch từ điểm và liên kết của cửa sổ rồi gửi mạch lên chip. Nghiệm đối chứng được dùng để chọn cửa sổ thử thách và chấm các mẫu do chip trả về. [Bản ghi gốc của bốn cửa sổ](../artifacts/deploy/quafu_ranking_misses_8_12_15_18_20261118T212601Z.json)

| N qubit | Lượt đến khi có nghiệm | Mẫu tối ưu / tổng phép đo |
|---:|---:|---:|
| 8 | 1 | 7 / 1.024 |
| 12 | 1 | 6 / 1.024 |
| 15 | 7 | 1 / 7.168 |
| 18 | 2 | 37 / 2.048 |

Chip và CPU có bước đổi chỗ đều đạt 4/4. Nếu chỉ tính lô đầu tiên trong các bản ghi này, QPU đạt 2/4. Nếu tính tối đa ba lô, QPU đạt 3/4. Cửa sổ 15 qubit cần đến lô thứ bảy. Đây là lý do phải công bố cả điều kiện dừng khi so sánh chất lượng nghiệm và thời gian.

### Hai lần chạy phần cứng khác

- **Dashboard 18 qubit.** Hệ thống đọc 64 giao dịch, gom 18 cảnh báo và chọn 4 hồ sơ. Phương án chip đo được giữ hai hồ sơ trong nhóm bốn điểm cao nhất và chọn thêm hai hồ sơ khác. CPU duyệt 3.060 cách chọn và xác nhận cùng giá trị QUBO `−1,067874098782923`. Máy khách nhận lô có nghiệm sau 29,07 s. Trong 8.192 lần đo có một mẫu tối ưu, thuộc tác vụ `2611190600591042404`. [Bản ghi dashboard](../artifacts/deploy/quafu_dashboard_18_20261118T220101Z.json)
- **Ủ lượng tử số hóa.** Một mạch 8 qubit, 3 bước Trotter, dùng toàn bộ hệ số QUBO trên cùng chip. Trong 2.048 lần đo, 225 mẫu chọn đúng 2 hồ sơ và 15 mẫu đạt tối ưu MILP. Mã tác vụ `2611190440279930041`. Hai lần chạy ở mục này được báo cáo riêng ngoài benchmark 44 cửa sổ. [Bản ghi mạch ủ số](../artifacts/deploy/qpu_check_quafu-anneal-qpu_20261118T204034Z.json)

## Kiểm chứng ngoại tuyến

```powershell
python scripts/verify_quafu_evidence.py
```

Lệnh đọc hai báo cáo gốc và 559 ứng viên xuất hiện trong benchmark, khôi phục giá trị từng cách chọn đúng K, đối chiếu các phương án CPU và đếm lại các mẫu tối ưu trong từng tác vụ. Kết quả lần lượt là 3.776/40.960 và 51/11.264 phép đo đạt tối ưu. Lệnh dùng thư viện chuẩn Python và có thể chạy độc lập với mô hình huấn luyện, ZIP AMLworld và dịch vụ Quafu. [Dữ liệu đầu vào](../data/quafu_benchmark_candidates.csv) · [Mã kiểm chứng](../scripts/verify_quafu_evidence.py)

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

Lệnh mở dashboard kiểm tra backend trên hai cửa sổ rồi mở giao diện ở chế độ QUBO. `-WindowSize 18` đặt mục tiêu gom 18 ứng viên mỗi lần. Khi hết thời gian chờ hoặc phát hết dữ liệu, dashboard có thể gửi một cửa sổ nhỏ hơn. Trong chế độ Quafu nhanh, giao diện hiển thị số mẫu tối ưu và thời gian máy khách nhận lô đạt. Ước lượng lấy mẫu dùng chu kỳ giả định `QFG_QUAFU_ASSUMED_SHOT_PERIOD_US`. Bản ghi Haituo cung cấp tần suất theo lô, nên đơn vị thời gian quan sát được là cả lô.

| Backend | Đường chạy |
|---|---|
| `quafu-fast-qpu` / `quafu-fast-mock` | Mạch một lớp khởi tạo theo rủi ro trên chip / mô phỏng cục bộ |
| `quafu-qpu` / `quafu-mock` | QAOA với bộ trộn XY trên chip / mô phỏng cục bộ |
| `quafu-anneal-qpu` / `quafu-anneal-mock` | Ủ số hóa trên chip / mô phỏng cục bộ |
| `exact` / `sqa` | Duyệt hết hoặc mô phỏng ủ lượng tử trên CPU |
| `dwave-qpu` / `dwave-hybrid` / `dwave-mock` | Annealer qua Leap / dịch vụ hybrid CQM / kiểm tra tích hợp cục bộ |

Để chạy tích hợp D-Wave, cài `python -m pip install -r requirements-quantum.txt`. Lệnh `python scripts/qpu_check.py --solver dwave-mock --windows 1` chạy mô phỏng. Với token `DWAVE_API_TOKEN`, dùng `python scripts/quantum_check.py --no-run` để liệt kê các solver khả dụng trước khi gửi một cửa sổ. Kết quả phần cứng nêu ở trên đều thuộc về Quafu Haituo.

## Mở rộng cửa sổ và giảm thời gian

Với ánh xạ một qubit cho mỗi ứng viên, phần cứng có nhiều qubit khả dụng hơn cho phép đưa nhiều hồ sơ vào cùng một lần tối ưu. Bảng dùng cùng quy tắc chọn K để so sánh số tổ hợp. Hai dòng cuối là kích thước đề xuất cho lần đo tiếp theo.

| Ứng viên / qubit | K hồ sơ | Số cách chọn đúng K | Trạng thái |
|---:|---:|---:|---|
| 18 | 4 | 3.060 | Đã đo trên Haituo |
| 24 | 6 | 134.596 | Đề xuất chạy trên chip lớn hơn |
| 32 | 8 | 10.518.300 | Đề xuất chạy trên chip lớn hơn |

Backend nhanh mặc định giới hạn ở 18 qubit. Để thử 24 hoặc 32, cần có đủ qubit kết nối, hiệu chuẩn phù hợp với mạch và đặt `QFG_QUAFU_FAST_MAX_QUBITS` theo kích thước được hỗ trợ. Bố trí qubit có thể ảnh hưởng đến số cổng hai qubit sau biên dịch. Còn thời gian người dùng chờ phụ thuộc cả vào số lô cần gửi, hàng đợi và thời gian trả kết quả. [Cấu hình bộ giải](../src/qf_guard/solvers.py) · [Ánh xạ mạch lên phần cứng](https://quantum.cloud.ibm.com/docs/en/qiskit/transpile)

Chọn trước cùng cửa sổ cho CPU và QPU, với tối đa một hoặc ba lô 1.024 lần đo mỗi cửa sổ. Tính tỷ lệ đạt tối ưu trên toàn bộ cửa sổ, kể cả cửa sổ dùng hết ngân sách. Ghi độ lệch mục tiêu khi có nghiệm tham chiếu, cùng trung vị và phân vị 95 (P95) của thời gian đến lúc trả kết quả. So sánh với CPU xếp theo điểm rồi đổi chỗ và với CPU giải chính xác ở các kích thước kiểm chứng được. Như vậy có thể xác định liệu máy lớn hơn rút ngắn thời gian ở cùng chất lượng nghiệm.
