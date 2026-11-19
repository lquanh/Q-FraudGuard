# Ước tính thời gian QPU để đạt nghiệm đủ tốt theo tham chiếu D-Wave

**Ví dụ với 32 hồ sơ cần chọn 8.** Nếu mỗi lần đọc có xác suất **1%** trả về phương án đạt ít nhất **99% giá trị tối ưu**, cần **299 lần đọc** để xác suất thu được ít nhất một nghiệm đạt mục tiêu lên **95%**. Theo mô hình thời gian D-Wave bên dưới, phần ủ thuần mất **5,98 ms**, cả chu kỳ lấy mẫu mất **61,21 ms**, và thời gian truy cập QPU gồm cả lập trình mất **76,29 ms** cho một tác vụ. Thời gian từ máy khách gửi yêu cầu đến nhận kết quả còn phụ thuộc dịch vụ và mạng. [Mô hình thời gian của D-Wave](https://docs.dwavequantum.com/en/latest/quantum_research/operation_timing.html)

Bảng này là **ước tính có điều kiện cho máy ủ lượng tử D-Wave**. Các lần chạy phần cứng của Q-FraudGuard được ghi trên chip cổng lượng tử Quafu Haituo. Xác suất thành công trên D-Wave cho các cửa sổ của dự án vẫn cần đo. [Bản ghi và cách đo trên Haituo](TRIEN_KHAI_QPU.md)

## Thế nào là một nghiệm đủ tốt?

Với một cửa sổ N ứng viên, chọn đúng K theo quy tắc của prototype. Gọi `F(x)` là tổng điểm rủi ro chuẩn hóa của các hồ sơ đã chọn cộng 0,005 cho mỗi cặp được chọn chung tài khoản. Gọi `F*` là giá trị lớn nhất với đúng K hồ sơ. Tài liệu này chọn mục tiêu **đúng K và `F(x) ≥ 0,99 × F*`**. Một lần đọc được tính sau khi giải mã chuỗi qubit, trước bước sửa nghiệm hay đổi chỗ trên CPU. Trong phép đo đề xuất, chuỗi qubit bị đứt được tính là một lần hụt. Tiêu chí 99% nói về hàm QUBO. Khả năng giữ giao dịch rửa tiền cần đo riêng bằng nhãn AML. [Hàm mục tiêu](../src/qf_guard/prioritization.py)

Gọi `p` là xác suất một lần đọc đạt mục tiêu **trên chính cửa sổ, cách nhúng và cấu hình chip đó**. Với các lần đọc độc lập, cùng xác suất `p`, cơ hội có ít nhất một nghiệm đạt sau `R` lần là `1 − (1 − p)^R`. Số lần đọc tối thiểu cho độ tin cậy 95% là `R = ceil(ln(0,05) / ln(1 − p))` với `0 < p < 1`. Nếu `p = 1`, chỉ cần một lần. Nếu `p = 0`, tăng số lần đọc vẫn không đạt mục tiêu.

| Xác suất đạt mục tiêu mỗi lần đọc, giả định | Lần đọc cần cho 95% | Tổng thời gian ủ ở 20 µs/lần |
|---:|---:|---:|
| 10% | 29 | 0,58 ms |
| 1% | 299 | 5,98 ms |
| 0,1% | 2.995 | 59,90 ms |

## Ước tính khi tăng số qubit

QUBO chọn đúng K có số hạng phạt `P(Σx − K)²`, tạo tương tác giữa mọi cặp biến. Backend D-Wave của dự án dùng `DWaveCliqueSampler`. Một biến logic có thể cần một chuỗi nhiều qubit vật lý. Cột qubit vật lý dưới đây là số qubit dùng trong **một phép nhúng tính ngoại tuyến trên đồ thị Pegasus P16 lý tưởng**, với `minorminer` và seed 42. Trên chip thật, các qubit và khớp nối đang hoạt động quyết định cách nhúng và kích thước tối đa. [Backend của dự án](../src/qf_guard/solvers.py) · [Cách D-Wave nhúng biến logic lên phần cứng](https://docs.dwavequantum.com/en/latest/quantum_research/embedding_intro.html)

Tốc độ tham chiếu lấy từ **bộ tham số lịch sử của `Advantage_system4.1`, tháng 8/2022**, do D-Wave phân phối trong Ocean. Ở chế độ ủ thuận mặc định, mô hình dùng 14.072,88 µs để lập trình cộng 1.000 µs ổn định sau lập trình, 20 µs để ủ mỗi lần và 20,54 µs chờ giữa các lần. Ocean ước tính thời gian đọc từ số qubit vật lý bằng nội suy log–log. Các giá trị này mô tả một cấu hình tham chiếu cụ thể. [Bộ tham số D-Wave trong Ocean 0.14.6](https://github.com/dwavesystems/dwave-cloud-client/blob/0.14.6/dwave/cloud/testing/mocks.py#L282) · [Hàm ước tính của Ocean](https://docs.dwavequantum.com/en/latest/ocean/api_ref_cloud/generated/dwave.cloud.solver.StructuredSolver.estimate_qpu_access_time.html)

| Hồ sơ / qubit logic | K | Qubit vật lý dùng | Chu kỳ mỗi lần đọc (µs) | Truy cập QPU nếu p = 10% (ms) | Nếu p = 1% (ms) | Nếu p = 0,1% (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 2 | 16 | 137,76 | 19,07 | 56,26 | 427,66 |
| 18 | 4 | 50 | 172,04 | 20,06 | 66,51 | 530,33 |
| 24 | 6 | 90 | 189,76 | 20,58 | 71,81 | 583,41 |
| 32 | 8 | 142 | 204,73 | 21,01 | 76,29 | 628,24 |
| 48 | 12 | 274 | 228,99 | 21,71 | 83,54 | 700,89 |
| 64 | 16 | 436 | 248,26 | 22,27 | 89,30 | 758,61 |
| 128 | 32 | 1.582 | 275,83 | 23,07 | 97,55 | 841,19 |

Mỗi ô thời gian giả định **một lần lập trình cho cả lô**. Chu kỳ lấy mẫu bằng thời gian ủ cộng thời gian đọc và chờ. Công thức của bảng là `T_truy_cập = 15,07288 ms + R × T_chu_kỳ` sau khi đổi đơn vị. Đây là đại lượng tương ứng với trường SAPI `qpu_access_time`. D-Wave ghi riêng `qpu_access_overhead_time`. Đồng hồ máy khách còn tính lập bài toán, tìm cách nhúng, truyền yêu cầu, xếp hàng, trả kết quả và hậu xử lý theo phạm vi đo. Nếu phải chia thành nhiều tác vụ, cần tính phí thời gian lập trình cho từng tác vụ. [Các trường thời gian SAPI](https://docs.dwavequantum.com/en/latest/quantum_research/operation_timing.html#sapi-timing-fields)

Ở cùng giả định `p = 1%`, tăng từ 32 lên 128 biến làm thời gian truy cập ước tính tăng từ 76,29 lên 97,55 ms. Nếu ở 128 biến `p` giảm xuống 0,1%, con số thành 841,19 ms. **Xác suất đạt chất lượng yêu cầu là đại lượng còn thiếu để dự báo tốc độ thực tế.** Trên 10 cửa sổ 18 ứng viên đã đo, CPU duyệt hết đạt đúng tối ưu trong 7,73 ms ở trung vị. Để tìm kích thước mà QPU có lợi về thời gian, cần đo hai hệ thống trên cùng cửa sổ, cùng tiêu chí chất lượng và cùng phạm vi đồng hồ. [Đối chứng CPU đã đo](../artifacts/deploy/quafu_verified_hits_8_12_15_18_20261118T212255Z.json)

## Tính lại bảng ngoại tuyến

Cài `python -m pip install -r requirements-quantum.txt` từ thư mục gốc. Bảng dùng Ocean SDK 9.4.0, `dwave-cloud-client` 0.14.6 và `minorminer` 0.2.22. Đoạn Python sau chỉ dựng đồ thị và gọi mô hình thời gian cục bộ.

```python
from math import ceil, log1p

from dwave.cloud.solver import StructuredSolver
from dwave.cloud.testing.mocks import qpu_pegasus_solver_data, qpu_problem_timing_data
from dwave.graphs import pegasus_graph
from minorminer.busclique import find_clique_embedding

graph = pegasus_graph(16)
profile = qpu_problem_timing_data("advantage")
spec = qpu_pegasus_solver_data(16)
spec["properties"]["problem_timing_data"] = profile
solver = StructuredSolver(client=None, data=spec)
program_us = profile["typical_programming_time"] + profile["default_programming_thermalization"]

for n in (8, 18, 24, 32, 48, 64, 128):
    embedding = find_clique_embedding(n, graph, seed=42, use_cache=False)
    if len(embedding) != n:
        raise ValueError(f"Không nhúng được đủ {n} biến")
    physical = len({q for chain in embedding.values() for q in chain})
    cycle_us = solver.estimate_qpu_access_time(physical, num_reads=1, annealing_time=20) - program_us
    access_ms = []
    for p in (0.10, 0.01, 0.001):
        reads = ceil(log1p(-0.95) / log1p(-p))
        access_us = solver.estimate_qpu_access_time(physical, num_reads=reads, annealing_time=20)
        access_ms.append(round(float(access_us) / 1000, 2))
    print(n, max(1, round(n / 4)), physical, round(float(cycle_us), 2), access_ms)
```

## Thay giả định bằng số đo

Chọn cửa sổ trước khi chạy và chốt ngưỡng chất lượng. Dùng nghiệm chính xác của CPU ở kích thước còn giải được. Với cửa sổ lớn, có thể dùng một cận trên đã chứng nhận cho `F*` để đặt mục tiêu bảo thủ, hoặc công bố rõ nếu ngưỡng chỉ so với nghiệm tốt nhất đã biết. Gửi số lần đọc cố định trên nhiều lô và ghi cả các lô hụt. Đếm số lần đạt trên tổng số lần đọc, tính khoảng bất định theo từng cửa sổ và kiểm tra tương quan giữa các lần đọc. Khi tương quan đáng kể, dùng tỷ lệ **lô** thành công quan sát được để ước tính thời gian đến mục tiêu.

Ghi đồng thời số qubit sau nhúng, tỷ lệ đứt chuỗi, số mẫu khả thi, các trường `qpu_programming_time`, `qpu_sampling_time`, `qpu_access_time` và thời gian máy khách. Đối chứng CPU gồm cả xếp theo điểm rồi đổi chỗ. Các số đo đó cho phép thay trực tiếp ba cột xác suất giả định bằng thời gian quan sát ở cùng mức chất lượng.
