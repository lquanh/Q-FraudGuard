# CPU và QPU khi chọn hồ sơ từ 40, 80 và 160 cảnh báo

Bộ giải lượng tử có thể vượt bộ giải cổ điển. Với bài toán hiện tại của Q-FraudGuard, **CPU vẫn là lựa chọn thực tế**. Chọn 10 trong 40 cảnh báo đã khả thi về số qubit. Thách thức là trả về một danh sách tốt ngang cách chọn gần đúng trên CPU (heuristic). CPU đang mất khoảng **1,6–3,2 mili giây** cho bước này ở hai cỡ cửa sổ 40 và 80 cảnh báo.

Với cách mã hóa hiện tại, mỗi cảnh báo cần một biến nhị phân. Điều kiện “chọn đúng 10” tạo liên kết giữa cả **780 cặp** trong 40 biến (`40 × 39 / 2`). Trên máy ủ lượng tử D-Wave, một biến có thể cần một chuỗi qubit vật lý để nối đủ các cặp. Các ánh xạ trong bảng đã được kiểm tra trên sơ đồ **Zephyr Z12 lý tưởng**. Số qubit thực dùng có thể thay đổi theo từng chip. [Cách D-Wave ánh xạ biến logic lên qubit vật lý](https://docs.dwavequantum.com/en/latest/quantum_research/embedding_intro.html)

| Bài toán | Qubit vật lý trong ánh xạ đã kiểm tra | Heuristic CPU, trung vị | MILP trên CPU, trung vị, có chứng nhận tối ưu |
|---|---:|---:|---:|
| Chọn 10 trong 40 | 160 | 1,64 ms | 6,59 ms |
| Chọn 20 trong 80 | 560 | 3,20 ms | 9,48 ms |
| Chọn 40 trong 160 | 1.920 | 9,53 ms | 20,56 ms |

Phép đo CPU do tác giả cung cấp dùng cùng **1.920 ứng viên AMLworld tổng hợp**, chia thành 48, 24 và 12 cửa sổ tương ứng, với một luồng tính toán BLAS. Độ hụt trung bình của heuristic so với điểm QUBO tối ưu là **1,0%** ở 40 cảnh báo, **2,2%** ở 80 và **3,3%** ở 160.

## Thời gian lấy mẫu trên QPU đặt tại công ty

Giả sử công ty dùng phần cứng tương đương `Advantage2_system2`. Đồng hồ bắt đầu sau khi bài QUBO đã được nạp và chỉ tính thời gian lấy mẫu trên chip (`qpu_sampling_time`). Cách tính này bỏ thời gian truyền mạng, xếp hàng, nạp hệ số và xử lý kết quả trên CPU. Nạp hệ số vẫn là một thao tác vật lý khi dùng máy tại chỗ. [Chu kỳ lập trình và lấy mẫu của D-Wave](https://docs.dwavequantum.com/en/latest/quantum_research/operation_timing.html#programming-cycle)

Mỗi lượt đọc trả về một phương án ứng viên cho cả cửa sổ. Với lịch ủ **20 µs**, thời gian đọc **17–95 µs** và thời gian ổn định chip **20,6 µs**, tổng thời gian cho **R lượt ≈ R × (57,6–135,6) µs**. Dải đọc của hãng bao quát từ một qubit đến toàn chip. Trên cùng một QUBO, **1.000 lượt** mất khoảng **57,6–135,6 ms**, **10.000 lượt** mất **0,576–1,356 s** và **một triệu lượt** mất **57,6–135,6 s**. Nếu chia thành nhiều lô, các con số này là tổng thời gian lấy mẫu của các lô. [Thông số Advantage2_system2](https://docs.dwavequantum.com/en/latest/quantum_research/solver_properties_specific.html#advantage2-system2) · [Cách D-Wave tính thời gian lấy mẫu](https://docs.dwavequantum.com/en/latest/quantum_research/operation_timing.html#anneal-read-cycle)

Khi đầu ra cần **một danh sách mỗi cửa sổ**, CPU chạy heuristic một lần. Bảng cho biết số lượt đọc nguyên lớn nhất mà tổng thời gian lấy mẫu QPU vẫn thấp hơn trung vị CPU, ở hai đầu của dải tham chiếu. Nếu đầu ra cần một triệu phương án cho cùng một cửa sổ, dự án chưa đo tốc độ sinh mẫu tương ứng trên CPU.

| Bài toán | CPU cho một danh sách | Ở 135,6 µs/lượt | Ở 57,6 µs/lượt |
|---|---:|---:|---:|
| Chọn 10 trong 40 | 1,64 ms | 12 lượt | 28 lượt |
| Chọn 20 trong 80 | 3,20 ms | 23 lượt | 55 lượt |
| Chọn 40 trong 160 | 9,53 ms | 70 lượt | 165 lượt |

Ở mức mặc định **100 lượt/cửa sổ** của dự án, để nhanh hơn CPU ở cỡ 160 cảnh báo, mỗi chu kỳ cần dưới **95,3 µs** (`9,53 ms / 100`), tức phần đọc qubit phải dưới **54,7 µs** sau khi trừ phần ủ và ổn định. Dự án chưa đo thời gian đọc của ánh xạ này trên chip thật hoặc số lượt cần để chọn đúng 40 hồ sơ với điểm QUBO ngang heuristic. [Cấu hình và trường thời gian của bộ giải D-Wave](../src/qf_guard/solvers.py)

### Một triệu cảnh báo đã được chấm rủi ro

Chia **1.000.000 cảnh báo** thành **6.250 cửa sổ độc lập**, mỗi cửa sổ chọn 40 trong 160 hồ sơ. Nếu xử lý tuần tự, thời gian chọn hồ sơ trên CPU ước tính **59,56 s** (`6.250 × 9,53 ms`). Đây là ngoại suy từ trung vị của 12 cửa sổ đã đo. Phía QPU dùng cùng số cửa sổ và cộng thời gian lấy mẫu.

| Lượt đọc mỗi cửa sổ | QPU cho một cửa sổ | QPU cho 6.250 cửa sổ | So với CPU ở mức 59,56 s |
|---:|---:|---:|---|
| 50 | 2,88–6,78 ms | 18,00–42,38 s | Lấy mẫu QPU nhanh hơn khoảng 1,4–3,3 lần |
| 100 | 5,76–13,56 ms | 36,00–84,75 s | Phụ thuộc thời gian đọc thực tế |
| 1.000 | 57,6–135,6 ms | 360–847,5 s | CPU nhanh hơn khoảng 6–14 lần |

Nếu 50 lượt cho một danh sách tốt ngang CPU, phần lấy mẫu QPU tiết kiệm khoảng **17–42 giây** trên một triệu cảnh báo trong kịch bản này. Tăng số cửa sổ cùng kích thước nhân thời gian của cả hai bên với cùng một hệ số. Nếu muốn xét chung cả triệu cảnh báo trong **một** bài QUBO, riêng mã hóa hiện tại đã cần một triệu biến logic. Chip tham chiếu có 4.506 qubit vật lý. [Số qubit của Advantage2_system2](https://docs.dwavequantum.com/en/latest/quantum_research/solver_properties_specific.html#advantage2-system2)

## Thời gian từ lúc gửi đến lúc nhận kết quả

Giả sử một QPU tương lai trả về danh sách có chất lượng ngang heuristic trong **5 ms**, tính cả gửi bài, chờ, lấy mẫu và nhận kết quả. Với 40 cảnh báo, CPU vẫn nhanh hơn khoảng **3,1 lần** nếu lấy thời gian CPU làm tròn là 1,6 ms. Với 160 cảnh báo, QPU sẽ nhanh hơn heuristic đã đo khoảng **1,9 lần** (`9,53 / 5`), tiết kiệm khoảng **4,5 ms mỗi cửa sổ** (`9,53 − 5 = 4,53 ms`).

**5 ms là giả định để tính ngưỡng.** Trên `Advantage2_system2`, D-Wave công bố riêng thời gian điển hình để nạp một bài mới đã khoảng **24,5 ms**. [Thông số QPU Advantage2_system2 của D-Wave](https://docs.dwavequantum.com/en/latest/quantum_research/solver_properties_specific.html#advantage2-system2)

Thêm qubit giúp xét chung nhiều cảnh báo hơn. Nếu ngân hàng vẫn xử lý từng nhóm 40, kể cả một QPU trả lời tức thì cũng chỉ tiết kiệm khoảng **1,6 ms** ở bước chọn hồ sơ, theo trung vị đã đo. Lý do đáng phát triển tiếp là khi điều tra viên cần xét chung những nhóm lớn hơn để lần theo dòng tiền và QPU có thể trả về danh sách tốt hơn trong thời hạn thực tế của ca trực.

### Một tác vụ trên Advantage2_system2 hiện tại

Với một cửa sổ mới, QPU nạp hệ số QUBO một lần rồi lấy số mẫu yêu cầu. Bảng dùng lịch ủ 20 µs và cùng dải thời gian đọc như trên cho cả ba ánh xạ 160, 560 và 1.920 qubit vật lý. Muốn tính sát từng ánh xạ cần đường cong thời gian đọc của đúng chip (`problem_timing_data`). [Cách Ocean tính theo số qubit đã nhúng](https://docs.dwavequantum.com/en/latest/quantum_research/operation_timing.html#estimating-access-time)

| Một bài QUBO mới | Tổng thời gian lấy mẫu | Nạp 24,5 ms + lấy mẫu | Nếu cộng 1 ms ổn định sau nạp |
|---|---:|---:|---:|
| 1 mẫu | 0,058–0,136 ms | 24,56–24,64 ms | 25,56–25,64 ms |
| 100 mẫu trong cùng tác vụ (mặc định `dwave-qpu`) | 5,76–13,56 ms | 30,26–38,06 ms | 31,26–39,06 ms |

Phép tính dùng thời gian nạp điển hình cộng số mẫu nhân với thời gian một chu kỳ. Cột cuối minh họa mức chờ ổn định 1 ms thường gặp trong hướng dẫn của hãng. Khi có kết quả thật, `qpu_programming_time` đã bao gồm khoản chờ này. [Chu kỳ lập trình và lấy mẫu của D-Wave](https://docs.dwavequantum.com/en/latest/quantum_research/operation_timing.html#programming-cycle) · [Mặc định số mẫu trong dự án](../src/qf_guard/solvers.py)

Thời gian **từ lúc gửi đến lúc nhận danh sách** còn gồm khởi tạo cấp thấp, truyền mạng, xếp hàng và xử lý kết quả. Dự án chưa có số đo đầu-cuối trên D-Wave cho ba cỡ cửa sổ này. Ở nhóm 160 cảnh báo, trung vị **9,53 ms** của heuristic CPU vẫn thấp hơn riêng thời gian nạp bài điển hình trên D-Wave. Để đối chiếu với phần cứng đã chạy, trên **Quafu Haituo** ở 18 qubit, trung vị từ lúc gửi đến lúc nhận lô 1.024 mẫu là **2,78 giây**, và từ lúc bắt đầu xử lý đến khi xác minh nghiệm là **2,83 giây** trên 10 cửa sổ. [Đồng hồ dịch vụ D-Wave](https://docs.dwavequantum.com/en/latest/quantum_research/operation_timing.html#qmi-timing) · [Bản ghi Haituo](../artifacts/deploy/quafu_verified_hits_8_12_15_18_20261118T212255Z.json)
