# Ước tính thời gian lấy mẫu QPU D-Wave

Với **1.000 lượt đọc cùng một QUBO đã nạp**, thời gian lấy mẫu trên máy tham chiếu `Advantage2_system2` ước tính là **57,6–135,6 ms**. Số lượt đọc là tham số do nhóm thí điểm chọn trước. Mỗi lượt cho một danh sách ứng viên. Hệ thống kiểm tra danh sách chọn đúng K hồ sơ, chấm điểm theo cùng hàm QUBO rồi giữ danh sách tốt nhất. Chất lượng nghiệm đã kiểm chứng trên Quafu Haituo được trình bày trong hồ sơ riêng. [Kết quả kiểm chứng nghiệm Haituo](TRIEN_KHAI_QPU.md)

## Giả định và cách tính

Kịch bản đặt thời gian ủ là **20 µs**. Theo thông số máy tham chiếu, thời gian đọc điển hình nằm trong khoảng **17–95 µs** tùy số qubit và độ trễ sau khi đọc là **20,6 µs** mỗi mẫu. Khi giữ các tham số này, thời gian lấy mẫu ước tính bằng `số lượt đọc × (thời gian ủ + thời gian đọc + độ trễ sau khi đọc)`. Một mili giây (ms) bằng 1.000 micro giây (µs). [Thông số Advantage2_system2](https://docs.dwavequantum.com/en/latest/quantum_research/solver_properties_specific.html#advantage2-system2) · [Chu kỳ ủ và đọc](https://docs.dwavequantum.com/en/latest/quantum_research/operation_timing.html#anneal-read-cycle)

| Số lượt đọc cùng một QUBO | Phần ủ ước tính (ms) | Phần đọc và chờ ước tính (ms) | Tổng thời gian lấy mẫu ước tính (ms) |
|---:|---:|---:|---:|
| 1 | 0,020 | 0,0376–0,1156 | 0,0576–0,1356 |
| 10 | 0,20 | 0,376–1,156 | 0,576–1,356 |
| 100 | 2 | 3,76–11,56 | 5,76–13,56 |
| **1.000** | **20** | **37,6–115,6** | **57,6–135,6** |
| 10.000 | 200 | 376–1.156 | 576–1.356 |
| 100.000 | 2.000 | 3.760–11.560 | 5.760–13.560 |

Một cửa sổ mới có tập cảnh báo hoặc hệ số QUBO khác cần được lập trình lại. Dùng **24,5 ms** làm giả định cho một lần lập trình máy tham chiếu, một tác vụ gồm một lần nạp và 1.000 lượt đọc dự kiến chiếm **82,1–160,1 ms** cho hai bước này. Nếu số lượt vượt giới hạn một tác vụ, tính thêm lần lập trình cho mỗi phần cần gửi. Để dự toán thời gian đến bàn chuyên viên, cộng cả dựng bài, truyền dữ liệu, xếp hàng, kiểm tra danh sách và ghi kết quả. [Thông số lập trình tham chiếu](https://docs.dwavequantum.com/en/latest/quantum_research/solver_properties_specific.html#advantage2-system2) · [Các thành phần thời gian dịch vụ](https://docs.dwavequantum.com/en/latest/quantum_research/operation_timing.html) · [Giới hạn lượt đọc](https://docs.dwavequantum.com/en/latest/quantum_research/solver_properties_all.html#num-reads-range)

## Chọn ngân sách lấy mẫu cho thí điểm

Với ngân sách đã dành riêng cho lấy mẫu, chia ngân sách cho thời gian của một lượt rồi làm tròn xuống. Ví dụ, nếu dành **100 ms** cho một cửa sổ đã nạp, mô hình cho khoảng **737–1.736 lượt đọc**. Khi nghiệm thu, ngân hàng có thể thay đổi số lượt đọc, giữ cố định các hồ sơ cùng số suất K, rồi so sánh điểm QUBO của danh sách thu được với kết quả đối chiếu trên CPU.

| Ngân sách lấy mẫu giả định | Số lượt đọc ước tính ở hai đầu dải |
|---:|---:|
| 100 ms | 737–1.736 |
| 500 ms | 3.687–8.680 |
| 1 giây | 7.374–17.361 |
| 5 giây | 36.873–86.805 |
| 15 giây | 110.619–260.416 |

## Quy mô cửa sổ và ánh xạ lên chip

QUBO kết hợp điểm rủi ro với liên kết giữa những hồ sơ dùng chung tài khoản. Ràng buộc chọn đủ K suất tạo liên kết giữa mọi cặp biến. Các ánh xạ dưới đây đã được kiểm tra trên đồ thị Zephyr Z12 lý tưởng. Khi thử nghiệm, cần đối chiếu với danh sách qubit và liên kết đang hoạt động trên solver được cấp. [Mã kiểm tra ánh xạ](../scripts/benchmark_zephyr_windows.py) · [Hướng dẫn ánh xạ của D-Wave](https://docs.dwavequantum.com/en/latest/quantum_research/embedding_intro.html)

| Chọn K / N cảnh báo | Qubit vật lý trong ánh xạ Z12 lý tưởng |
|---:|---:|
| 10 / 40 | 160 |
| 12 / 48 | 216 |
| 16 / 64 | 352 |
| 20 / 80 | 520 |
| 24 / 96 | 720 |
| 28 / 112 | 952 |
| 32 / 128 | 1.216 |
| 36 / 144 | 1.512 |
| 40 / 160 | 1.840 |
| 44 / 176 | 2.200 |
| 46 / 184 | 2.392 |

## Lưu lượng giao dịch trước sàng lọc

Lấy tỷ lệ **129.766 cảnh báo trên 1.015.448 giao dịch** ở ngưỡng vận hành AMLworld làm giả định đầu vào. Chia số cảnh báo cho **160** để ước tính số cửa sổ chọn 40 hồ sơ. Ví dụ, với một triệu giao dịch mỗi giây, mô hình cho khoảng 127.792 cảnh báo và 798,70 cửa sổ mỗi giây. Nếu dành một lượt đọc cho mỗi cửa sổ đã nạp, tổng thời gian lấy mẫu ước tính là **46–108 ms** cho mỗi giây dữ liệu đến. Số lượt để đạt chất lượng mong muốn sẽ được xác định trong thí điểm. [Số cảnh báo ở ngưỡng vận hành](../artifacts/training/amlworld_full_operating_points.json)

| Giao dịch/s giả định | Cảnh báo/s dự kiến | Cửa sổ/s dự kiến | Giây lấy mẫu QPU ước tính cho một giây dữ liệu |
|---:|---:|---:|---:|
| 10.000 | 1.278 | 7,99 | 0,0005–0,0011 |
| 50.000 | 6.390 | 39,93 | 0,0023–0,0054 |
| 100.000 | 12.779 | 79,87 | 0,0046–0,0108 |
| 250.000 | 31.948 | 199,67 | 0,0115–0,0271 |
| 500.000 | 63.896 | 399,35 | 0,0230–0,0542 |
| 750.000 | 95.844 | 599,02 | 0,0345–0,0812 |
| **1.000.000** | **127.792** | **798,70** | **0,0460–0,1083** |
| 1.500.000 | 191.688 | 1.198,05 | 0,0690–0,1625 |
| 2.000.000 | 255.584 | 1.597,40 | 0,0920–0,2166 |
| 3.000.000 | 383.376 | 2.396,10 | 0,1380–0,3249 |
| 5.000.000 | 638.959 | 3.993,50 | 0,2300–0,5415 |
| 7.500.000 | 958.439 | 5.990,24 | 0,3450–0,8123 |
| 9.000.000 | 1.150.127 | 7.188,29 | 0,4140–0,9747 |
| 10.000.000 | 1.277.919 | 7.986,99 | 0,4601–1,0830 |
| 15.000.000 | 1.916.878 | 11.980,49 | 0,6901–1,6246 |
| 20.000.000 | 2.555.837 | 15.973,98 | 0,9201–2,1661 |
| 25.000.000 | 3.194.797 | 19.967,48 | 1,1501–2,7076 |

Các phép nhân dùng số cửa sổ chưa làm tròn. Với R lượt đọc mỗi cửa sổ, nhân thời gian trong cột cuối với R. Khi lập kế hoạch công suất, tính thêm số tác vụ cần nạp, phần hàng đợi và các bước nghiệp vụ cho từng cửa sổ. Ngân hàng có thể thay tỷ lệ cảnh báo của dữ liệu tổng hợp bằng tỷ lệ quan sát trong thí điểm.

## Khi đầu vào đã là cảnh báo

Nếu ngân hàng chuyển trực tiếp luồng cảnh báo, lấy số cảnh báo mỗi giây chia cho 160. Bảng giả định mỗi cửa sổ đủ 160 cảnh báo và đã nạp xong trước một lượt đọc.

| Cảnh báo/s giả định | Cửa sổ/s dự kiến | Giây lấy mẫu QPU ước tính cho một giây dữ liệu |
|---:|---:|---:|
| 100.000 | 625,00 | 0,0360–0,0847 |
| 250.000 | 1.562,50 | 0,0900–0,2119 |
| 500.000 | 3.125,00 | 0,1800–0,4237 |
| **1.000.000** | **6.250,00** | **0,3600–0,8475** |
| 1.500.000 | 9.375,00 | 0,5400–1,2713 |
| 2.000.000 | 12.500,00 | 0,7200–1,6950 |
| 3.000.000 | 18.750,00 | 1,0800–2,5425 |
| 5.000.000 | 31.250,00 | 1,8000–4,2375 |

Ở mức một triệu cảnh báo/giây, ngân sách trung bình cho một cửa sổ là 160 µs. Một lượt đọc dự kiến dùng 36–84,75% ngân sách đó. Tại ngân hàng, thời gian phản hồi được tính từ lúc nhận giao dịch đến lúc danh sách đã xác minh xuất hiện trên bàn chuyên viên. Đây là chỉ tiêu nghiệm thu cho toàn bộ quy trình.
