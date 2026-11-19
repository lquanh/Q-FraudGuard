# Đánh giá mô hình chấm điểm giao dịch trên CPU

Thí nghiệm dùng `amlworld_hi_small_4d.csv`, một mẫu giao dịch tổng hợp IBM AMLworld. Mọi cấu hình chấm cùng một tập kiểm tra. Mô hình được chọn theo PR-AUC trên tập xác thực. Ở các cấu hình dùng danh sách tài khoản đã xác nhận, nhãn được sử dụng sau 24 giờ.

**Kết quả của mô hình được chọn.** PR-AUC trên tập kiểm tra đạt **0,518**. Trong 4% giao dịch có điểm cao nhất, mô hình giữ được **85,5%** số giao dịch rửa tiền của tập kiểm tra.

## Dữ liệu theo thứ tự thời gian

| Tập | Giao dịch | Giao dịch rửa tiền |
|---|---:|---:|
| Huấn luyện | 827.730 | 1.223 |
| Xác thực | 275.910 | 238 |
| Kiểm tra | 275.911 | 339 |

## So sánh các cấu hình

Recall ở top 4% là tỷ lệ giao dịch rửa tiền tìm thấy khi chỉ xem 4% giao dịch có điểm cao nhất. PR-AUC trong báo cáo được tính bằng average precision. Các khoảng trong bảng là khoảng tin cậy 95% ước lượng bằng bootstrap trên giao dịch. Cột cuối so sánh recall với dòng “Mô hình đối chứng huấn luyện trên mẫu hai ngày” trên cùng các mẫu bootstrap.

| Cấu hình | PR-AUC xác thực | PR-AUC kiểm tra | ROC-AUC kiểm tra | Recall ở top 4% | Khoảng chênh lệch recall so với đối chứng (điểm phần trăm) |
|---|---:|---:|---:|---:|---:|
| Mô hình đối chứng huấn luyện trên mẫu hai ngày | — | 0,266 (0,219 đến 0,313) | 0,960 | 74,9% (70,1% đến 79,5%) | +0,0 đến +0,0 |
| Đặc trưng cơ bản | 0,236 | 0,412 (0,354 đến 0,464) | 0,973 | 81,4% (77,1% đến 85,6%) | +2,9 đến +10,6 |
| Thêm đặc trưng liên ngân hàng, tự chuyển và chênh lệch tiền tệ | 0,229 | 0,435 (0,376 đến 0,488) | 0,973 | 81,7% (77,4% đến 85,9%) | +3,0 đến +10,8 |
| Thêm hình thức thanh toán | 0,281 | 0,431 (0,375 đến 0,485) | 0,980 | 84,7% (80,5% đến 88,7%) | +5,9 đến +13,2 |
| Thêm danh sách tài khoản đã xác nhận | 0,368 | 0,524 (0,464 đến 0,574) | 0,985 | 89,1% (85,7% đến 92,4%) | +10,2 đến +18,0 |
| Đặc trưng giao dịch và danh sách đã xác nhận, bỏ hình thức thanh toán | 0,340 | 0,501 (0,440 đến 0,553) | 0,983 | 88,5% (85,0% đến 91,9%) | +9,6 đến +17,2 |
| **LightGBM tinh chỉnh trên nhóm đặc trưng chọn ở tập xác thực (được chọn)** | 0,416 | 0,518 (0,457 đến 0,569) | 0,983 | 85,5% (81,6% đến 89,2%) | +6,6 đến +14,9 |

Mô hình của lần chạy này được lưu với tên `amlworld_tuned_scorer.joblib`. Thời gian huấn luyện, dò tham số và đánh giá là 1748,0 giây. Các tham số LightGBM, điểm từng lượt dò và số liệu trước khi làm tròn nằm trong [báo cáo JSON của cùng lần chạy](amlworld_training_report.json).
