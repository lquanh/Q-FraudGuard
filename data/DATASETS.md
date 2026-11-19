# Dữ liệu dùng trong Q-FraudGuard

Benchmark Quafu dùng giao dịch ngân hàng tổng hợp **IBM AMLworld HI-Small**. Điểm rủi ro của mỗi giao dịch và các tài khoản gửi, nhận xác định bài toán chọn hồ sơ trên QPU. Hai giao dịch tạo thành một cặp liên kết khi chúng có chung ít nhất một tài khoản. [Nguồn IBM AMLworld](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) · [Giấy phép và ghi công](licenses/LICENSES.md)

## Kiểm lại 44 cửa sổ đã chạy trên chip

Bản trích đi kèm có 559 giao dịch riêng biệt xuất hiện trong 44 cửa sổ. Mỗi dòng chứa `transaction_id`, `risk_score`, `sender_id` và `receiver_id`. Hai báo cáo JSON lưu danh sách ID theo thứ tự bit trong từng cửa sổ cùng tần suất các mẫu chip trả về. Ghép các ID này với CSV cho phép khôi phục bài toán và kiểm lại kết quả bằng CPU. [CSV ứng viên](quafu_benchmark_candidates.csv) · [40 cửa sổ chọn trước](../artifacts/deploy/quafu_verified_hits_8_12_15_18_20261118T212255Z.json) · [4 cửa sổ thử thách](../artifacts/deploy/quafu_ranking_misses_8_12_15_18_20261118T212601Z.json)

Chạy từ thư mục gốc với các tệp trên. Lệnh chỉ dùng thư viện chuẩn Python.

```powershell
python scripts/verify_quafu_evidence.py
```

## Chọn dữ liệu để chạy dashboard hoặc huấn luyện

| Mục đích | Dữ liệu | Cách dùng |
|---|---|---|
| Mở dashboard cục bộ | `paysim.csv`, mẫu PaySim gồm 99.416 giao dịch | Đặt `QFG_DATASET=paysim`. Mẫu và mô hình đi kèm dự án. |
| Đánh giá các mô hình chấm điểm trên CPU | `amlworld_hi_small_4d.csv`, mẫu bốn ngày từ 3 đến 6/9/2022 | Là dữ liệu của báo cáo so sánh các bộ đặc trưng và cấu hình LightGBM. [Báo cáo bốn ngày](../artifacts/training/amlworld_training_report.md) |
| Tạo điểm ứng viên cho một lần chạy QPU mới | `downloads/ibm_amlworld.zip`, giao dịch HI-Small từ 1 đến 10/9/2022 | Script huấn luyện đọc trực tiếp thành viên `HI-Small_Trans.csv` và chia giao dịch theo thời gian. |
| Đánh giá theo vụ rửa tiền | `ibm_amlworld/HI-Small_Patterns.txt` | Trích thành viên cùng tên từ ZIP IBM khi cần đối chiếu theo vụ. |

**Tạo luồng AMLworld đầy đủ.** Từ thư mục gốc, lệnh dưới huấn luyện mô hình, tạo `data/amlworld_hi_small_full_stream.csv` cho dashboard và ghi `artifacts/training/full_scores_test.pkl` cho benchmark. CSV phát lại chứa 20.000 dòng cuối của tập xác thực làm lịch sử, sau đó là 1.015.448 giao dịch thuộc tập kiểm tra. [Hướng dẫn chạy trên chip](../docs/TRIEN_KHAI_QPU.md)

```powershell
python scripts/train_amlworld_full.py --target-recall 0.96 --stream-output data/amlworld_hi_small_full_stream.csv --save-test-scores
```

Để đối chiếu số liệu sàng lọc và chuyển chuyên viên trong README, lọc các dòng có `risk_score` lớn hơn hoặc bằng ngưỡng tương ứng trong tệp điểm kiểm tra, rồi đếm số dòng và tổng nhãn `is_fraud`. [Ngưỡng sàng lọc ở mục tiêu 0,96](../artifacts/training/amlworld_full_operating_points.json) · [Ngưỡng chuyển chuyên viên](../artifacts/models/amlworld_full_scorer.review.json)

Để tạo mẫu hai ngày hoặc bốn ngày cho các thí nghiệm chấm điểm đã lưu, chạy lệnh tương ứng.

```powershell
python scripts/prepare_amlworld.py
python scripts/prepare_amlworld.py --days 4 --output data/amlworld_hi_small_4d.csv
```

## Các ZIP nguồn lưu cục bộ

| Tệp trong `data/downloads/` | Nội dung |
|---|---|
| `ibm_amlworld.zip` | Các bộ giao dịch AML tổng hợp, tài khoản và danh mục mẫu rửa tiền |
| `paysim_full.zip` | 6.362.620 giao dịch tiền di động tổng hợp |
| `elliptic.zip` | Đồ thị Bitcoin gồm 203.769 giao dịch, 234.355 cạnh và các đặc trưng |
| `bank_account_fraud.zip` | Bộ Base và 5 biến thể, mỗi bộ 1.000.000 bản ghi |

Dự án lưu mã SHA-256 cho bốn ZIP để kiểm tra bản tải về. Elliptic và Bank Account Fraud Suite nằm trong kho nghiên cứu cục bộ. Các mẫu và kết quả dùng cho prototype được chỉ rõ ở hai mục trên. [Mã SHA-256](downloads/SHA256SUMS.txt) · [Nguồn tải và điều khoản từng bộ](licenses/LICENSES.md)
