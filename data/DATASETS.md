# Dữ liệu dùng trong Q-FraudGuard

Benchmark Quafu dùng giao dịch ngân hàng tổng hợp **IBM AMLworld HI-Small**. Điểm rủi ro của mỗi giao dịch và các tài khoản gửi, nhận xác định bài toán chọn hồ sơ trên QPU. Hai giao dịch tạo thành một cặp liên kết khi chúng có chung ít nhất một tài khoản. [Nguồn IBM AMLworld](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) · [Giấy phép và ghi công](licenses/LICENSES.md)

## Kiểm lại 44 cửa sổ đã chạy trên chip

Bản trích đi kèm có 559 giao dịch riêng biệt xuất hiện trong 44 cửa sổ. Mỗi dòng chứa `transaction_id`, `risk_score`, `sender_id`, `receiver_id` và nhãn rửa tiền `is_fraud` (0 hoặc 1). Hai báo cáo JSON lưu danh sách ID theo thứ tự bit cùng tần suất các mẫu chip trả về. Bốn cột đầu cho phép khôi phục bài toán. Nhãn được ghép theo ID từ `full_scores_test.pkl` và đối chiếu với phần kiểm tra (`split=stream`) của CSV phát lại để đánh giá sau khi chọn hồ sơ. [CSV ứng viên](quafu_benchmark_candidates.csv) · [40 cửa sổ chọn trước](../artifacts/deploy/quafu_verified_hits_8_12_15_18_20261118T212255Z.json) · [4 cửa sổ thử thách](../artifacts/deploy/quafu_ranking_misses_8_12_15_18_20261118T212601Z.json)

Trong 559 giao dịch có 3 nhãn dương. Cả ba nằm ở tập cửa sổ chọn trước. QPU và CPU xếp theo điểm đều chọn trúng 3/130 lượt hồ sơ ở tập này, và 0/13 ở tập thử thách. Mẫu số cộng K của mỗi cửa sổ. Bốn cửa sổ thử thách được chọn vì top-K bỏ lỡ tối ưu QUBO nên cần báo cáo riêng khi đánh giá chất lượng phát hiện.

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

## Mô hình nào được dùng?

Dashboard đọc `QFG_DATASET` để chọn mô hình. Mặc định là `amlworld_full`. Nếu có `QFG_MODEL_PATH`, đường dẫn đó được ưu tiên. Bảng dưới nhận diện sáu tệp `.joblib` hiện có trong thư mục mô hình cục bộ. Các tệp JSON đi kèm lưu ngưỡng hoặc chỉ số đánh giá. [Ánh xạ trong dashboard](../dashboard/app.py)

| Cấu hình dashboard | Tệp trong `artifacts/models/` | Vai trò |
|---|---|---|
| `amlworld_full` (mặc định) | `amlworld_full_scorer.joblib` | Luồng AMLworld mười ngày chia theo thời gian, có danh sách tài khoản đã xác nhận, bỏ hình thức thanh toán |
| `amlworld` | `amlworld_lightgbm_scorer.joblib` | Mô hình cơ sở trên mẫu hai ngày |
| `amlworld_tuned` | `amlworld_noformat_tuned_scorer.joblib` | Mô hình bốn ngày đã tinh chỉnh, bỏ hình thức thanh toán |
| `paysim` | `paysim_lightgbm_scorer.joblib` | Mô hình cho luồng PaySim |
| Chọn qua đường dẫn | `amlworld_tuned_scorer.joblib` | Mô hình được chọn trong báo cáo bốn ngày có dùng hình thức thanh toán, PR-AUC kiểm tra 0,518 |
| Chọn qua đường dẫn | `amlworld_confirmed_scorer.joblib` | Biến thể thử nghiệm có dùng danh sách tài khoản đã xác nhận |

| Phép đánh giá | Ngày giao dịch trong gói dữ liệu | Huấn luyện | Xác thực | Kiểm tra | Rửa tiền trên tập kiểm tra | PR-AUC kiểm tra |
|---|---|---:|---:|---:|---:|---:|
| Mô hình được chọn trong báo cáo bốn ngày | 3–6/9/2022 | 827.730 | 275.910 | 275.911 | 339 | 0,518 |
| Mô hình của luồng `amlworld_full` | 1–10/9/2022 | 3.046.342 | 1.015.447 | 1.015.448 | 1.143 | 0,434 |

PR-AUC ở đây là average precision, tính trên toàn bộ dải điểm của từng tập kiểm tra. Báo cáo huấn luyện mười ngày ghi ngưỡng nhắm recall 95% trên tập xác thực. Luồng hiện tại dùng ngưỡng nhắm 96%. Bảng ngưỡng trong README ghi kết quả của cả hai trên cùng tập điểm mười ngày. [Báo cáo bốn ngày](../artifacts/training/amlworld_training_report.md) · [Báo cáo mười ngày](../artifacts/training/amlworld_full_report.json) · [Kết quả từng ngưỡng](../artifacts/training/amlworld_full_operating_points.json)

Các bản ghi Quafu chỉ rõ nguồn điểm `full_scores_test.pkl` và lưu mã giao dịch đã chọn. Điểm và tài khoản của cả 559 giao dịch khớp với bản xuất của luồng đầy đủ. Mã SHA-256 của tệp mô hình cục bộ được lưu cùng ngưỡng chuyển chuyên viên. Bản ghi chip chưa lưu mã băm mô hình tại thời điểm chạy. [Ngưỡng và mã băm của mô hình mặc định](../artifacts/models/amlworld_full_scorer.review.json)

## Các ZIP nguồn lưu cục bộ

| Tệp trong `data/downloads/` | Nội dung |
|---|---|
| `ibm_amlworld.zip` | Các bộ giao dịch AML tổng hợp, tài khoản và danh mục mẫu rửa tiền |
| `paysim_full.zip` | 6.362.620 giao dịch tiền di động tổng hợp |
| `elliptic.zip` | Đồ thị Bitcoin gồm 203.769 giao dịch, 234.355 cạnh và các đặc trưng |
| `bank_account_fraud.zip` | Bộ Base và 5 biến thể, mỗi bộ 1.000.000 bản ghi |

Dự án lưu mã SHA-256 cho bốn ZIP để kiểm tra bản tải về. Elliptic và Bank Account Fraud Suite nằm trong kho nghiên cứu cục bộ. Các mẫu và kết quả dùng cho prototype được chỉ rõ ở hai mục trên. [Mã SHA-256](downloads/SHA256SUMS.txt) · [Nguồn tải và điều khoản từng bộ](licenses/LICENSES.md)
