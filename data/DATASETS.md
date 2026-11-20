# Dữ liệu và mô hình Q-FraudGuard

IBM AMLworld HI-Small cung cấp giao dịch ngân hàng tổng hợp có thời điểm, tài khoản gửi, tài khoản nhận và nhãn rửa tiền. Q-FraudGuard dùng các trường giao dịch để chấm điểm. Nhãn `is_fraud` trong bản trích AMLworld biểu thị nhãn rửa tiền của nguồn và dùng để đánh giá kết quả. PaySim cung cấp một luồng giao dịch tiền di động tổng hợp riêng cho dashboard. [Nguồn và giấy phép từng bộ](licenses/LICENSES.md)

## Mô hình nào được dùng?

Dashboard chọn mô hình theo `QFG_DATASET`. `QFG_MODEL_PATH` cho phép chỉ định artifact cụ thể. Khi dùng luồng `live`, adapter và dashboard phải cùng dùng tên luồng, đường dẫn mô hình và kho hồ sơ. [Cấu hình dashboard](../dashboard/app.py) · [Hướng dẫn Core Banking](../docs/CORE_BANKING.md)

| Cấu hình | Artifact trong `artifacts/models/` | Phạm vi |
|---|---|---|
| `amlworld_full` (mặc định) | `amlworld_full_scorer.joblib` | Mẫu mười ngày, có đặc trưng tài khoản đã xác nhận và loại đặc trưng hình thức thanh toán. Nguồn điểm cho benchmark Haituo. |
| `amlworld_tuned` | `amlworld_noformat_tuned_scorer.joblib` | Biến thể bốn ngày bỏ hình thức thanh toán. Mô hình được chọn trong báo cáo so sánh bốn ngày có tên riêng là `amlworld_tuned_scorer.joblib`. |
| `amlworld` | `amlworld_lightgbm_scorer.joblib` | Mô hình cơ sở trên mẫu hai ngày. |
| `paysim` | `paysim_lightgbm_scorer.joblib` | Mô hình và mẫu PaySim 99.416 giao dịch đi kèm dự án. |
| Sandbox TGNN, qua `QFG_MODEL_PATH` | `amlworld_tgnn_pilot_scorer.joblib` | Mẫu hai ngày, trọng số TGNN 5% trong cấu hình pilot. [Lệnh huấn luyện](../docs/CORE_BANKING.md#tgnn-trong-bước-chấm-điểm) |

### Kết quả theo từng tập kiểm tra

| Thí nghiệm AMLworld | Huấn luyện | Xác thực | Kiểm tra | Nhãn dương trong tập kiểm tra | Average Precision |
|---|---:|---:|---:|---:|---:|
| Mô hình sàng lọc mười ngày, 1–10/9/2022 | 3.046.342 | 1.015.447 | 1.015.448 | 1.143 | 0,4341 |
| So sánh cấu hình bốn ngày, 3–6/9/2022 | 827.730 | 275.910 | 275.911 | 339 | 0,518 ở mô hình được chọn |
| Pilot TGNN hai ngày, 3–4/9/2022 | 248.887 | 82.962 | 82.963 | 168 | 0,4046 ở đối chứng và 0,4166 với TGNN 5% |

Mỗi thí nghiệm chia giao dịch theo thời gian. Khi so sánh hai cấu hình, dùng cùng tập kiểm tra và cùng ngân sách cảnh báo trong một thí nghiệm. Mô hình mười ngày dùng ngưỡng vận hành nhắm recall 96% trên tập xác thực. Trên tập kiểm tra, ngưỡng này giữ 1.089 nhãn dương trong 129.766 cảnh báo. [Báo cáo mười ngày](../artifacts/training/amlworld_full_report.json) · [Kết quả từng ngưỡng](../artifacts/training/amlworld_full_operating_points.json) · [Báo cáo bốn ngày](../artifacts/training/amlworld_training_report.md) · [Đối chứng TGNN](../artifacts/training/amlworld_tgnn_pilot_comparison.json)

## Tái lập các luồng dữ liệu

Chế độ `recorded` đọc 64 giao dịch đã chấm điểm, ngưỡng điều tra và mã băm mô hình trong CSV đi kèm. Dashboard đối chiếu 18 ứng viên và 4 mã hồ sơ với chuỗi bit trong bản ghi Quafu. Chế độ `replay` có thể chạy trên mẫu PaySim đi kèm bằng `QFG_DATASET=paysim`. [Dữ liệu phiên Haituo](quafu_dashboard_replay.csv) · [Bản ghi phần cứng](../artifacts/deploy/quafu_dashboard_18_20261118T220101Z.json) · [Chạy dashboard](../README.md#chạy-phiên-minh-họa)

Để tạo mô hình AMLworld trên máy mới, đặt gói IBM tại `data/downloads/ibm_amlworld.zip`. Lệnh đầu đọc `HI-Small_Trans.csv`, huấn luyện trên kỳ 1–10/9 với 5.077.237 giao dịch chia theo thời gian, xuất điểm cho benchmark và CSV phát lại kỳ kiểm tra. CSV gồm 20.000 giao dịch xác thực làm lịch sử và 1.015.448 giao dịch kiểm tra. Chế độ `loop` đọc đủ 5.078.345 giao dịch trong thành viên ZIP, kể cả giao dịch sau ngày 10/9. Hai lệnh sau tạo mẫu hai và bốn ngày. [Nguồn IBM AMLworld](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) · [Mã huấn luyện](../scripts/train_amlworld_full.py) · [Mã SHA-256 của các ZIP](downloads/SHA256SUMS.txt)

```powershell
python scripts/train_amlworld_full.py --target-recall 0.96 --stream-output data/amlworld_hi_small_full_stream.csv --save-test-scores
python scripts/prepare_amlworld.py
python scripts/prepare_amlworld.py --days 4 --output data/amlworld_hi_small_4d.csv
```

`full_scores_test.pkl` trong `artifacts/training/` chứa điểm, tài khoản và nhãn để tính lại số hồ sơ ở từng ngưỡng. Ngưỡng chuyển chuyên viên nhắm 2% lưu lượng trên tập xác thực. Trên 1.015.448 giao dịch kiểm tra của `amlworld_full`, 31.407 giao dịch qua ngưỡng, trong đó 925 có nhãn rửa tiền (precision 2,95%). Tệp `.review.json` lưu ngưỡng và mã băm mô hình. Khi đối chiếu theo vụ rửa tiền, dùng thêm `HI-Small_Patterns.txt` từ ZIP. [Ngưỡng chuyển chuyên viên](../artifacts/models/amlworld_full_scorer.review.json)

## Kiểm lại 44 cửa sổ đã chạy trên chip

Bản trích có 559 giao dịch riêng biệt. Bốn trường `transaction_id`, `risk_score`, `sender_id` và `receiver_id` cho phép dựng lại QUBO. Hai báo cáo JSON lưu thứ tự ứng viên và tần suất mẫu đo. Chạy `python scripts/verify_quafu_evidence.py` để kiểm tra nghiệm và đối chiếu nhãn sau khi chọn. [CSV ứng viên](quafu_benchmark_candidates.csv) · [Mã kiểm chứng](../scripts/verify_quafu_evidence.py)

| Nhóm cửa sổ | Lượt có nhãn rửa tiền trong danh sách QPU | Lượt có nhãn trong danh sách Top-K |
|---|---:|---:|
| 40 cửa sổ chọn trước | 3/130 | 3/130 |
| 4 cửa sổ chọn vì Top-K bỏ lỡ tối ưu QUBO | 0/13 | 0/13 |

Mẫu số cộng số suất K qua các cửa sổ. Hai nhóm ghi nhận chất lượng nghiệm QUBO và số nhãn theo cùng ngân sách điều tra. [40 cửa sổ chọn trước](../artifacts/deploy/quafu_verified_hits_8_12_15_18_20261118T212255Z.json) · [4 cửa sổ chọn riêng](../artifacts/deploy/quafu_ranking_misses_8_12_15_18_20261118T212601Z.json)
