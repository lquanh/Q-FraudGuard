# Tích hợp Core Banking với Q-FraudGuard

Adapter HTTP v1 đã chạy trong sandbox với cùng mô hình chấm điểm và sổ hồ sơ của dashboard. Ngân hàng có thể gửi giao dịch, gửi biên nhận xác minh và đọc nhật ký theo con trỏ. Q-FraudGuard cung cấp hồ sơ cùng bằng chứng điều tra. Core Banking của ngân hàng phê duyệt và thực thi hành động tác động đến giao dịch hoặc tài khoản. [Adapter HTTP](../src/qf_guard/core_banking.py) · [Kiểm thử hợp đồng](../tests/test_core_banking.py)

## Chạy ở môi trường tích hợp của ngân hàng

1. Cài adapter bằng `python -m pip install -e ".[core-banking]"`. Nếu cần huấn luyện lại TGNN trên dữ liệu đã phê duyệt, dùng thêm extra `tgnn`.
2. Cấp `QFG_CASE_DB`, `QFG_STREAM` và `QFG_MODEL_PATH` cho adapter. Chọn tên luồng sandbox riêng theo phiên bản mô hình và dùng đúng ba giá trị này cho dashboard. Đặt `QFG_SOURCE=live` ở dashboard.
3. Cấp ba bearer token ngẫu nhiên, khác nhau và có tối thiểu 32 ký tự qua kho bí mật của ngân hàng. Adapter yêu cầu đúng token cho nhóm thao tác. Ngân hàng có thể đặt adapter sau API gateway nội bộ để thực thi TLS hoặc mTLS, giới hạn tốc độ và kiểm soát ứng dụng gọi API.
4. Khởi chạy một tiến trình adapter trên máy có quyền đọc artifact và ghi sổ vận hành. API gateway gửi yêu cầu về địa chỉ loopback này.

```powershell
python -m uvicorn qf_guard.core_banking:create_app --factory --host 127.0.0.1 --port 8766 --workers 1 --no-access-log
```

Trong sandbox có thể dùng mô hình PaySim đi kèm bằng cách đặt `QFG_MODEL_PATH=artifacts/models/paysim_lightgbm_scorer.joblib` và `QFG_DATASET=paysim` cho dashboard. Đặc tả máy đọc được tại `GET /openapi.json`. Cấu hình đã kiểm thử dùng SQLite, một worker và một luồng giao dịch có thứ tự thời gian. Để kết nối hệ thống thật, ngân hàng cấp endpoint và thông tin xác thực, đối chiếu ánh xạ trường, kiểm thử phục hồi và đo độ trễ ở mức tải mục tiêu.

## Hợp đồng HTTP v1

| Endpoint | Quyền | Dữ liệu và kết quả |
|---|---|---|
| `GET /healthz` | Mạng nội bộ | Kiểm tra luồng chấm điểm và sổ vận hành khả dụng. |
| `POST /v1/transactions` | `QFG_CORE_INGEST_TOKEN` | Nhận từ 1 đến 256 giao dịch theo thứ tự thời gian. Trả về `received`, `inserted` và `duplicates`. |
| `GET /v1/cases?transaction_id=…` | `QFG_CORE_READ_TOKEN` | Tra cứu Risk Score, Graph Score, nguồn điểm đồ thị, trạng thái và phiên bản hồ sơ. `state` và `version` có thể là `null` với giao dịch không có hồ sơ xử lý. |
| `POST /v1/verification-receipts` | `QFG_CORE_RECEIPTS_TOKEN` | Ghi nhận thông tin bổ sung hoặc kết quả xác minh. Cùng `receipt_id` và cùng nội dung trả về cùng xác nhận khi thử lại. |
| `GET /v1/case-events?after=0&limit=250` | `QFG_CORE_READ_TOKEN` | Trả sự kiện theo ID tăng dần và `next_after`. Có thể lọc theo `transaction_id`. |

Các endpoint nghiệp vụ nhận header `Authorization: Bearer <token>`. Body tối đa 1 MiB. Mã **401** báo lỗi xác thực, **413** báo vượt kích thước, **422** báo sai hợp đồng, **404** báo không tìm thấy giao dịch và **409** báo trùng mã khác nội dung, dữ liệu đến muộn hoặc phản hồi không khớp hồ sơ.

### Luồng giao dịch

```json
{
  "transactions": [{
    "transaction_id": "sandbox-transfer-0001",
    "timestamp": "2030-01-01T09:30:00+07:00",
    "sender_id": "acct-token-a",
    "receiver_id": "acct-token-b",
    "amount": 1500000,
    "amount_currency": "VND",
    "type": "TRANSFER",
    "payment_format": "FAST_TRANSFER",
    "country": "VN"
  }]
}
```

Ngân hàng ánh xạ mã giao dịch ổn định và mã tài khoản đã token hóa. Cùng một tài khoản giữ cùng mã trong luồng để TGNN nhận đúng liên kết. `timestamp` là thời điểm giao dịch có múi giờ. Với nhiều sự kiện trùng thời điểm, thứ tự nguồn quyết định sự kiện nào đến trước. Đơn vị số tiền khi vận hành phải nhất quán với dữ liệu huấn luyện. Số dư và thiết bị là trường tùy chọn. Nguồn ngân hàng giữ dữ liệu nhận dạng khách hàng và dữ liệu xác thực trong hệ thống của mình.

Gửi lại cùng mã giao dịch và nội dung không tạo thêm sự kiện. Nếu một lô có mã mâu thuẫn hoặc giao dịch mới đến trước thời điểm đã ghi, cả lô bị từ chối. Bên kết nối cần sắp theo thời gian nguồn và xử lý các giao dịch đến muộn trong quy trình đối soát.

Để đối soát bằng tệp trong sandbox, CLI nhận mỗi giao dịch trên một dòng JSONL. Với cùng `QFG_MODEL_PATH`, `QFG_CASE_DB` và `QFG_STREAM` như adapter, chạy `python scripts/stream.py ingest --input events.jsonl`. Lệnh `python scripts/stream.py case-history --after 0 --limit 100` đọc diễn biến hồ sơ theo con trỏ. [CLI nhập giao dịch và đọc nhật ký](../scripts/stream.py)

### Phản hồi xác minh và nhật ký

```json
{
  "receipt_id": "bank-verification-0001",
  "transaction_id": "sandbox-transfer-0001",
  "account_id": "acct-token-a",
  "expected_version": 1,
  "action": "verify_ok",
  "reason": "Kênh xác minh của ngân hàng đã đối chiếu yêu cầu."
}
```

`expected_version` lấy từ trạng thái hồ sơ. `account_id` phải khớp tài khoản của yêu cầu. `info_received` áp dụng cho hồ sơ đang chờ bổ sung. `verify_ok` và `verify_failed` áp dụng cho hồ sơ đang chờ xác minh. Adapter ghi phản hồi cùng nhật ký rồi đóng yêu cầu theo phiên bản. Kênh ngân hàng lưu `receipt_id` để gửi lại khi mất kết nối.

Consumer chỉ lưu `next_after` sau khi đã xử lý xong cả trang và dùng `id` của sự kiện làm khóa chống trùng. `state` là trạng thái tại thời điểm ghi sự kiện. `current_state` và `current_version` là trạng thái hiện tại tại lúc đọc. Kết quả xác minh và đề nghị của chuyên viên là đầu vào của quy trình phê duyệt tại ngân hàng.

## TGNN trong bước chấm điểm

TGNN (Temporal Graph Neural Network) tổng hợp thông tin từ **tối đa 8 giao dịch trước đó trên mỗi tài khoản trong 7 ngày**. Mô hình dùng số tiền, thời gian cách giao dịch hiện tại, hướng chuyển và việc lặp lại cùng đối tác. Dashboard hiển thị **TGNN Score** và adapter trả `graph_model=tgnn` khi dùng mô hình này. Các điểm thành phần được kết hợp thành Risk Score. Bước sàng lọc áp dụng ngưỡng đã hiệu chỉnh trên tập xác thực, sau đó QUBO mới nhận các ứng viên để phân bổ suất xem xét. [Mã TGNN](../src/qf_guard/temporal_graph.py) · [Quy trình chấm điểm](../src/qf_guard/scoring.py)

```powershell
python scripts/prepare_amlworld.py
python scripts/train.py --data data/amlworld_hi_small.csv --dataset-name amlworld_hi_small_sample --target-candidate-rate 0.15 --tgnn-epochs 6 --tgnn-min-weight 0.05 --output artifacts/models/amlworld_tgnn_pilot_scorer.joblib
```

Lệnh dùng ZIP AMLworld trong `data/downloads/`, tạo mẫu hai ngày, huấn luyện và lưu artifact pilot. PyTorch dùng khi huấn luyện, NumPy dùng khi suy luận. Tham số `--tgnn-min-weight 0.05` dành tối thiểu 5% trọng số cho TGNN trong đợt thử. Để tái tạo đối chứng, bỏ tham số này và lưu mô hình dưới tên khác. Khi thử một mô hình mới, đặt `QFG_MODEL_PATH` và dùng tên luồng mới để giữ nguyên lịch sử của các hồ sơ đã chấm. [Chuẩn bị dữ liệu](../data/DATASETS.md) · [Lệnh huấn luyện](../scripts/train.py)

### Kết quả sandbox trên AMLworld

Mẫu AMLworld tổng hợp có 414.812 giao dịch. Tập kiểm tra theo thời gian có 82.963 giao dịch, gồm 168 giao dịch có nhãn rửa tiền. Hai cấu hình dùng cùng tập kiểm tra và cùng ngân sách 12.445 cảnh báo (15%). Trọng số pilot chọn trên validation là 95% Supervised Learning và 5% TGNN.

| Cấu hình chấm điểm | Average Precision (PR-AUC) | Nhãn rửa tiền trong 12.445 cảnh báo |
|---|---:|---:|
| Trọng số tự chọn trên validation, TGNN 0% | 0,4046 | 157/168 |
| Pilot có TGNN 5% | **0,4166** | **157/168** |

Average Precision phản ánh chất lượng xếp hạng trên toàn dải điểm. Ở ngân sách 15%, hai cấu hình tìm thấy cùng số giao dịch mang nhãn. Bảng dùng đúng 12.445 vị trí xếp hạng. Số cảnh báo phát ra ở ngưỡng vận hành được tính riêng trong bản ghi đối chứng. [Trọng số, kích thước mẫu và kết quả đối chứng](../artifacts/training/amlworld_tgnn_pilot_comparison.json)
