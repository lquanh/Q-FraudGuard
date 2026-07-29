# Danh mục dữ liệu Q-FraudGuard

Ngày truy xuất: 2026-07-29.

## Dữ liệu huấn luyện và đánh giá

| Bộ dữ liệu | Vị trí cục bộ | Nội dung đã tải | Giấy phép | Vai trò dự kiến |
|---|---|---|---|---|
| PaySim | `paysim_full/` | 6.362.620 giao dịch mobile money có nhãn | CC BY-SA 4.0 | Bộ dữ liệu chính để phát triển prototype |
| Elliptic Bitcoin Dataset | `elliptic/` | 203.769 node giao dịch, 234.355 cạnh và vector đặc trưng | CC BY-NC-ND 4.0 | Thí nghiệm graph/TGNN với nhãn hợp pháp, phi pháp và chưa xác định |
| Bank Account Fraud Suite | `bank_account_fraud/` | Base và 5 biến thể, mỗi file 1.000.000 bản ghi | CC BY-NC-SA 4.0 | Baseline gian lận mở tài khoản trong phạm vi phi thương mại |
| IBM AMLworld | `downloads/ibm_amlworld.zip` | Archive đầy đủ gồm 18 file, 41.608.549.656 byte khi giải nén | CDLA-Sharing-1.0 | Thí nghiệm AML theo giao dịch và đồ thị |

Nguồn:

- PaySim: <https://www.kaggle.com/datasets/ealaxi/paysim1>
- Elliptic: <https://www.kaggle.com/datasets/ellipticco/elliptic-data-set>
- Bank Account Fraud Suite: <https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022>
- IBM AMLworld: <https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml>

Lưu ý: trang Kaggle hiện ghi Bank Account Fraud Suite theo giấy phép CC BY-NC-SA 4.0. Catalog dùng thông tin này thay cho mô tả CC BY-NC-ND 4.0 ban đầu.

## Quy tắc sử dụng

- Tôn trọng điều khoản phi thương mại và không phái sinh của Elliptic.
- Chỉ dùng Bank Account Fraud Suite trong phạm vi giấy phép CC BY-NC-SA 4.0.
- Không đưa các archive lớn trong `downloads/` vào Docker image runtime.
- Việc lặp lại PaySim từ đầu chỉ mô phỏng stream; không tạo thêm bản ghi trong file nguồn.

## Trạng thái IBM AMLworld

Archive đầy đủ có dung lượng 8.176.169.418 byte và đã mở/kiểm tra central
directory thành công. Hai nhóm HI-Small và LI-Small có tổng dung lượng giải nén
1.207.810.843 byte. Do ổ D: không còn đủ dung lượng an toàn, các CSV Small vẫn
được giữ bên trong archive; chỉ `ibm_amlworld/HI-Small_Patterns.txt` đã được
giải nén. Không có file CSV giải nén dở dang được giữ lại.
