# Nguồn và giấy phép dữ liệu

Bảng dưới ghi giấy phép mà bên cung cấp công bố trên trang của từng bộ dữ liệu. Mã nguồn Q-FraudGuard sử dụng giấy phép MIT. [Giấy phép mã nguồn](../../LICENSE)

| Bộ dữ liệu | Vai trò trong dự án | Nguồn và giấy phép dữ liệu |
|---|---|---|
| IBM AMLworld HI-Small | Huấn luyện mô hình chấm điểm và lập cửa sổ cho benchmark Quafu Haituo | [IBM Transactions for Anti Money Laundering](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) · [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/) |
| PaySim | Mẫu giao dịch tiền di động dùng cho dashboard cục bộ | [Synthetic Financial Datasets for Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| Elliptic Bitcoin Dataset | Bộ dữ liệu đồ thị trong kho nghiên cứu | [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) · [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) |
| Bank Account Fraud Dataset Suite | Các bộ dữ liệu gian lận mở tài khoản trong kho nghiên cứu | [Bank Account Fraud Dataset Suite](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) · [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |

## Bản trích AMLworld dùng để kiểm chứng QPU

Từ dữ liệu tổng hợp của IBM, Q-FraudGuard chọn các bản ghi xuất hiện trong 44 cửa sổ, gán mã giao dịch, ghép mã ngân hàng với mã tài khoản và bổ sung điểm rủi ro do mô hình chấm. Dòng đầu của CSV ghi nguồn, các thay đổi và giấy phép CDLA-Sharing-1.0. Khi phân phối lại bản trích, giữ thông tin này cùng liên kết tới điều khoản. [CSV ứng viên](../quafu_benchmark_candidates.csv) · [Điều khoản CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/)

Vị trí các ZIP, mẫu PaySim và cách tạo đầu vào cho một lần chạy mới được ghi trong [danh mục dữ liệu](../DATASETS.md).
