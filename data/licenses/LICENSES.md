# Nguồn và giấy phép dữ liệu, mô hình và kết quả

Bảng dưới ghi nguồn phát hành và giấy phép của từng bộ dữ liệu. Mã nguồn Q-FraudGuard dùng giấy phép MIT. [Giấy phép mã nguồn](../../LICENSE) · [Thành phần bên thứ ba](../../THIRD_PARTY_NOTICES.md)

| Bộ dữ liệu và ghi công | Phạm vi sử dụng | Nguồn và giấy phép dữ liệu |
|---|---|---|
| IBM AMLworld HI-Small · IBM, Erik Altman và cộng sự | Huấn luyện mô hình chấm điểm, benchmark Quafu Haituo và phát lại | [IBM AML-Data](https://github.com/IBM/AML-Data) · [Bản phát hành Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) · [CDLA-Sharing-1.0, toàn văn](CDLA-Sharing-1.0.txt) |
| PaySim · Edgar López-Rojas, Ahmad Elmir và Stefan Axelsson | Huấn luyện mô hình và phát giao dịch tiền di động ở dashboard cục bộ | [Bản phát hành của Edgar López-Rojas](https://www.kaggle.com/datasets/ealaxi/paysim1) · [CC BY-SA 4.0, toàn văn](CC-BY-SA-4.0.txt) |
| Elliptic Bitcoin Dataset · Elliptic | Bộ dữ liệu đồ thị trong kho nghiên cứu cục bộ | [Bản phát hành của Elliptic](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) · [CC BY-NC-ND 4.0, toàn văn](CC-BY-NC-ND-4.0.txt) |
| Bank Account Fraud Dataset Suite · Sérgio Jesus và cộng sự, Feedzai | Bộ dữ liệu gian lận mở tài khoản trong kho nghiên cứu cục bộ | [Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) ghi [CC BY-NC-SA 4.0](CC-BY-NC-SA-4.0.txt). [Datasheet của Feedzai, Q44 và Author Statement](https://raw.githubusercontent.com/feedzai/bank-account-fraud/main/documents/datasheet.pdf) ghi [CC BY-NC-ND 4.0](CC-BY-NC-ND-4.0.txt). |

Giấy phép Elliptic giới hạn quyền được cấp ở mục đích phi thương mại và chỉ cho chia sẻ dữ liệu ở dạng chưa chỉnh sửa. Cả hai nguồn BAF đều ghi giới hạn phi thương mại và khác nhau về quyền phân phối bản đã chỉnh sửa. Xác nhận điều khoản áp dụng với Feedzai trước khi phân phối bản chỉnh sửa hoặc dùng dữ liệu BAF cho mục đích thương mại. Các ZIP gốc nằm trong `data/downloads/` cục bộ. [Mã SHA-256 của các bản tải](../downloads/SHA256SUMS.txt)

## Tệp dữ liệu phân phối cùng dự án

| Tệp | Thay đổi so với nguồn | Giấy phép khi phân phối dữ liệu |
|---|---|---|
| [`paysim.csv`](../paysim.csv) | Mẫu 99.416 giao dịch từ PaySim | [CC BY-SA 4.0](CC-BY-SA-4.0.txt) · [Ghi công đi kèm CSV](../paysim.csv.license) |
| [`quafu_benchmark_candidates.csv`](../quafu_benchmark_candidates.csv) | Chọn giao dịch thuộc 44 cửa sổ, gán mã giao dịch, ghép mã ngân hàng và tài khoản, thêm điểm rủi ro, giữ nhãn rửa tiền gốc | [CDLA-Sharing-1.0](CDLA-Sharing-1.0.txt) · Ghi công cũng có ở dòng đầu CSV. |
| [`quafu_dashboard_replay.csv`](../quafu_dashboard_replay.csv) | Chọn 64 giao dịch, chuẩn hóa trường, gán mã giao dịch, thêm điểm và cờ sàng lọc, giữ nhãn gốc | [CDLA-Sharing-1.0](CDLA-Sharing-1.0.txt) · Ghi công cũng có ở dòng đầu CSV. |

Các CSV, pickle điểm, chỉ mục phát lại và cơ sở dữ liệu xuất từ AMLworld có thể giữ các trường gốc. Khi phân phối các bản chứa dữ liệu IBM, áp dụng CDLA-Sharing-1.0 và giữ ghi công, nguồn và điều khoản. Khi chia sẻ bản trích hoặc bản chỉnh sửa PaySim, giữ ghi công, liên kết giấy phép và mô tả thay đổi. Chia sẻ bản chuyển thể chịu điều khoản ShareAlike của CC BY-SA 4.0. [CDLA-Sharing-1.0, điều 3](https://cdla.dev/sharing-1-0/) · [CC BY-SA 4.0, điều 3 và 4](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en) · [Cách tạo các tệp](../DATASETS.md)

## Mô hình đã lưu và kết quả đo

| Tệp | Dữ liệu lưu trong tệp | Ghi công và giấy phép của dữ liệu nhúng |
|---|---|---|
| [`paysim_lightgbm_scorer.joblib`](../../artifacts/models/paysim_lightgbm_scorer.joblib) | Lịch sử 79.532 giao dịch PaySim đã chuẩn hóa, cùng tham số mô hình | [CC BY-SA 4.0](CC-BY-SA-4.0.txt) cho phần dữ liệu · [Ghi công đi kèm mô hình](../../artifacts/models/paysim_lightgbm_scorer.joblib.license) |
| `amlworld_full_scorer.joblib` và các scorer AMLworld tạo cục bộ | Lịch sử AMLworld đã chuẩn hóa. Tệp `amlworld_full_scorer.joblib` của báo cáo mặc định lưu 4.061.789 giao dịch và danh sách thời điểm xác nhận tài khoản. | [CDLA-Sharing-1.0](CDLA-Sharing-1.0.txt) cho phần dữ liệu · [Ghi công đi kèm mô hình đầy đủ](../../artifacts/models/amlworld_full_scorer.joblib.license) |

Khi phân phối lại một tệp mô hình chứa lịch sử giao dịch, gửi kèm ghi công và giấy phép của lịch sử đó. Bản báo cáo chỉ chứa chỉ số tổng hợp được phát hành cùng tài liệu dự án. Bản ghi benchmark và phiên dashboard dùng mã giao dịch để nối với các bản trích có giấy phép ở trên. Các chuỗi bit và số đo phản ánh kết quả thí nghiệm, với điều khoản truy cập dịch vụ nêu trong danh mục bên thứ ba. [Benchmark Quafu](../../docs/TRIEN_KHAI_QPU.md) · [Điều khoản QPU](../../THIRD_PARTY_NOTICES.md#dịch-vụ-qpu-và-các-bản-triển-khai)
