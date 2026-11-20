# Q-FraudGuard

Q-FraudGuard là công cụ sàng lọc giao dịch và ưu tiên hồ sơ điều tra cho ngân hàng. Khi một ca trực chỉ đủ nguồn lực xem `K` hồ sơ, hệ thống kết hợp điểm rủi ro với các liên kết chung tài khoản để chọn danh sách. Chuyên viên có thể lần theo dòng tiền, xem căn cứ của cảnh báo và ghi lại quyết định ngay trong hồ sơ.

**Nguyên mẫu Q-FraudGuard đã hoàn thiện và sẵn sàng tích hợp với hệ thống ngân hàng thực tế để triển khai thử nghiệm.**

**[Hướng dẫn cài đặt và chạy](HUONG_DAN_CHAY.md)** · [Quy trình của chuyên viên](docs/THIET_KE_DASHBOARD.md) · [Tích hợp Core Banking](docs/CORE_BANKING.md)

## Từ giao dịch đến quyết định

| Bước | Hệ thống xử lý | Kết quả |
|---|---|---|
| Sàng lọc | Mô hình tính Risk Score theo từng giao dịch. Cấu hình pilot có thể dùng TGNN để khai thác lịch sử chuyển tiền giữa các tài khoản. | Cảnh báo, điểm thành phần và lý do chuyển xử lý. |
| Ưu tiên | QUBO chọn đúng K hồ sơ trong một nhóm ứng viên theo điểm rủi ro và các cặp hồ sơ chung tài khoản. Bản host chạy trên CPU. Người vận hành có thể chuyển bước này sang QPU. | Danh sách xem xét theo ngân sách của ca trực. |
| Điều tra | Chuyên viên mở Transaction Graph, xem số tiền, thời điểm, chiều chuyển và các giao dịch liên quan trước khi ghi quyết định. | Nhật ký gắn với mã giao dịch, người xử lý, thời điểm và lý do. |
| Đối soát | Adapter Core Banking nhận giao dịch, ghi biên nhận xác minh và cung cấp nhật ký qua API. | Ngân hàng phê duyệt và thực thi việc tiếp tục, đóng băng giao dịch hoặc hạn chế tài khoản. |

## Kết quả đã kiểm chứng

| Phép thử | Kết quả | Phạm vi |
|---|---|---|
| Sàng lọc AMLworld | Với cùng 40.618 vị trí đầu theo điểm, mô hình tìm thấy **975** giao dịch mang nhãn rửa tiền, so với **956** của mô hình cơ sở. | 1.015.448 giao dịch kiểm tra tổng hợp. [Báo cáo mô hình](artifacts/training/amlworld_full_report.json) |
| TGNN | Với 5% trọng số cho TGNN, Average Precision tăng từ **0,4046 lên 0,4166**. Cả hai cấu hình giữ **157/168** giao dịch mang nhãn trong 12.445 cảnh báo đầu. | Thí nghiệm riêng trên 82.963 giao dịch kiểm tra. [Đối chứng TGNN](artifacts/training/amlworld_tgnn_pilot_comparison.json) |
| Ưu tiên trên chip Quafu Haituo | **44/44 cửa sổ** có ít nhất một mẫu chọn đúng K và đạt giá trị QUBO tối ưu. Heuristic CPU đổi chỗ cũng đạt tối ưu ở cả 44 cửa sổ. | 8–18 ứng viên mỗi cửa sổ, gồm 40 cửa sổ chọn trước và 4 cửa sổ mà Top-K bỏ lỡ nghiệm tối ưu. [Bản ghi và phép kiểm chứng](docs/TRIEN_KHAI_QPU.md) |

Bộ kết nối Core Banking qua HTTP v1 đã được kiểm thử trong sandbox với ba nghiệp vụ nhận giao dịch, ghi biên nhận xác minh và tra cứu nhật ký. Khi triển khai thử nghiệm, ngân hàng ánh xạ trường dữ liệu, cấp quyền kết nối và nghiệm thu theo tải thực tế. [Hợp đồng API](docs/CORE_BANKING.md) · [Biên bản chạy thử](minh_chung/phien_nghiep_vu.json)

## Chạy phiên minh họa

Bản demo đi kèm phiên Haituo đã lưu gồm 64 giao dịch, 18 ứng viên và 4 hồ sơ được chọn. Trên sổ mới, mở `AML-004060379` để xem hai giao dịch khác chung tài khoản gửi. Có thể chạy phiên này bằng dữ liệu trong repo mà không cần tài khoản QPU. [Cài đặt và mở demo](HUONG_DAN_CHAY.md#chạy-phiên-minh-họa)

### Tài khoản demo

Mở trang đăng nhập để vào bàn làm việc. Hai tài khoản có cùng quyền sử dụng bàn điều tra. Nhật ký ghi tên tài khoản đã thực hiện thao tác. [Đăng nhập](https://quanhlq.site/login) · [Bàn làm việc](https://quanhlq.site/dashboard)

| Tên đăng nhập | Mật khẩu |
|---|---|
| `chuyenvien1` | `1` |
| `admin` | `admin` |

Hai mật khẩu trên dành cho bản demo với dữ liệu tổng hợp. Có thể đặt `QFG_DASHBOARD_OPERATOR_PASSWORD` và `QFG_DASHBOARD_ADMIN_PASSWORD` trong môi trường khởi động để thay mật khẩu mặc định.

## Phát lặp toàn bộ AMLworld

Chế độ `loop` phát 5.078.345 giao dịch HI-Small theo thứ tự thời gian và quay lại đầu nguồn sau mỗi vòng. Chế độ `live` nhận sự kiện do bên tích hợp gửi vào. [Chạy luồng phát lặp](HUONG_DAN_CHAY.md#phát-lặp-toàn-bộ-amlworld) · [Chạy Core Banking sandbox](HUONG_DAN_CHAY.md#chạy-core-banking-sandbox)

## Tài liệu triển khai

- **Dữ liệu và mô hình** — nguồn AMLworld, PaySim, cách chia tập theo thời gian và artifact tương ứng. [Danh mục dữ liệu](data/DATASETS.md)
- **Ước tính thời gian QPU.** Với D-Wave Advantage2_system2, 1.000 lượt đọc cùng một bài đã nạp dự kiến cần **57,6–135,6 mili giây** cho bước lấy mẫu. Dự toán đầu cuối cần cộng thời gian dựng và nạp bài, truyền dữ liệu, chờ dịch vụ và xác minh danh sách. [Ước tính D-Wave](docs/UOC_TINH_THOI_GIAN_DWAVE.md)
- **Chi phí và lợi ích** — dự toán năm đầu, kịch bản can thiệp và ngưỡng hòa vốn theo phần tổn thất ngân hàng thực chịu. [Báo cáo tài chính](docs/CHI_PHI_VA_LOI_ICH_TAI_CHINH.md) · [Bảng tính](docs/MO_HINH_TAI_CHINH.xlsx)

## Giấy phép và ghi công

Mã nguồn và tài liệu do Q-FraudGuard viết dùng giấy phép MIT. Dữ liệu, mô hình và thành phần bên thứ ba có điều khoản riêng. [Giấy phép MIT](LICENSE) · [Thành phần bên thứ ba](THIRD_PARTY_NOTICES.md) · [Dữ liệu và mô hình](data/licenses/LICENSES.md)
