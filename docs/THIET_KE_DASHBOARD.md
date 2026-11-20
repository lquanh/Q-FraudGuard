# Bàn điều tra Q-FraudGuard

Bàn điều tra giúp chuyên viên trả lời ba câu hỏi khi nhận cảnh báo. Giao dịch nào cần xem trước? Những tài khoản nào đã chuyển tiền cho nhau? Ai đã xử lý hồ sơ và dựa trên căn cứ nào? Hệ thống tự tạo ứng viên từ điểm sàng lọc. QUBO phân bổ số suất xem xét của ca trực theo Risk Score và liên kết chung tài khoản. [Logic sàng lọc](../src/qf_guard/scoring.py) · [Hàm mục tiêu ưu tiên](../src/qf_guard/prioritization.py)

## Thông tin trước khi ra quyết định

| Khu vực | Căn cứ chuyên viên có thể kiểm tra |
|---|---|
| **Bàn điều tra** | Mã và thời điểm giao dịch, số tiền và loại tiền, kênh thanh toán, hai tài khoản, Risk Score so với ngưỡng, lý do chuyển xử lý và phản hồi khách hàng gần nhất. |
| **Transaction Graph** | Mũi tên cho biết hướng tiền. Trỏ chuột, chạm hoặc dùng phím Tab trên giao dịch để xem mã, số tiền, thời điểm UTC và hai tài khoản. Chọn khoảng 1 giờ, 24 giờ hoặc 7 ngày. Nét đứt nối các hồ sơ tương đồng khi mở lớp cùng phiên. |
| **Giao dịch liên quan** | Đếm giao dịch trước hồ sơ đang xem trong cùng luồng và vòng dữ liệu. Xem tối đa 30 giao dịch gần nhất trong bảng chi tiết, mở từng giao dịch rồi quay về hồ sơ gốc. |
| **Risk Score Breakdown** | Xem các tín hiệu Supervised Learning, Anomaly Detection, Graph Score và Rule-based Score. Với hồ sơ được chấm bằng mô hình đồ thị theo thời gian, nhãn hiển thị là **TGNN Score**. |
| **Phiên ưu tiên** | Xem danh sách được chọn và đối chiếu với Top-K trên cùng nhóm ứng viên. Mở hồ sơ từ kết quả của phiên. |
| **Nhật ký** | Tra người thực hiện, thời điểm UTC, hành động và lý do. Mở lại hồ sơ hoặc phiên từ nhật ký. |

Liên kết chung tài khoản giúp chuyên viên tìm các giao dịch cần đọc cùng nhau. Kết luận nghiệp vụ dựa trên dòng tiền, chứng từ và kết quả xác minh của từng hồ sơ. [Truy vấn hồ sơ và lịch sử](../src/qf_guard/operations.py) · [Giải nghĩa điểm thành phần](../src/qf_guard/explainability.py)

## Xử lý hồ sơ trong luồng trực tiếp

| Sự kiện | Q-FraudGuard ghi nhận | Người quyết định |
|---|---|---|
| Ứng viên cần bổ sung thông tin | Mở yêu cầu bổ sung. Khi kênh ngân hàng gửi phản hồi hợp lệ, ghi nguồn, thời điểm và lý do, rồi đóng yêu cầu. | Ngân hàng quyết định việc tiếp tục giao dịch. |
| Cần xác minh thêm một lớp | Ghi kết quả xác minh của ngân hàng, đối chiếu tài khoản và phiên bản hồ sơ rồi cập nhật nhật ký. | Ngân hàng duyệt hoặc từ chối theo quy trình của mình. |
| Rủi ro cao hoặc tài khoản đang bị hạn chế | Chuyển vào **Cần chuyên viên quyết định** và thông báo trong ứng dụng. | Chuyên viên nhận hồ sơ, xem bằng chứng, ghi lý do và chọn hành động phù hợp. |
| Chuyên viên chọn đóng băng giao dịch, tạm ngưng hoặc khôi phục tài khoản | Ghi người thực hiện, đối tượng và trạng thái trong sổ hồ sơ. Các giao dịch mới từ tài khoản gửi đang tạm ngưng tiếp tục được chuyển xem xét. | Core Banking của ngân hàng phê duyệt và thực thi hành động trên tài khoản thật. |

Màn hình **Phản hồi khách hàng** cho phép diễn tập trên dữ liệu phát lại. Ở luồng trực tiếp, kênh ngân hàng gửi phản hồi qua `POST /v1/verification-receipts`. Mã biên nhận dùng để thử lại khi mất kết nối. Bên ngân hàng có thể đọc tiếp nhật ký theo con trỏ qua `GET /v1/case-events`. [Hợp đồng Core Banking](CORE_BANKING.md#phản-hồi-xác-minh-và-nhật-ký) · [Quy tắc chuyển trạng thái](../src/qf_guard/casework.py)

## Phiên minh họa để nghiệm thu

Phiên Haituo đã lưu gồm **64 giao dịch, 18 ứng viên và 4 hồ sơ được chọn**. Ba hồ sơ cùng gửi tiền từ tài khoản có đuôi **8660** lúc **16:10 UTC ngày 08/09/2022**. Chế độ `recorded` mở lại phiên từ dữ liệu đi kèm. Các bước sau dùng một sổ chưa có thao tác của chuyên viên. [Hướng dẫn khởi chạy](../README.md#chạy-phiên-minh-họa) · [Bản ghi phiên](../artifacts/deploy/quafu_dashboard_18_20261118T220101Z.json)

1. Mở **Bàn điều tra**. Có 1 hồ sơ đủ ngưỡng điều tra và 3 đề xuất bổ sung. Mở `AML-004060379` để xem hai giao dịch cùng tài khoản gửi trong nhóm được chọn.
2. Tiếp nhận đề xuất này và ghi lý do. Hàng chờ có 2 hồ sơ, còn 2 đề xuất. Kết quả chọn gốc của phiên vẫn tra cứu được.
3. Ghi quyết định yêu cầu xác minh, rồi mở **Nhật ký** để tra người thực hiện và lý do theo mã hồ sơ.
4. Mở **Phiên ưu tiên**. Đối chiếu bốn hồ sơ với bốn điểm cao nhất và mở lại các giao dịch khác nhau giữa hai danh sách. [Giao dịch và nhãn của phiên minh họa](../data/quafu_dashboard_replay.csv)

## Bảo trì giao diện

`dashboard/app.py` và `workspace.css` định nghĩa bàn điều tra. Bản xem trước HTML và bộ kết xuất từ sổ hồ sơ dùng chung một mẫu. Để sửa design token, cài công cụ bằng `npm ci --prefix dashboard/design`, chỉnh `tokens.json` rồi chạy `python dashboard/design/build_design_system.py` để xuất CSS. Lệnh `python dashboard/design/build_engraving.py` tạo lại họa tiết đầu trang. [Dashboard](../dashboard/app.py) · [Mẫu HTML](../dashboard/ui/index.html) · [Design token](../dashboard/design/design-system/q-fraudguard/tokens.json) · [Giấy phép UI/UX Pro Max](../dashboard/design/LICENSE-ui-ux-pro-max.txt) · [Giấy phép Taste Skill](../dashboard/design/LICENSE-taste-skill.txt)
