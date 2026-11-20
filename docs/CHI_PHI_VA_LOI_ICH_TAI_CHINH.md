# Chi phí triển khai và lợi ích tài chính của Q-FraudGuard

**Ngân sách năm đầu cho một kênh chuyển tiền là 822 triệu đồng nếu công ty đã có QPU dùng chung, hoặc 2,342 tỷ đồng nếu thuê riêng D-Wave Leap.** Trong kịch bản ngăn thêm 50 vụ mức 40 triệu và 10 vụ mức 500 triệu trong 10 tháng vận hành, lợi ích ròng của toàn sản phẩm lần lượt là **1,923 tỷ đồng** và **403 triệu đồng**. Đây là dự toán theo số vụ ngăn thêm, phần tổn thất ngân hàng thực chịu và chi phí máy. Có thể thay từng đầu vào tại trang *Ca nghiệp vụ*. [Bảng tính tài chính](MO_HINH_TAI_CHINH.xlsx)

Trong phương án dùng QPU, mô hình sàng lọc tạo cảnh báo và QUBO chọn hồ sơ theo số suất điều tra. Nhật ký cung cấp mã hồ sơ, quyết định và thời điểm can thiệp. Ngân hàng đối soát các mã này với tổn thất, khoản thu hồi và thời gian xử lý để tính giá trị tài chính. [Quy trình điều tra](THIET_KE_DASHBOARD.md) · [Nhật ký qua Core Banking](CORE_BANKING.md#phản-hồi-xác-minh-và-nhật-ký)

## Bằng chứng từ chính sản phẩm

Hai cấu hình sàng lọc dùng cùng kỳ huấn luyện và cùng 1.015.448 giao dịch kiểm tra AMLworld tổng hợp, trong đó có 1.143 giao dịch mang nhãn rửa tiền. Trong bảng, cấu hình cơ sở dùng nhóm đặc trưng ban đầu. Cấu hình được chọn dùng thêm thông tin tài khoản từng được xác nhận và bộ tham số huấn luyện tương ứng. [Báo cáo thử nghiệm](../artifacts/training/amlworld_full_report.json) · [Cách tính các mốc](../scripts/train_amlworld_full.py)

| Phép đối chiếu trên cùng dữ liệu | Mô hình cơ sở | Mô hình được chọn | Ý nghĩa cho đội điều tra |
|---|---:|---:|---|
| Số giao dịch cần gắn cờ để đạt khoảng 95% recall, xác định hồi cứu bằng nhãn kiểm tra | 160.747 | 127.411 | Giảm 33.336 cảnh báo, tương đương 20,7% so với cơ sở. |
| Số giao dịch mang nhãn rửa tiền trong cùng 40.618 vị trí đầu theo điểm (4% lưu lượng) | 956 | 975 | Thêm 19 giao dịch mang nhãn ở cùng số lượt xem xét. |

Hai dòng dùng cùng điểm sàng lọc. Ở bước phân bổ suất điều tra, Quafu Haituo trả về mẫu đạt tối ưu QUBO trên **44/44 cửa sổ**. Đây là kết quả kỹ thuật trên chip. Lợi ích bằng tiền chỉ được ghi nhận khi đối soát được vụ can thiệp tăng thêm so với quy trình hiện hành. Giá thuê D-Wave là đầu vào cho phương án ngân sách. [Phép đo phần cứng](TRIEN_KHAI_QPU.md) · [Đối chiếu nhãn theo suất điều tra](../data/DATASETS.md#kiểm-lại-44-cửa-sổ-đã-chạy-trên-chip)

## Bằng chứng thực tế về các khoản có thể tiết kiệm

| Cơ chế tạo giá trị | Quan sát đã công bố | Cách sử dụng khi đánh giá sản phẩm |
|---|---|---|
| Cảnh báo còn kịp để can thiệp | Đến 29/6/2026, SIMO đã cảnh báo khoảng 4,6 triệu lượt khách hàng tại Việt Nam. Hơn 1,5 triệu lượt tạm dừng hoặc hủy giao dịch, với tổng giá trị hơn 5.200 tỷ đồng. | Ghi riêng giá trị giao dịch tạm dừng và tổn thất ròng ngân hàng xác nhận tránh được. [NHNN tại họp báo ngày 2/7/2026](https://baochinhphu.vn/nhnn-chan-lua-dao-hon-5200-ty-dong-canh-bao-46-trieu-giao-dich-nghi-ngo-10226070218174313.htm) |
| Sàng lọc tốt hơn giúp chuyên viên tập trung | HSBC công bố hệ thống đánh giá rủi ro của họ phát hiện lượng tội phạm tài chính cao gấp 2–4 lần trước đây, đồng thời giảm 60% số ca cảnh báo sai. | Đo phút điều tra và số lần liên hệ khách hàng ở cùng mức bao phủ. Các tỷ lệ của HSBC là kết quả của riêng ngân hàng đó. [HSBC, tháng 6/2024](https://www.hsbc.com/news-and-views/views/hsbc-views/harnessing-the-power-of-ai-to-fight-financial-crime) |
| Một phần tổn thất có thể trở thành khoản ngân hàng phải chi | Tại Anh, trong 12 tháng đầu của cơ chế hoàn trả đối với lừa đảo chuyển tiền được người dùng xác nhận, 173 triệu GBP đã được hoàn cho nạn nhân, bằng 88% số tiền mất thuộc phạm vi cơ chế. | Dùng khoản bồi hoàn, phân chia trách nhiệm và số thu hồi thực tế của ngân hàng Việt Nam để tính phần ngân hàng chịu. [Cơ quan quản lý hệ thống thanh toán Anh, tháng 2/2026](https://www.psr.org.uk/media/b50p3mlm/psr-2024-app-scam-performance-data-february-2026.pdf) |
| Sự cố có thể làm mất khách hàng | Trong khảo sát tại Mỹ với 2.209 người mà bản thân hoặc hộ gia đình đã mất tiền do lừa đảo, 19% cho biết họ đã đổi tổ chức tài chính vì vụ việc. | Đo chênh lệch tỷ lệ rời bỏ ở chính ngân hàng và dùng lãi đóng góp trên mỗi khách. Mô hình uy tín bên dưới dùng giả định riêng. [PYMNTS Intelligence và Featurespace, hình 6](https://www.pymnts.com/wp-content/uploads/2024/10/PYMNTS-The-Impact-of-Financial-Scams-on-Consumers-Finances-and-Banking-Habits-October-2024.pdf#page=14) |
| Thử nghiệm lượng tử với dữ liệu giao dịch ngân hàng | Lloyds Banking Group và IBM đã thử trong 9 tháng, phân tích quan hệ giữa các tài khoản trung chuyển trên dữ liệu giao dịch thực đã ẩn danh, dùng máy lượng tử IBM. | Đây là tiền lệ để thử hồ sơ có liên kết trên QPU. Hiệu quả tài chính của Q-FraudGuard vẫn cần đo trên các quyết định của ngân hàng triển khai. [Lloyds Banking Group](https://www.lloydsbankinggroup.com/insights/exploring-quantum-computing-at-lloyds-banking-group.html) |

## Tần suất lừa đảo tại các ngân hàng

### Những vụ ngân hàng đã can thiệp

| Ngân hàng và kỳ quan sát | Số vụ hoặc lượt được ghi nhận | Bình quân trong kỳ | Giá trị giao dịch và nguồn |
|---|---|---:|---|
| Agribank, toàn hệ thống, năm 2025 | Nhân viên ngăn gần 60 vụ lừa đảo qua mạng | Gần 5 vụ/tháng | Bảo vệ hơn **17 tỷ đồng** cho khách hàng. [Agribank, tổng kết năm 2025](https://www.agribank.com.vn/vn/ve-agribank/tin-tuc-su-kien/tai-chinh-ngan-hang/agribank-la-chan-tin-cay-giup-khach-hang-bao-toan-hang-chuc-ty-dong-truoc-van-nan-lua-dao-so) |
| Agribank Thanh Hóa và Yên Bái, năm 2025 | Thanh Hóa phát hiện 4 vụ, Yên Bái 5 vụ. Cả hai chi nhánh thuộc dòng toàn hệ thống ở trên. | Thanh Hóa 1 vụ/3 tháng, Yên Bái 1 vụ/2,4 tháng | Gần **1,85 tỷ đồng** tại Thanh Hóa và gần **700 triệu đồng** tại Yên Bái. [Agribank, số liệu chi nhánh](https://www.agribank.com.vn/vn/ve-agribank/tin-tuc-su-kien/tai-chinh-ngan-hang/agribank-la-chan-tin-cay-giup-khach-hang-bao-toan-hang-chuc-ty-dong-truoc-van-nan-lua-dao-so) |
| BIDV, lũy kế đến hết năm 2025 | Hơn **143.000 lượt** khách hàng dừng chuyển tiền đến tài khoản nghi lừa đảo | Nguồn chỉ công bố số lũy kế | Giá trị các giao dịch dừng lên tới **644 tỷ đồng**. Chỉ tiêu này đo phản ứng của khách sau cảnh báo. [BIDV, Báo cáo phát triển bền vững 2025, trang PDF 81](https://bidv.com.vn/wps/wcm/connect/fefc0922-870d-4585-9762-0b572c91106f/B%C3%A1o%2Bc%C3%A1o%2BPTBV%2BBIDV%2B2025_F.pdf?MOD=AJPERES#page=81) |
| DBS/POSB, các chi nhánh tại Singapore, năm 2025 | Nhân viên ngăn khoảng **550 vụ** khách suýt bị lừa | Khoảng **46 vụ/tháng** | Tránh khoản mất tiềm tàng **21,8 triệu SGD** cho khách. [DBS/POSB](https://www.dbs.com/newsroom/Seniors_increasingly_embrace_digital_banking_while_continuing_to_visit_branches_DBSPOSB_enhances_both_experiences) |
| DBS, OCBC, Standard Chartered và UOB cùng Cảnh sát Singapore, 1/3–30/4/2025 | Chiến dịch liên hệ hơn 3.500 khách có nguy cơ và ngăn hơn **1.100 vụ** | Hơn **550 vụ/tháng** trong hai tháng của cả chiến dịch | Ước tránh **51,7 triệu SGD** tổn thất. DBS có mặt ở cả hai dòng Singapore, nên mỗi dòng được đọc riêng. [Cảnh sát Singapore, thông báo kết quả chiến dịch](https://www.police.gov.sg/media-hub/news/2025/05/20250509_joint_police_bank_operation_safeguarded_savings) |

### Số lệnh chuyển tiền gian lận trên một triệu lệnh

Tại Anh, cơ quan quản lý hệ thống thanh toán (PSR) công bố cùng một mẫu số cho 14 nhóm ngân hàng. Họ đếm các lệnh Faster Payments mà khách hàng bị lừa tự xác nhận chuyển tiền (APP) trên **mỗi một triệu lệnh khách hàng gửi đi**. Báo cáo xét các giao dịch trước ngày 7/10/2024 có hồ sơ khép lại trong năm 2024. [PSR, phạm vi thống kê và biểu đồ trang 10](https://www.psr.org.uk/media/b50p3mlm/psr-2024-app-scam-performance-data-february-2026.pdf#page=10)

| Ngân hàng hoặc mốc đối chiếu tại Anh | Lệnh APP gian lận / 1 triệu lệnh gửi đi |
|---|---:|
| HSBC / First Direct | 61 |
| Barclays | 84 |
| Lloyds / Bank of Scotland / Halifax | 92 |
| Nationwide | 103 |
| Starling | 131 |
| **Trung vị của 14 nhóm, mỗi nhóm có trọng số như nhau** | **89,5** |
| Khoảng quan sát trên đủ 14 nhóm | 30–131 |

Số lệnh và số vụ có thể chênh lệch đáng kể. Năm 2025, UK Finance ghi nhận **519.433 lệnh thanh toán trong 248.070 vụ APP**, bình quân **2,09 lệnh/vụ**. Tại Việt Nam, trong một vụ mạo danh người thân tại Đà Nẵng, một nạn nhân chuyển **680 triệu đồng qua hai lệnh đến cùng một tài khoản**. Đây là tình huống cụ thể để thử đồ thị giao dịch và chức năng ưu tiên các hồ sơ chung tài khoản. Khi đối soát tiền tránh mất, hai lệnh của nạn nhân thuộc cùng một vụ. [UK Finance, báo cáo gian lận 2026, trang 33](https://www.ukfinance.org.uk/system/files/2026-06/UK%20Finance%20Fraud%20Report%202026.pdf#page=33) · [Bộ Công an, chuyên án tại Đà Nẵng](https://bocongan.gov.vn/bai-viet/bat-doi-tuong-lua-dao-chiem-doat-hang-ty-dong-bang-thu-doan-mao-danh-nguoi-than-lua-doi-tien-d22-t45599)

### Những quan sát có ý nghĩa với dự toán

- **Can thiệp trước khi tiền rời tài khoản.** Ngày 2/12/2025, tại TPBank Quế Võ, nhân viên ngân hàng và Công an Bắc Ninh kịp ngăn một khách chuyển **450 triệu đồng** cho người mạo danh công an. Nếu sản phẩm giúp ngân hàng thí điểm ngăn thêm một vụ cùng mệnh giá, với giả định thu hồi 25%, ngân hàng chịu 50% phần gốc còn mất và tránh được 1 triệu đồng tra soát, khoản chi tránh được sẽ là **450 × (1 − 25%) × 50% + 1 = 169,75 triệu đồng**. TPBank công bố giá trị khách hàng giữ được. Các tỷ lệ để tính phần ngân hàng hưởng là giả định trong dự toán. [TPBank, công bố ngày 5/12/2025](https://tpb.vn/tin-tuc/tin-tpbank/kip-thoi-ngan-nguoi-dan-chuyen-450-trieu-dong-cho-doi-tuong-lua-dao)
- **Tổn thất tập trung ở một số vụ lớn.** Theo Cảnh sát Singapore, năm 2024 các vụ mất từ **100.000 SGD** chiếm **3,3%** số vụ lừa đảo trên toàn quốc và **70,8%** số tiền mất. Một trường hợp ngân hàng thực chi là chiến dịch giả mạo OCBC cuối năm 2021. Có 790 nạn nhân mất **13,7 triệu SGD**, khoảng 80% số tiền mất tập trung trong tám ngày từ 23 đến 30/12. OCBC xác nhận đã hoàn tất chi trả thiện chí toàn bộ cho các nạn nhân. Hai quan sát này giúp xác định giá trị của thời gian phản ứng và lý do cần đo cả số tiền cùng với số vụ. [Cảnh sát Singapore, báo cáo thường niên 2024](https://www.police.gov.sg/-/media/SPF/Files/Publications/PDF/SPF-Annual-Report-2024.pdf) · [OCBC, thông báo ngày 30/1/2022](https://www.ocbc.com/group/media/release/2022/media-statement-30-jan-phishing-scam.page)
- **Mốc ra quyết định cho ngân hàng triển khai.** Với một triệu giao dịch mỗi tháng, 10 tháng vận hành và 60 triệu đồng chi nhân sự thực giảm, phương án QPU dùng chung hòa vốn nếu toàn sản phẩm giúp ngăn thêm **48 vụ mức 40 triệu**, bình quân **4,8 vụ/tháng**. Thuê riêng Leap cần **143 vụ** cùng mức, bình quân **14,3 vụ/tháng**. Nếu đã ngăn thêm 50 vụ mức 40 triệu, ngưỡng thuê riêng còn **8 vụ mức 500 triệu trong 10 tháng**. Khi thí điểm, ngân hàng đếm sự cố riêng còn lọt qua các kiểm soát hiện tại trên chính kênh đó và xác nhận phần tiền mình thực chịu. Con số 40 vụ còn lọt/tháng trong dự toán là giả định cần kiểm tra. Trang *Tần suất thực tế* trong bảng tính có số của đủ 14 nhóm ngân hàng Anh và tính lại các mốc hòa vốn theo ngân sách đã nhập. [Bảng tính tài chính](MO_HINH_TAI_CHINH.xlsx)

## Ngân sách triển khai QPU cho một kênh chuyển tiền

Kỳ dự toán gồm **2 tháng tích hợp và 10 tháng vận hành trong năm đầu**, trước thuế. Quy mô để tính là 1 triệu giao dịch và 1.200 hồ sơ chuyên viên rà soát mỗi tháng vận hành. Ngân hàng đã có nền tảng dữ liệu, danh tính, trực vận hành và hệ thống phê duyệt. Phí máy hoặc phần phân bổ được tính **đủ 12 tháng từ đầu dự án**, kể cả thời gian tích hợp. Các đơn giá nhân sự là giả định chi phí đầy đủ. Để đối chiếu, ITviec ghi nhận lương trung vị là 37,8 triệu đồng/tháng cho lập trình viên back-end và 41,3 triệu cho kỹ sư dữ liệu tại Việt Nam. [Khảo sát ITviec 2025–2026](https://itviec.com/bao-cao/luong-it-va-thi-truong-tuyen-dung-it-vietnam)

| Khoản chi, đơn vị triệu đồng | Cách lập dự toán | Chi phí |
|---|---|---:|
| Tích hợp nền tảng một lần | Khảo sát 12 ngày × 2, kết nối 30 × 2,4, phân quyền và phục hồi 20 × 2,5, nghiệm thu 12 × 2, đào tạo 5 × 2. Cộng dự phòng 20%. | 216 |
| Tích hợp QPU một lần | 12 ngày × 2,5 triệu và dự phòng 20%. Gồm kết nối máy, ánh xạ QUBO, kiểm tra nghiệm, thời hạn trả kết quả và xử lý khi tác vụ lỗi. | 36 |
| Vận hành nền tảng mỗi tháng hoạt động | Hạ tầng dùng chung 5, hỗ trợ kỹ thuật 10, kiểm tra mô hình 4, liên hệ và xác minh phát sinh 3 (200 lượt × 15.000 đồng, gồm cả cảnh báo nhầm). | 22 |
| Vận hành phần mềm kết nối QPU mỗi tháng hoạt động | 2 ngày kỹ sư × 2,5 triệu để giám sát hạn mức, đối soát tác vụ chạy lại và chất lượng nghiệm. | 5 |
| **Năm đầu trước phí máy** | **252 triệu đầu tư một lần + 10 tháng × 27 triệu** | **522** |

Mốc tham khảo cho máy chủ ứng dụng thử nghiệm là gói Linux Lightsail 2 vCPU, RAM 8 GB, IPv4 công cộng giá 44 USD/tháng. Với 100 GB bản sao tính phí ở 0,05 USD/GB-tháng, tổng là 49 USD. Theo tỷ giá lập ngân sách **26.000 VND/USD**, số này tương đương 1,274 triệu đồng/tháng. Phần 5 triệu đồng hạ tầng trong bảng còn gồm lưu trữ nghiệp vụ và giám sát. [Bảng giá AWS Lightsail](https://aws.amazon.com/lightsail/pricing/)

| Hình thức sử dụng QPU thật | Phí máy cho 12 tháng | Tổng chi năm đầu | Chi một năm vận hành đủ 12 tháng từ năm hai |
|---|---:|---:|---:|
| Máy đặt tại công ty, **đã có và dùng chung** | 300 triệu phân bổ | **822 triệu** | **624 triệu** |
| Thuê riêng **D-Wave Leap Access** | 70.000 USD × 26.000 = 1.820 triệu | **2.342 triệu** | **2.144 triệu** |

**Máy công ty.** Ví dụ dùng nguyên giá đã lắp đặt 240 tỷ đồng, phân bổ trong 5 năm, cùng 12 tỷ đồng/năm cho bảo trì, điện, làm lạnh, mặt bằng và đội vận hành máy. Tổng chi phí sở hữu ước 60 tỷ đồng/năm. Dự án sử dụng 0,5% công suất tính phí, tương ứng 300 triệu đồng/năm. Đây là các đầu vào giả định cần thay bằng sổ tài sản, hợp đồng bảo trì và số liệu công suất. 822 triệu là chi phí phân bổ cho dự án. **Nếu mua máy mới chỉ để phục vụ Q-FraudGuard**, cùng bộ giả định cho ngân sách tiền phải chi năm đầu **252 tỷ 522 triệu đồng** (240 tỷ mua máy + 12 tỷ vận hành máy + 522 triệu triển khai ứng dụng). Một tiền lệ thương mại là cam kết triển khai Advantage2 trị giá 10 triệu EUR của SQT, có quyền chọn mua. Giá mua cụ thể cần báo giá cho cấu hình của công ty. [Công bố của D-Wave về thỏa thuận SQT](https://ir.dwavequantum.com/news/news-details/2025/Swiss-Quantum-Technology-SA-Signs-10M-Agreement-to-Deploy-D-Wave-Advantage2-Annealing-Quantum-Computer/default.aspx)

**Thuê riêng.** AWS Marketplace niêm yết gói Leap Access 70.000 USD/12 tháng khi tra ngày 20/11/2026 và cho phép chào giá riêng. Dự toán dành toàn bộ phí hợp đồng cho dự án. Ô phí bổ sung mặc định bằng 0 theo giả định hợp đồng đủ hạn mức QPU vật lý cho số cửa sổ, số lần đo và lần chạy lại. Cần xác nhận hạn mức và phụ phí với nhà cung cấp. [Niêm yết D-Wave Leap Access](https://aws.amazon.com/marketplace/pp/prodview-swm25ss53l2ss)

**Công suất phân bổ.** Với trần 8.760 giờ khả dụng/năm, tỷ lệ 0,5% tương ứng 43,8 giờ. Nếu 12,78% trong 10 triệu giao dịch trở thành cảnh báo và mỗi cửa sổ gom 18 cảnh báo, dự án cần xử lý khoảng 71.000 cửa sổ. Phân bổ toàn bộ 43,8 giờ cho vận hành cho ngân sách **2,22 giây truy cập QPU/cửa sổ**, gồm các lượt chạy lại. Khi lập hợp đồng, trừ giờ dành cho tích hợp và đối chiếu thời gian chiếm dụng tính phí với nhu cầu giờ cao điểm. [Tỷ lệ cảnh báo AMLworld](../artifacts/training/amlworld_full_operating_points.json) · [Cách D-Wave đo thời gian truy cập QPU](https://docs.dwavequantum.com/en/latest/quantum_research/operation_timing.html)

## Nếu một vụ lọt qua, ngân hàng mất bao nhiêu

Ví dụ dưới đây xét **một vụ gian lận thanh toán thực sự gây mất tiền**. Giả định thu hồi được 25% giá trị giao dịch, ngân hàng chịu 50% phần còn lại và chi thêm 1 triệu đồng/vụ cho dịch vụ tra soát bên ngoài sau tổn thất. Phần chi này chỉ được tiết kiệm nếu can thiệp thành công giúp tránh khoản phải trả. Nhãn rửa tiền trong AMLworld phản ánh hoạt động đáng ngờ. Việc quy đổi một vụ thành tiền đòi hỏi chứng từ tổn thất riêng.

| Giá trị chuyển (triệu đồng) | Thu hồi | Còn mất sau thu hồi | Phần gốc ngân hàng chịu | Phần gốc khách hàng hoặc bên khác chịu | Tổng chi phí của ngân hàng kể cả tra soát |
|---:|---:|---:|---:|---:|---:|
| 10 | 2,5 | 7,5 | 3,75 | 3,75 | 4,75 |
| **40** | **10** | **30** | **15** | **15** | **16** |
| 500 | 125 | 375 | 187,5 | 187,5 | 188,5 |
| 3.200 | 800 | 2.400 | 1.200 | 1.200 | 1.201 |

Với vụ 40 triệu, can thiệp kịp thời có giá trị **16 triệu đồng cho ngân hàng** theo giả định trên. Nếu ngân hàng không phải chịu phần gốc, giá trị trực tiếp của cùng vụ chỉ còn **1 triệu đồng** chi tra soát có thể tránh. Để thấy quy mô một vụ lớn ngoài thực tế, Agribank Quế Võ ghi nhận giao dịch viên đã giúp khách dừng chuyển 500 triệu đồng ngày 15/12/2025. Con số 188,5 triệu trong bảng là phép tính tình huống của báo cáo này. [Trường hợp Agribank Quế Võ](https://www.agribank.com.vn/vn/ve-agribank/tin-tuc/dtl/!ut/p/z0/fY3JCgIxDIafZo6hdRvxKCIuB1FE0F4kdkonjqQz0hZ9e6Pi1UvIx78po47KMGbyGCkw3oRPpjz3e_PhcrjX28VoPtG7crw4bGbbgV711FqZ_wZpoGvXmakyNnB0j6iO6C_ITaGzA_R3-kIkhphsoS1aeENTo1AVGDJBrJFreRwX-pcBWxPwR-iSgxzgIlEm4YZaiQSRPbL45OQEt4RQYRBOTydr0gYjrSHeySWQJa_axpxeDNF2aw!!/)

## Giá trị ròng theo kịch bản can thiệp

Mỗi vụ trong bảng là một sự cố được ngăn thêm so với các kiểm soát hiện có. Nhiều giao dịch thuộc cùng vụ chỉ tính một lần. Mô hình giả định 40 vụ còn lọt mỗi tháng và 1.200 hồ sơ được rà soát mỗi tháng. Nếu thời gian mỗi hồ sơ giảm từ 12 xuống 9 phút, đội điều tra có thêm 60 giờ/tháng. Với 250.000 đồng/giờ và 40% số giờ giúp giảm tiền ngoài giờ hoặc thuê ngoài, tổng chi nhân sự tránh được trong 10 tháng là **60 triệu đồng**.

Mệnh giá trong các kịch bản dùng để định giá sau can thiệp. Hàm QUBO hiện chấm theo điểm rủi ro và liên kết chung tài khoản. Để đạt kịch bản B hoặc C, giao dịch giá trị lớn phải qua sàng lọc và xuất hiện trong danh sách điều tra đủ sớm để ngân hàng hành động. [Hàm mục tiêu hiện hành](../src/qf_guard/prioritization.py)

| Tình huống độc lập, trong 10 tháng | Vụ ngăn thêm so với hiện tại | Tổng lợi ích trước chi phí | Ròng khi dùng máy công ty | Ròng khi thuê riêng Leap |
|---|---|---:|---:|---:|
| Mốc đối chiếu, 2 vụ bán lẻ/tháng | 20 vụ × 40 triệu | 380 triệu | −442 triệu | −1.962 triệu |
| **A. Chuyển tiền bán lẻ** | **60 vụ × 40 triệu** | **1.020 triệu** | **+198 triệu** | −1.322 triệu |
| **B. Điều tra các giao dịch có chung tài khoản** | **50 vụ × 40 triệu và 10 vụ × 500 triệu** | **2.745 triệu** | **+1.923 triệu** | **+403 triệu** |
| **C. Ca trực có giao dịch đặc biệt lớn** | **20 vụ × 40 triệu và 2 vụ × 3,2 tỷ** | **2.782 triệu** | **+1.960 triệu** | **+440 triệu** |

**A. Bán lẻ.** Mốc hòa vốn với máy công ty là **48 vụ** trong 10 tháng, bằng cách lấy (822 − 60) / 16 rồi làm tròn lên. Nếu toàn bộ giờ tiết kiệm được dùng để nhận thêm việc, cần **52 vụ**. Với 60 vụ và có khoản giảm thực chi nhân sự, ROI năm đầu là **24,1%**. Ngân sách tối đa cho phần máy ở kịch bản này là **498 triệu đồng/năm**, tương đương 0,83% của nền tảng 60 tỷ đồng/năm. Nếu thuê riêng cả gói, cần ít nhất **143 vụ** mức 40 triệu trong 10 tháng để toàn sản phẩm hòa vốn khi có khoản tiết kiệm nhân sự.

**B. Nhiều giao dịch liên quan.** Khi nhiều cảnh báo chung tài khoản, QUBO có thể ưu tiên hồ sơ liên quan trong cùng cửa sổ. Chuyên viên đối chiếu dòng tiền để ngân hàng can thiệp kịp. Mười vụ lớn trong bảng là 10 sự cố độc lập. Phép tính là 50 × 16 + 10 × 188,5 + 60 = 2.745 triệu đồng. Ngay cả khi chịu toàn bộ phí Leap, sau 50 vụ thông thường chỉ cần **8 vụ 500 triệu** để hòa vốn. Với 10 vụ lớn, ROI thuê riêng là **17,2%**. Trường hợp Quế Võ cho thấy quy mô 500 triệu đồng có thể gặp trong thực tế. [Trường hợp Agribank Quế Võ](https://www.agribank.com.vn/vn/ve-agribank/tin-tuc/dtl/!ut/p/z0/fY3JCgIxDIafZo6hdRvxKCIuB1FE0F4kdkonjqQz0hZ9e6Pi1UvIx78po47KMGbyGCkw3oRPpjz3e_PhcrjX28VoPtG7crw4bGbbgV711FqZ_wZpoGvXmakyNnB0j6iO6C_ITaGzA_R3-kIkhphsoS1aeENTo1AVGDJBrJFreRwX-pcBWxPwR-iSgxzgIlEm4YZaiQSRPbL45OQEt4RQYRBOTydr0gYjrSHeySWQJa_axpxeDNF2aw!!/)

**C. Giao dịch đặc biệt lớn.** Tại Hàm Thuận Bắc, Agribank ghi nhận cán bộ chi nhánh và công an đã giúp khách tránh một vụ lừa chuyển 3,2 tỷ đồng ngày 3/5/2024. Theo cơ cấu bồi hoàn giả định của mô hình, mỗi vụ tương tự có giá trị 1.201 triệu đồng đối với ngân hàng. Sau 20 vụ thông thường, **1 vụ đặc biệt** đủ bù tổng chi máy chung, **2 vụ** đủ bù tổng chi thuê riêng. Tần suất hai vụ là tình huống để tính ngưỡng. Việc nhận diện và ngăn chặn vụ được dẫn nguồn do nhân sự ngân hàng và cơ quan chức năng thực hiện. [Trường hợp Agribank Hàm Thuận Bắc](https://www.agribank.com.vn/vn/ve-agribank/tin-tuc/dtl?current=true&urile=wcm%3Apath%3A%2Fagbank%2Fve-agribank%2Ftin-tuc%2Fcac-tin-khac%2Fdon-vi-thanh-vien%2Fagribank-chi-nhanh-ham-thuan-bac-binh-thuan-ngan-chan-mot-vu-lua-dao-3-2-ty-dong)

**Tỷ lệ bồi hoàn tại điểm hòa vốn.** Ở kịch bản A dùng máy chung, ngân hàng phải chịu ít nhất **39%** phần gốc còn mất sau thu hồi để đạt hòa vốn. Ở kịch bản B thuê riêng, mốc tương ứng là **42,3%**. Khi lập ngân sách, thay giả định 50% bằng tỷ lệ thực từ sổ bồi hoàn của từng nhóm vụ. [Phân tích độ nhạy trong bảng tính](MO_HINH_TAI_CHINH.xlsx)

## Uy tín và giá trị khách hàng

Một biểu hiện có thể định giá của mất uy tín là khách rời ngân hàng sau một sự cố được nhiều người biết. Ví dụ độc lập cho một năm vận hành đủ 12 tháng, giả sử tối đa một sự cố như vậy. Có 15.000 khách khác bị ảnh hưởng về niềm tin, ngoài những người đã được hạch toán thiệt hại trực tiếp. Nếu sự cố xảy ra, giả sử tỷ lệ rời bỏ tăng **2 điểm phần trăm** so với nhóm tương đồng, lãi đóng góp sau chi phí phục vụ là **1,2 triệu đồng/khách/năm**. Chênh lệch xác suất dưới đây là giả định cần kiểm tra với bộ phận quản trị rủi ro.

| Chỉ số | Quy trình hiện tại | Giả định có Q-FraudGuard | Giá trị kỳ vọng bảo toàn |
|---|---:|---:|---:|
| Xác suất có sự cố ảnh hưởng uy tín trong năm | 8% | 5% | Giảm 3 điểm phần trăm |
| Số khách mất thêm nếu sự cố xảy ra | 300 | 300 | 9 khách theo kỳ vọng năm |
| Lãi đóng góp mất trong 12 tháng sau sự cố, tính theo xác suất | 28,8 triệu | 18 triệu | **10,8 triệu đồng** |
| Hiện giá 3 năm lãi đóng góp mất, nếu nhóm khách đó rời hẳn, chiết khấu 10%/năm | 71,6 triệu | 44,8 triệu | **26,9 triệu đồng** |

Riêng một sự cố đã xảy ra có thể làm mất **360 triệu đồng** lãi đóng góp trong 12 tháng theo giả định trên. Hai dòng giá trị là hai kỳ hạn định giá cùng nhóm khách, làm tròn độc lập. Báo cáo tách chúng khỏi ROI năm đầu. Để thay giả định bằng bằng chứng, đối chiếu tỷ lệ rời bỏ sau 90 và 180 ngày giữa các nhóm có lịch sử sử dụng tương đồng, kể cả khách từng bị cảnh báo nhầm. Dùng biên đóng góp từ sổ quản trị và thời gian khách thực sự ngừng sử dụng để ước tính phần giá trị doanh nghiệp bị ảnh hưởng. [Căn cứ khảo sát về việc đổi tổ chức tài chính](https://www.pymnts.com/wp-content/uploads/2024/10/PYMNTS-The-Impact-of-Financial-Scams-on-Consumers-Finances-and-Banking-Habits-October-2024.pdf#page=14)

## Giá trị QPU cần tạo thêm so với CPU

Nếu đã có cùng phần mềm sàng lọc, đồ thị, bàn điều tra và nhật ký, dùng CPU làm đối chứng giúp định giá riêng quyết định đầu tư phần cứng. Phiên bản CPU so sánh có chi phí 436 triệu đồng năm đầu. So trên **cùng hồ sơ, cùng số suất K, cùng thời hạn ngân hàng còn có thể can thiệp**, giữ cố định lợi ích nhân sự. Mỗi cột dưới xét một loại vụ riêng.

| Phương án | Chi phí tăng thêm so với cùng sản phẩm chạy CPU trong năm đầu | Vụ 40 triệu cần ngăn thêm | Vụ 500 triệu cần ngăn thêm | Vụ 3,2 tỷ cần ngăn thêm |
|---|---:|---:|---:|---:|
| Máy QPU dùng chung | 822 − 436 = **386 triệu** | 25 | **3** | 1 |
| Thuê riêng Leap | 2.342 − 436 = **1.906 triệu** | 120 | **11** | **2** |

Ví dụ, nếu trong cùng tập hồ sơ, QPU giúp chuyên viên can thiệp kịp **3 vụ 500 triệu nhiều hơn CPU**, lợi ích riêng của lựa chọn QPU dùng chung là **3 × 188,5 − 386 = 179,5 triệu đồng**. Nếu hai bộ giải giúp ngăn cùng số vụ, phần mềm vẫn có thể tạo giá trị so với quy trình cũ như bảng A–C, còn chi phí tăng thêm để chọn QPU vẫn là 386 triệu hoặc 1.906 triệu đồng. Phép thử hiện có mới xác nhận chất lượng nghiệm theo hàm QUBO. Để chứng minh chênh lệch bằng tiền, đối soát từng ca ngăn thêm với hồ sơ tổn thất đã xác nhận. [Bằng chứng phần cứng của Q-FraudGuard](TRIEN_KHAI_QPU.md)

## Đo những gì trong đợt thí điểm

1. **Đo mức hiện tại.** Chọn một kênh giao dịch và một khoảng theo dõi cố định. Ghép theo mã sự cố để lấy tiền chi trả thực, thu hồi sau cùng, hóa đơn tra soát, số phút xử lý và số liên hệ nhầm. Gộp các cảnh báo cùng vụ.
2. **Đo phần tăng thêm.** Giữ quy trình bảo vệ hiện hành cho mọi giao dịch. Chạy song song rồi triển khai so le theo nhóm hoặc ca tương đồng, giữ cùng năng lực điều tra. So tiền mất ròng trên một triệu giao dịch sau cùng một thời hạn thu hồi, gồm cả vụ bị bỏ lọt và chi phí do cảnh báo nhầm. Với QPU, so riêng với CPU trên cùng cửa sổ.
3. **Đối soát giá trị.** Ngân hàng xác nhận kết quả can thiệp và tổng hợp chi phí từ hệ thống của mình. Nhật ký Q-FraudGuard cung cấp mã hồ sơ và mốc thời gian để ghép. Chỉ tính phần tiền và giờ thực trả tránh được một lần. Báo riêng giá trị bảo vệ cho khách hàng, giờ công tái phân bổ và tác động giữ chân khách. [Phạm vi nhật ký](THIET_KE_DASHBOARD.md)
