# Kịch bản thuyết trình demo Q-FraudGuard

Thời lượng đề xuất: 6–8 phút.

## Chuẩn bị trước khi trình bày

1. Mở `http://localhost:8765`.
2. Refresh trang một lần và chờ bảng bắt đầu cập nhật.
3. Kiểm tra panel cảnh báo bên phải đã xuất hiện.
4. Thu nhỏ panel trước khi bắt đầu để màn hình chính dễ quan sát.
5. Đặt mức zoom trình duyệt khoảng 90–100%.

## 1. Mở đầu — 30 giây

**Thao tác:** Mở trang Q-FraudGuard và chưa thao tác vào cảnh báo.

**Lời trình bày:**

> Đây là Q-FraudGuard, prototype hỗ trợ phát hiện và xử lý gian lận tài chính
> theo luồng thời gian thực. Hệ thống không chỉ chấm điểm từng giao dịch riêng
> lẻ mà còn phân tích quan hệ giữa các giao dịch, sau đó tối ưu hóa nhóm hồ sơ
> cần ưu tiên điều tra.
>
> Luồng xử lý chính gồm ba tầng: AI lọc và tính risk score, TGNN phân tích quan
> hệ trên graph thời gian, cuối cùng QUBO và Simulated Quantum Annealing lựa
> chọn nhóm ứng viên cần ưu tiên.

## 2. Giới thiệu luồng realtime — 45 giây

**Thao tác:** Chỉ vào trạng thái `Real-time processing`, sau đó chỉ vào các dòng
đang liên tục xuất hiện trong bảng.

**Lời trình bày:**

> Dashboard đang mô phỏng kết nối với một transaction stream. Trong demo,
> nguồn dữ liệu là phần test của PaySim. Hệ thống đọc giao dịch liên tục, tạo
> timestamp mới và chạy inference bằng model đã được huấn luyện offline.
>
> Khi đọc hết PaySim, con trỏ quay lại đầu phần test để duy trì luồng mô phỏng.
> Dữ liệu gốc không bị sửa và không có quá trình training nào diễn ra trên giao
> diện này.

## 3. Giải thích các chỉ số — 60 giây

**Thao tác:** Lần lượt chỉ vào bốn thẻ chỉ số phía trên.

**Lời trình bày:**

> Thẻ thứ nhất là tổng số giao dịch hệ thống đã xử lý trong phiên hiện tại.
>
> Thẻ thứ hai là số giao dịch vượt qua ngưỡng rủi ro và được giữ lại làm ứng
> viên. Đây là ngưỡng cố định được lựa chọn từ tập validation, không phải top
> một phần trăm của từng batch.
>
> Thẻ thứ ba là tỷ lệ fraud thực tế được phát hiện. Chỉ số này được tính bằng
> số giao dịch fraud có mặt trong tập ứng viên chia cho tổng số fraud thực tế
> đã đi qua luồng.
>
> Thẻ cuối là latency inference trung bình trên mỗi giao dịch.

## 4. Giải thích bảng giao dịch — 90 giây

**Thao tác:** Chỉ lần lượt vào các cột `Risk score`, `Ứng viên`, `SQA chọn` và
`Nhãn thật`.

**Lời trình bày:**

> Mỗi dòng là một giao dịch vừa đi qua hệ thống. Risk score tổng hợp bốn nguồn
> tín hiệu: xác suất từ LightGBM, điểm bất thường từ Isolation Forest, tín hiệu
> graph và điểm từ rule engine.
>
> Cột “Ứng viên” trả lời câu hỏi: giao dịch này có đủ rủi ro để chuyển sang
> hàng đợi xử lý hay không.
>
> Cột “SQA chọn” có ý nghĩa khác. SQA không quyết định một giao dịch có phải
> fraud hay không. Solver nhận tập ứng viên đã được AI lọc và lựa chọn một nhóm
> hồ sơ nên được ưu tiên trong cửa sổ hiện tại, dựa trên risk score, quan hệ
> graph và giới hạn nguồn lực điều tra.
>
> Vì vậy, một giao dịch có thể là ứng viên nhưng không được SQA chọn ở lượt
> hiện tại. Giao dịch đó vẫn nằm trong hàng đợi và vẫn phải được xử lý.
>
> “Nhãn thật” là ground truth của PaySim, chỉ được dùng trong demo để đánh giá
> khả năng phát hiện. Trong môi trường thực tế, nhãn này thường chỉ có sau quá
> trình xác minh hoặc điều tra.

## 5. Giải thích pipeline AI, TGNN và QUBO — 90 giây

**Thao tác:** Giữ màn hình ở bảng giao dịch; có thể chỉ vào một dòng ứng viên.

**Lời trình bày:**

> Ở tầng đầu tiên, hệ thống kiểm tra schema, chuẩn hóa dữ liệu và tạo đặc trưng
> giao dịch cùng đặc trưng hành vi dựa trên lịch sử trước thời điểm hiện tại.
> LightGBM, Isolation Forest và rule engine tạo các điểm thành phần để lọc ứng
> viên.
>
> Ở tầng thứ hai, TGNN xem giao dịch trong bối cảnh graph thời gian. Mục tiêu là
> nhận biết các mẫu quan hệ như nhiều tài khoản chuyển vào một điểm, một tài
> khoản phân tán tiền sang nhiều nơi hoặc chuỗi chuyển tiếp nhanh. Khi dự đoán
> tại thời điểm t, hệ thống chỉ sử dụng lịch sử trước t để tránh nhìn thấy dữ
> liệu tương lai.
>
> Ở tầng cuối, mỗi ứng viên được biểu diễn bằng một biến nhị phân trong QUBO.
> Hàm mục tiêu kết hợp mức rủi ro và quan hệ giữa các ứng viên. Prototype dùng
> OpenJij Simulated Quantum Annealing chạy trên CPU để giải bài toán này, nên
> demo không cần QPU, D-Wave API hoặc kết nối cloud.

## 6. Demo xử lý cảnh báo — 90 giây

**Thao tác:**

1. Bấm biểu tượng chuông để mở panel nếu panel đang thu nhỏ.
2. Chọn một cảnh báo có mức risk cao.
3. Chỉ vào mã giao dịch, mức risk, số tiền và khuyến nghị.
4. Di chuột lên nút `Thực thi hành động` để thể hiện hiệu ứng phản hồi.
5. Có thể giữ hành động khuyến nghị hoặc chọn một hành động khác.
6. Bấm `Thực thi hành động`.
7. Chờ xác nhận thành công và chỉ ra rằng cảnh báo đã biến mất.

**Lời trình bày:**

> Khi một giao dịch vượt ngưỡng ứng viên, hệ thống đưa giao dịch đó vào panel
> cần xử lý bên phải. Các cảnh báo được sắp xếp theo mức độ ưu tiên.
>
> Với mỗi cảnh báo, hệ thống cung cấp một hành động khuyến nghị. Người vận hành
> vẫn là người ra quyết định cuối cùng và có thể chọn một hành động khác như
> yêu cầu xác minh, tạm giữ, chuyển điều tra hoặc từ chối giao dịch.
>
> Tôi sẽ thực thi hành động cho giao dịch này. Sau khi hệ thống ghi nhận thành
> công, một xác nhận xuất hiện và cảnh báo được xóa khỏi hàng đợi. Trong khi tôi
> thao tác, luồng giao dịch và quá trình chấm điểm phía sau vẫn tiếp tục chạy.

## 7. Kết luận — 45 giây

**Thao tác:** Thu nhỏ panel để quay lại toàn bộ bảng realtime.

**Lời trình bày:**

> Q-FraudGuard minh họa một quy trình fraud detection hybrid hoàn chỉnh: AI xử
> lý toàn bộ luồng giao dịch, TGNN bổ sung ngữ cảnh quan hệ và QUBO/SQA tối ưu
> hóa nhóm hồ sơ cần ưu tiên.
>
> Đầu ra của hệ thống không chỉ là nhãn fraud hoặc non-fraud. Hệ thống tạo risk
> score, hàng đợi ưu tiên và khuyến nghị hành động để hỗ trợ người vận hành đưa
> ra quyết định.
>
> Trong demo, PaySim cung cấp ground truth để đánh giá. Khi triển khai thực tế,
> cùng kiến trúc có thể nhận transaction stream nội bộ của ngân hàng và mở rộng
> thêm tín hiệu rủi ro liên ngân hàng mà không cần chia sẻ toàn bộ lịch sử giao
> dịch thô.

## Câu trả lời ngắn cho các câu hỏi thường gặp

### SQA có phát hiện fraud không?

> Không. AI và mô hình graph tạo risk score và lọc ứng viên. SQA tối ưu hóa
> nhóm ứng viên nên được ưu tiên xử lý.

### Vì sao ứng viên nhưng SQA không chọn?

> Nguồn lực điều tra trong mỗi cửa sổ có giới hạn. SQA chọn tổ hợp có tổng mức
> ưu tiên tốt nhất; các ứng viên còn lại vẫn được giữ trong hàng đợi.

### Có sử dụng máy tính lượng tử thật không?

> Không. Prototype dùng OpenJij Simulated Quantum Annealing trên CPU. Thiết kế
> QUBO cho phép thay solver trong tương lai nếu cần.

### Model có được train trên dashboard không?

> Không. Model được train và calibration offline. Dashboard chỉ tải artifact
> đã lưu để chạy inference realtime.

### Tại sao giao diện có nhãn thật?

> Đây là ground truth của PaySim để minh họa và tính metric trong demo. Hệ
> thống không sử dụng nhãn thật làm đầu vào khi chấm điểm giao dịch.

### Việc chạy lại PaySim có làm thay đổi dữ liệu không?

> Không. Đây chỉ là cơ chế replay trong bộ nhớ để mô phỏng stream liên tục.

## Phương án ứng biến khi demo

- Nếu chưa có cảnh báo: tiếp tục trình bày bảng và chờ luồng chạy thêm vài giây.
- Nếu panel che bảng: bấm dấu `−` để thu nhỏ, sau đó dùng biểu tượng chuông để
  mở lại.
- Nếu có nhiều cảnh báo: chọn cảnh báo trên cùng vì danh sách đã được sắp theo
  mức ưu tiên.
- Nếu được hỏi về tỷ lệ fraud: nhấn mạnh đây là tỷ lệ fraud thực tế lọt vào tập
  ứng viên trên phần dữ liệu đã chạy qua, không phải tỷ lệ ứng viên là fraud.
