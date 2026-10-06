# BTVN#3 — Hướng dẫn chạy và báo cáo Agent đặt vé máy bay

> Bài tập SE373: tìm hiểu LangChain/LangGraph, tạo công cụ giả lập, cài đặt harness, triển khai ba mẫu ReAct, Plan-then-Execute, Lai và so sánh hiệu quả.

- **Ba mẫu:** `react_agent.py`, `plan_execute_agent.py`, `hybrid_agent.py`
- **Phần dùng chung:** `flight_core.py`, `strategy_runtime.py`
- **Đánh giá:** `evaluate_three_agents.py`, `benchmark_three_agents.json`
- **Môi trường:** Python 3.10 trở lên; `langchain>=1.0,<2.0`, `langgraph>=1.0,<2.0`.

## 1. Hướng dẫn chạy

Mở PowerShell tại thư mục chứa ba file agent. Tạo môi trường Python riêng, cài thư viện, rồi chạy từng mẫu:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe react_agent.py --scenario all
.\.venv\Scripts\python.exe plan_execute_agent.py --scenario all
.\.venv\Scripts\python.exe hybrid_agent.py --scenario all
```

Nếu máy không có `py`, dùng `python -m venv .venv`. Kiểm tra phiên bản bằng `py -3 --version` hoặc `python --version`. Chương trình không cần API key: `MockFlightModel` chạy cục bộ, dữ liệu chuyến bay và thanh toán đều giả lập.

Các lệnh để xem chi tiết và đánh giá:

```powershell
# Một chiến lược, một kịch bản; in JSON đầy đủ
.\.venv\Scripts\python.exe hybrid_agent.py --scenario 3 --json

# Cho phép giữ chỗ và thanh toán giả lập trong phiên chạy mới này
.\.venv\Scripts\python.exe hybrid_agent.py --scenario 3 --approve --json

# Chạy 15 trường hợp từ ba file mới, cập nhật benchmark_three_agents.json
.\.venv\Scripts\python.exe evaluate_three_agents.py

```

Mỗi file agent chọn sẵn một mẫu; `--scenario` nhận `1` đến `5` hoặc `all`. Nếu bỏ tùy chọn, file đó chạy cả 5 kịch bản. `--json` in mỗi kết quả trên một dòng JSON; nếu không dùng, CLI in bảng tóm tắt. `--approve` chỉ áp dụng cho phiên chạy hiện tại. Chạy lại cùng kịch bản tạo một `FlightStore` mới, không tiếp tục booking từ phiên trước.

Nếu PowerShell chặn lệnh kích hoạt môi trường ảo, cứ gọi trực tiếp `.\.venv\Scripts\python.exe` như trên; không cần đổi execution policy. Trên macOS/Linux, dùng `python3 -m venv .venv`, `./.venv/bin/python -m pip install -r requirements.txt` và thay đường dẫn Python tương ứng.

### Cách đọc trạng thái

| Trạng thái | Ý nghĩa |
|---|---|
| `COMPLETED` | `get_booking` đã xác minh booking `confirmed`, `is_paid=True` và chuyến hợp lệ |
| `APPROVAL_REQUIRED` | Harness dừng trước thao tác cần quyền; xem câu hỏi trong `handoff` |
| `HANDOFF` | Cần người xử lý tiếp; xem `stop_reason`, `da_thu`, `trang_thai` |
| `UNSUPPORTED_CLAIM` | Câu trả lời chứa dữ kiện không có trong kết quả tool đã quan sát |

Ví dụ, kịch bản 3 không có `--approve` dừng ở `APPROVAL_REQUIRED` sau 2 lượt tool. Chạy lại với `--approve` cho phép hoàn tất luồng giả lập. Kịch bản 1 và 2 được đánh dấu phê duyệt sẵn trong bộ đánh giá để đo luồng hoàn thành.

## 2. Mục tiêu và nền tảng

Bài làm mô phỏng đặt vé cho hành khách `Nguyen Van A`, ngày **07/10/2026**, chặng mặc định **SGN → DAD**, khởi hành **trước 12:00**, giá **không quá 2.000.000 đồng**. Ngày và dữ liệu được cố định trong mã để kết quả lặp lại được; đây không phải tra cứu chuyến bay trực tiếp.

Theo [tài liệu LangChain về agents](https://docs.langchain.com/oss/python/langchain/agents), agent dùng model gọi công cụ theo vòng lặp; harness là phần bao quanh để định hình hành vi. Bài này dùng `@tool` của LangChain để định nghĩa công cụ, `BaseChatModel` giả lập để tạo quyết định và [`StateGraph` của LangGraph](https://docs.langchain.com/oss/python/langgraph/graph-api) để điều phối `decide → act → decide/end`. Đây là luồng riêng bằng LangGraph, không gọi `create_agent`. [Tài liệu LangChain về tools](https://docs.langchain.com/oss/python/langchain/tools) mô tả công cụ là các hàm có tên, mô tả và đầu vào có cấu trúc mà agent có thể gọi.

## 3. Thiết kế và cài đặt

### 3.1. Mockup công cụ

`FlightStore` giữ chuyến bay, ghế và booking trong bộ nhớ. Mỗi lần gọi hàm `run()` của một mẫu tạo một kho riêng. Năm công cụ `@tool` cùng dùng kho này:

| Tool | Vai trò | Kết quả chính |
|---|---|---|
| `search_flights` | Tìm theo điểm đi, điểm đến, ngày | `flights`, `count` |
| `check_seat` | Xem ghế trống | `available_seats` |
| `book_seat` | Giữ ghế giả lập | Booking `held`, `is_paid=False` |
| `pay` | Thanh toán giả lập | Booking `confirmed`, `is_paid=True` |
| `get_booking` | Đọc lại booking | Căn cứ xác nhận hoàn thành |

Kho có `VN122` (08:10, 1.850.000đ), `QH118` (15:40, 1.640.000đ), `VJ602` (09:30, 1.450.000đ, không hoàn hủy) và `VN128` (11:15, 2.450.000đ). `QH118` vi phạm giờ; `VN128` vượt ngân sách. Giá và mã chỉ là dữ liệu mock.

### 3.2. Harness

| Thành phần | Quy tắc kiểm bằng code | Khi không đạt |
|---|---|---|
| `FlightConstraints` | So chặng, ngày, giờ khởi hành, giá với yêu cầu | Chặn `book_seat`; ReAct/Lai thử chuyến khác, Plan bàn giao |
| `CompletionCriteria` | Booking có mã, chuyến đã biết, đúng ràng buộc, `confirmed`, đã thanh toán | Không báo `COMPLETED` |
| `PermissionChecker` | Vé trên 1.500.000đ hoặc không hoàn hủy cần duyệt trước `book_seat`; `pay` luôn cần duyệt khi chưa có quyền | `APPROVAL_REQUIRED` trước tool nhạy cảm |
| `handoff` | Ghi lý do dừng, việc đã thử, trạng thái, câu hỏi | Trả gói bàn giao có cấu trúc |
| `LoopDetector` | Bắt lặp tool/tham số, observation hoặc không tăng tiến độ nghiệp vụ | Dừng với lý do `LOOP` hoặc `STALL` |
| `GroundingChecker` | Đối chiếu mã chuyến, booking, ghế, ngày, giờ, giá trong câu trả lời với observation | `UNSUPPORTED_CLAIM` |

Mỗi phiên có giới hạn **15 lượt model hoặc 15 lượt tool**. Agent chỉ được công bố hoàn thành sau khi `get_booking` xác nhận trạng thái bằng code. Gói bàn giao gồm `stop_reason`, `da_thu`, `trang_thai`, `cau_hoi_cho_nguoi`.

### 3.3. Ba mẫu agent

Mỗi mẫu có **một file chạy riêng** và một lớp policy chứa quyết định cùng cách phục hồi khi tool báo chuyến không đạt. `flight_core.py` chứa kho dữ liệu, tools và harness; `strategy_runtime.py` dựng cùng đồ thị LangGraph và cung cấp CLI. Bộ đánh giá gọi trực tiếp ba file bên dưới:

| Mẫu | Cách quyết định | Khi chuyến đầu không đạt |
|---|---|---|
| **ReAct** (`react`) | Model chọn tool sau mỗi observation | Loại chuyến sai và chọn lại |
| **Plan-then-Execute** (`plan-execute`) | Model tạo kế hoạch 5 bước một lần; code thực thi tuần tự | Kế hoạch cố định dừng và bàn giao |
| **Lai** (`hybrid`) | Lập kế hoạch trước; model chọn tham số từng bước và đối chiếu kế hoạch | Loại chuyến sai, lập lại phần kế hoạch còn lại |

Ba mẫu dùng chung tools, dữ liệu và harness. `MockFlightModel` trả lời xác định theo trạng thái phiên; phép đo phản ánh cách điều phối trong bài, chưa đo LLM thật.

## 4. Phương pháp đánh giá

`evaluate_three_agents.py` chạy **3 mẫu × 5 kịch bản = 15 trường hợp**, ghi chi tiết vào `benchmark_three_agents.json`:

1. **Luồng hợp lệ:** chuyến đầu đáp ứng yêu cầu và được duyệt.
2. **Cần đổi hướng:** chuyến đầu `QH118` bay sau 12:00.
3. **Cần phê duyệt:** vé `VJ602` không hoàn hủy, chưa có quyền đặt.
4. **Không có chuyến:** chặng `VTG → VCS` không có dữ liệu.
5. **Bịa dữ kiện:** model phát biểu `VN999`, ghế `5C`, giá `1.200.000đ` trước khi có observation.

Chỉ `COMPLETED` được tính là đặt vé thành công. `APPROVAL_REQUIRED`, `HANDOFF` và `UNSUPPORTED_CLAIM` là các điểm dừng an toàn. Số lượt tool là số lần **thực thi**, không gồm lần bị harness chặn. Token được ước lượng bằng `(số ký tự prompt + số ký tự nội dung trả lời) // 4`; đây không phải usage của nhà cung cấp model. Chi phí API là **0 USD** vì không gọi LLM bên ngoài.

## 5. Kết quả và phân tích

Bảng sau lấy từ `benchmark_three_agents.json` của ba file mới. Thời gian có thể thay đổi khi chạy lại.

| Mẫu | Hoàn thành | Lượt model | Lượt tool | Token ước lượng | Tổng thời gian (s) | Vi phạm được bắt |
|---|---:|---:|---:|---:|---:|---:|
| ReAct | 2/5 (40%) | 18 | 14 | 887 | 0,022098 | 4 |
| Plan-then-Execute | 1/5 (20%) | 5 | 10 | 114 | 0,012212 | 3 |
| Lai | 2/5 (40%) | 22 | 14 | 968 | 0,017924 | 3 |

| Kịch bản | ReAct | Plan-then-Execute | Lai |
|---|---|---|---|
| 1. Luồng hợp lệ | `COMPLETED` | `COMPLETED` | `COMPLETED` |
| 2. Chuyến đầu sai giờ | `COMPLETED` | `HANDOFF` | `COMPLETED` (1 lần lập lại kế hoạch) |
| 3. Cần phê duyệt | `APPROVAL_REQUIRED` | `APPROVAL_REQUIRED` | `APPROVAL_REQUIRED` |
| 4. Không có chuyến | `HANDOFF` | `HANDOFF` | `HANDOFF` |
| 5. Bịa dữ kiện | `UNSUPPORTED_CLAIM` | `UNSUPPORTED_CLAIM` | `UNSUPPORTED_CLAIM` |

Plan-then-Execute dùng ít lượt model nhất vì chỉ lập kế hoạch một lần, nhưng không phục hồi ở kịch bản 2. ReAct và Lai hoàn thành thêm kịch bản đó nhờ xem lại kết quả tool; Lai lập lại kế hoạch một lần. Cả ba dừng trước `book_seat` ở kịch bản 3 khi chưa được duyệt và chặn dữ kiện bịa ở kịch bản 5. Tỷ lệ trên **5 kịch bản nhỏ**, với model và dữ liệu cố định, không phải ước lượng độ chính xác thống kê ngoài thực tế.

## 6. Kiểm chứng và giới hạn

Trong quá trình phát triển, **16/16 test đạt**: 15 test hành vi gốc và 1 test xác nhận ba file có thể chạy CLI riêng. Đã đối chiếu 15 kịch bản của ba module mới với kết quả trước khi tách: trạng thái, số lượt model/tool, token ước lượng, số lần chặn, số lần lập lại kế hoạch và số bước đều trùng. Bộ nộp tối giản không chứa file test; người chấm có thể chạy `evaluate_three_agents.py` để tái hiện 15 trường hợp và xem chi tiết trong JSON.

Giới hạn: không gọi API vé hay thanh toán thật; tồn kho chỉ trong bộ nhớ và không xử lý đồng thời; không có checkpoint để khôi phục phiên qua tiến trình khác. `GroundingChecker` kiểm một số dữ kiện có định dạng rõ, chưa xác minh toàn bộ nghĩa của câu tiếng Việt. `--approve` cấp quyền đầu một phiên chạy mới, không phải quy trình phê duyệt tương tác giữa chừng. Token và thời gian cục bộ không đại diện cho chi phí hoặc độ trễ của LLM thật.

## 7. Tệp nộp và nguồn tham khảo

Bộ nộp gồm đúng **9 file**: ba file agent `react_agent.py`, `plan_execute_agent.py`, `hybrid_agent.py`; hai file dùng chung `flight_core.py`, `strategy_runtime.py`; `evaluate_three_agents.py`; `benchmark_three_agents.json`; `requirements.txt`; và báo cáo này. Chạy lại `evaluate_three_agents.py` để tái tạo JSON và đối chiếu bảng ở mục 5. Không đưa môi trường ảo hay cache vào bộ nộp.

**Tài liệu tham khảo:** [LangChain Agents](https://docs.langchain.com/oss/python/langchain/agents), [LangChain Tools](https://docs.langchain.com/oss/python/langchain/tools), [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api). Số liệu và mô tả cài đặt dựa trên các file mã nguồn và kết quả đánh giá trong thư mục bài làm.
