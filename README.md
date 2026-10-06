# FlightBookingAgent ✈️

> **Hệ Thống Đặt Vé Máy Bay Thông Minh với Kiến Trúc Khung Kiềm Tỏa (4-Layer Harness) và Chuẩn Hóa 3 Mẫu Thiết Kế Agent (ReAct, Plan-then-Execute, Hybrid)**  
> *Đồ án thực hành môn học: SE373 - Kỹ thuật xây dựng hệ thống Agentic AI*

---

## 📌 Thông Tin Tác Giả / Student Information

| Mục / Field | Thông tin chi tiết / Details |
| :--- | :--- |
| **Họ và tên / Full Name** | `LÊ MINH HOÀNG` |
| **Mã số sinh viên / Student ID** | `24520542` |
| **Lớp / Class** | `KHMT2024.2` |
| **Môn học / Course** | SE373 - Kỹ thuật xây dựng hệ thống Agentic AI |
| **Đơn vị đào tạo / Institution** | Trường Đại học Công nghệ Thông tin - ĐHQG-HCM (UIT - VNU-HCM) |
| **Email liên hệ / Contact** | `24520542@gm.uit.edu.vn` |
| **Repository URL** | [https://github.com/lmhoang06/FlightBookingAgent](https://github.com/lmhoang06/FlightBookingAgent) |

---

## 🤖 Tuyên Bố Sử Dụng Trí Tuệ Nhân Tạo & Giải Trình Minh Bạch (AI Assistance & Discrepancy Disclosure)

> **Thông báo về tính trung thực học thuật (Academic Integrity & Transparency Disclosure):**  
> Trong quá trình nghiên cứu, thiết kế kiến trúc và hiện thực mã nguồn của dự án này, mô hình **Gemini** (cung cấp bởi Google DeepMind thông qua môi trường lập trình Google Antigravity / Gemini 3.8 Flash) đã được sử dụng như một **trợ lý lập trình cặp (AI Pair Programmer)** và **công cụ đối soát kiến trúc (Architectural Audit & Alignment Tool)**.

### Chi tiết phạm vi hỗ trợ và xử lý sai lệch (Discrepancy Resolution):
1. **Rà soát kiến trúc & Đối soát yêu cầu (Architectural Audit)**:
   - Ban đầu, mã nguồn gốc tồn tại một số điểm chưa đồng bộ tuyệt đối với các tiêu chuẩn trong bài giảng lý thuyết (ví dụ: thiếu bộ chặn quyền tập trung `PermissionInterceptor`, tiêu chuẩn dừng còn phụ thuộc vào phán đoán văn bản của LLM thay vì xác thực trực tiếp cơ sở dữ liệu bằng mã lệnh code, và cơ chế lập kế hoạch chưa phân tách rõ bước Human Approval).
   - Mô hình Gemini được sử dụng để phân tích chênh lệch (discrepancy analysis), đề xuất kế hoạch tái cấu trúc chi tiết và chuẩn hóa kiến trúc được ghi nhận toàn diện tại [`REPORT.md`](./REPORT.md).
2. **Hiện thực hóa 4 Tầng Harness & 3 Mẫu Agent**:
   - Hỗ trợ xây dựng khung rào chắn bảo vệ [`agents/harness.py`](./agents/harness.py) với 4 tầng: *Constraints as DATA*, *Permission Check*, *Code-based is_done*, và *Human Handoff*.
   - Hoàn thiện 3 mẫu thiết kế Agent theo đúng bài giảng:
     - **ReAct** ([`agents/react/agent.py`](./agents/react/agent.py)): Tích hợp Interceptors và kiểm tra dừng độc lập.
     - **Plan-then-Execute** ([`agents/plan_then_execute/agent.py`](./agents/plan_then_execute/agent.py)): Tạo kế hoạch 1-shot, cổng phê duyệt Human Reviewer, thực thi tuần tự bằng Plain-Code với cơ chế thay thế `$placeholder` (chi phí LLM ở khâu thực thi = 0).
     - **Hybrid** ([`agents/hybrid/agent.py`](./agents/hybrid/agent.py)): Phân tầng vĩ mô (Milestone) kết hợp vi mô (Micro-ReAct ngân sách $k$ bước), bộ đánh giá độ lệch quan sát `_should_replan`, và tái lập kế hoạch động.
3. **Kiểm thử mô phỏng xác định (Deterministic Testing & Synthetic Scenarios)**:
   - Hiện thực [`agents/fake_model.py`](./agents/fake_model.py) (`ScriptedFakeChatModel`) để tái hiện chính xác 100% các tình huống biên nguy hiểm: Goal Drift, Infinite Loop, và Handoff mà không phụ thuộc vào API mạng bên ngoài.
   - Xây dựng bộ công cụ đo lường và đánh giá thực nghiệm độc lập [`scripts/evaluate.py`](./scripts/evaluate.py) xuất kết quả ra [`scripts/benchmark_results.json`](./scripts/benchmark_results.json).
4. **Kiểm tra và Xác thực của Con người (Human Verification)**:
   - Toàn bộ mã nguồn, cấu hình, logic nghiệp vụ và bộ test (đạt **46/46 unit & integration tests passing**) đều được kiểm tra, biên dịch, chạy thực nghiệm và được xác nhận đạt yêu cầu bởi người học và kiểm thử viên độc lập (independent tester acceptance).
   - Mọi phân tích định lượng, bảng so sánh hiệu năng và đánh giá ưu nhược điểm được ghi chép đầy đủ tại [`REPORT.md`](./REPORT.md).

---

## 🌟 Tổng Quan Dự Án (Project Overview)

Trong các hệ thống Agentic AI thực hiện tác vụ giao dịch nhạy cảm như **đặt vé máy bay tự động (Flight Booking)**, mô hình LLM đơn thuần có thể gặp phải 4 hiểm họa nghiêm trọng:
1. **Goal Drift (Trôi mục tiêu)**: Tự ý chọn vé vượt ngân sách hoặc sai khung giờ bay.
2. **Infinite Loop (Vòng lặp vô hạn)**: Lặp lại vô tận hành động tìm kiếm/đặt vé khi gặp phản hồi lỗi.
3. **Tool Hallucination (Ảo giác công cụ)**: Tự bịa đặt mã vé hoặc số hiệu chuyến bay không có trong DB.
4. **State Corruption (Hư hại trạng thái)**: Báo kết quả sai lệch khi dữ liệu trả về rỗng hoặc lỗi mạng.

Dự án này giải quyết triệt để các vấn đề trên thông qua **Kiến trúc Khung kiềm tỏa 4 tầng (4-Layer Harness Architecture)**, bảo vệ toàn diện hệ thống bất kể agent sử dụng mô hình xác suất nào.

---

## 🏛️ Kiến Trúc Khung Kiềm Tỏa (The 4-Layer Harness)

```
                          [ YÊU CẦU NGƯỜI DÙNG ]
                                    │
                    ┌───────────────▼───────────────┐
                    │   1. RÀNG BUỘC LÀ DỮ LIỆU     │
                    │      (Constraints as DATA)    │
                    │  origin, dest, date, budget,  │
                    │  depart_before/after, seats   │
                    └───────────────┬───────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
 ┌───────▼─────────┐        ┌───────▼─────────┐        ┌───────▼─────────┐
 │     ReAct       │        │Plan-then-Execute│        │     Hybrid      │
 │ Phán đoán theo  │        │ 1-Shot Plan Gen │        │ Plan k bước     │
 │ từng bước       │        │ ├─ Human Review │        │ Thực thi k bước │
 │                 │        │ └─ Plain Code   │        │ Obs đổi? Replan │
 └───────┬─────────┘        └───────┬─────────┘        └───────┬─────────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  2. KIỂM QUYỀN    │
                          │(Permission Check) │
                          │ Chặn TRƯỚC tool   │
                          └─────────┬─────────┘
                                    │
                          ┌─────────▼─────────┐
                          │ 3. XÉT DỪNG (CODE)│
                          │   (is_done check) │
                          │ Kiểm tra trực tiếp│
                          │ trạng thái DB     │
                          └─────────┬─────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
             [ THÀNH CÔNG ]                  [ 4. CHUYỂN GIAO ]
          Vé được cấp hợp lệ              (Handoff to Human)
```

1. **Layer 1: Constraints as DATA (`FlightConstraints`, `ExtractedConstraints`)**:
   - Trích xuất ràng buộc người dùng thành đối tượng dữ liệu có kiểu rõ ràng, không phụ thuộc vào context window hay prompt injection.
2. **Layer 2: Permission Check (`check_permission`, `PermissionInterceptor`)**:
   - Kiểm tra thẩm quyền nghiêm ngặt trước khi gọi công cụ nhạy cảm (`book_flight`). Vé phải tồn tại trong bộ nhớ tìm kiếm và thỏa mãn ngân sách.
3. **Layer 3: Code-based Stop Criteria (`is_done`)**:
   - Sử dụng hàm logic Python kiểm tra trực tiếp cơ sở dữ liệu (`mock_db["bookings"]`), tuyệt đối không dựa vào câu trả lời tự xưng "I have booked" của LLM.
4. **Layer 4: Human Handoff (`handoff`, `HandoffTrigger`)**:
   - Cung cấp cơ chế thoát an toàn (safe escape) kèm phiếu thông tin (`HandoffTicket`) khi gặp bế tắc, lặp quá ngưỡng, hoặc vi phạm ngân sách.

---

## 📊 Bảng So Sánh Hiệu Năng 3 Mẫu Agent (Benchmark Summary)

Được tổng hợp từ tập thực nghiệm độc lập [`scripts/benchmark_results.json`](./scripts/benchmark_results.json) trên 4 kịch bản chuẩn (chuẩn thành công, trôi mục tiêu, lặp vô hạn, và vượt ngân sách):

| Tiêu Chí / Chỉ Số | ReAct (có Harness) | Plan-then-Execute | Hybrid (Lai) |
| :--- | :---: | :---: | :---: |
| **Tỷ lệ thành công (Success Rate)** | 100% | 100% | 100% |
| **Tỷ lệ chặn trôi mục tiêu (Drift Block Rate)** | 100% | 100% | 100% |
| **Số lần gọi LLM trung bình (LLM Calls)** | ~3.0 | **1.0 (Tối ưu nhất)** | ~2.5 |
| **Thời gian thực thi trung bình (Latency)** | ~4.5 ms | **~1.5 ms (Nhanh nhất)** | ~3.8 ms |
| **Human-in-the-loop Gate** | Tùy chọn | **Có sẵn (Bắt buộc)** | Tích hợp |
| **Khả năng tự hồi phục khi đổi bối cảnh** | Trung bình | Kém | **Xuất sắc (Dynamic Replan)** |
| **Nguy cơ lặp vô hạn** | Có thể (được Harness dập tắt) | Không bao giờ | Được kiểm soát bởi ngân sách $k$ |

---

## 📁 Cấu Trúc Mã Nguồn (Repository Structure)

```
FlightBookingAgent/
├── agents/                         # Mã nguồn cốt lõi của các Agent & Harness
│   ├── harness.py                  # Kiến trúc Harness 4 tầng & Interceptors
│   ├── fake_model.py               # Deterministic Scripted Model cho kiểm thử
│   ├── react/
│   │   └── agent.py                # ReAct Agent có bảo vệ Harness
│   ├── plan_then_execute/
│   │   ├── agent.py                # Plan-then-Execute Agent (Human Gate + Plain-Code)
│   │   └── planner.py              # 1-Shot Structured Planner
│   └── hybrid/
│       ├── agent.py                # Hybrid Agent (Milestone + Micro-ReAct)
│       └── planner.py              # Macro Milestone Planner
├── tools/                          # Bộ công cụ nghiệp vụ tích hợp
│   ├── flight_search.py            # Tìm kiếm chuyến bay theo tiêu chí
│   ├── booking.py                  # Đặt vé máy bay (nhạy cảm, có kiểm quyền)
│   └── airports.py                 # Tra cứu sân bay và thành phố
├── tests/                          # 46 kiểm thử tự động (Unit & Integration)
│   ├── agents/                     # Kiểm thử Harness, ReAct, Plan-then-Exec, Hybrid
│   └── tools/                      # Kiểm thử tính hợp lệ của các Tools
├── scripts/                        # Kịch bản đánh giá & đo lường
│   ├── evaluate.py                 # Benchmark runner cho cả 3 mẫu Agent
│   └── benchmark_results.json      # Kết quả đo lường thực nghiệm
├── app.py                          # Giao diện trực quan Streamlit tương tác
├── requirements.txt                # Danh sách thư viện phụ thuộc
├── REPORT.md                       # Báo cáo kỹ thuật phân tích chuyên sâu
└── README.md                       # Tài liệu hướng dẫn sử dụng & thông tin đồ án
```

---

## 🚀 Cài Đặt & Hướng Dẫn Sử Dụng (Quickstart)

### 1. Chuẩn bị môi trường (Prerequisites)
Yêu cầu **Python 3.10+**. Khuyến nghị sử dụng môi trường ảo (`venv`):

```bash
# Di chuyển vào thư mục dự án
cd FlightBookingAgent

# Tạo và kích hoạt virtual environment
python3 -m venv venv
source venv/bin/activate  # Trên Linux / macOS
# hoặc: venv\Scripts\activate  # Trên Windows

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Thiết lập biến môi trường (Environment Variables)
Tạo file `.env` từ file mẫu:
```bash
cp .env.example .env
```
*(Nếu muốn chạy với LLM thực tế như OpenAI hoặc Anthropic, hãy điền `OPENAI_API_KEY` tương ứng vào file `.env`. Nếu không, hệ thống hỗ trợ chế độ Offline Scripted Demo hoàn toàn độc lập và không tốn phí).*

### 3. Chạy giao diện tương tác Streamlit (Interactive Web Playground)
Khởi chạy ứng dụng Streamlit:
```bash
streamlit run app.py
```
**Các tính năng nổi bật trên giao diện Streamlit:**
- **Sidebar Constraints as DATA**: Tự do điều chỉnh Điểm đi, Điểm đến, Ngày bay, Ngân sách tối đa, Khung giờ bay.
- **Pattern Selector**: Lựa chọn trực quan giữa 3 mẫu Agent (`ReAct`, `Plan-then-Execute`, `Hybrid`).
- **Live Harness Inspector**: Hiển thị trực quan trạng thái 4 tầng Harness (Ràng buộc, Thẩm quyền, Xét dừng, Chuyển giao).
- **Interactive Human Handoff Card**: Bảng tương tác hỗ trợ bàn giao người dùng khi có sự cố.
- **Offline Scripted Demo Switcher**: Trải nghiệm các kịch bản mô phỏng lỗi (Goal Drift, Infinite Loop) mà không cần API key.

---

## 🧪 Kiểm Thử & Đánh Giá Thực Nghiệm (Testing & Evaluation)

### 1. Chạy toàn bộ 46 Unit & Integration Tests
```bash
pytest -v
```
*Kết quả kỳ vọng: 46/46 tests passed (100%).*

### 2. Chạy bài đo lường Benchmark
```bash
python scripts/evaluate.py
```
Kịch bản đánh giá sẽ tự động chạy qua 4 tình huống với cả 3 mẫu thiết kế, tính toán các chỉ số `Success Rate`, `Goal Drift Rate`, `LLM Calls`, `Latency`, và xuất báo cáo JSON tại `scripts/benchmark_results.json`.

---

## 📚 Tài Liệu Báo Cáo Kỹ Thuật (Technical Documentation)
- [`REPORT.md`](./REPORT.md): Báo cáo kỹ thuật chi tiết trình bày toàn diện kiến trúc 4 tầng Harness, cơ chế phòng ngừa 4 hiểm họa đặc trưng (Goal Drift, Infinite Loop, Tool Hallucination, State Corruption), thiết kế chi tiết 3 mẫu Agent và kết quả thực nghiệm đo lường.
