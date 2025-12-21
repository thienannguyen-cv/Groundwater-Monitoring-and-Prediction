# 🌊 Groundwater Monitoring and Prediction System
### *AI-Driven Hydrological Intelligence Framework (HDVO Architecture)*

An advanced system for monitoring and predicting groundwater levels (GWL) using Artificial Intelligence, built on the **HDVO (Human-in-the-Loop Hypothesis-Driven Validation and Optimization)** architecture.

---

## 🇺🇸 ENGLISH VERSION

### 🚀 Deployment Guide
This project uses a Hybrid architecture: **React (Frontend)** and **Flask (Backend Proxy)** to secure API keys and handle complex AI logic.

1. **Backend (Render):** (Was Deployed)
2. **Frontend (Vercel):** Connect your repo. Set `VITE_API_URL` (your backend URL) and `VITE_CLIENT_KEY`.

### 🧩 Architecture: The HDVO Framework
The system implements the **HDVO Architecture**, bridging the **Semantic Gap** between raw data and real-world logic:

* **L1 (Hypothesis):** Encodes knowledge into executable JS prediction functions.
* **L2 (Execution):** Seamlessly bridges Python/Firebase data to the logic engine.
* **L3 (Optimization):** A continuous loop that validates logic against **Ground Truth** and **Human Insight**.

### 💡 Why HDVO? The "Self-Service" Advantage
Traditional systems are "blind" to sudden context changes. HDVO allows users to **"scratch their own itch"** through Human-in-the-loop:

* **Scenario:** A local pump breaks down unexpectedly. Sensors won't show the impact for hours.
* **AI Solution:** The user inputs a simple hint: *"Pump #2 is offline for repairs."*
* **Result:** The AI immediately rewrites the prediction logic (L1) to account for the lack of extraction, providing an accurate forecast long before the data "catches up."

### 🧠 Key AI Features
- **Self-Synthesizing Logic:** AI improves prediction functions based on error analysis (even without human hints).
- **Explainable AI:** Provides a `theory` field for every change, ensuring scientific transparency.

---

## 🇻🇳 TIẾNG VIỆT

### 🚀 Hướng dẫn Triển khai
Dự án sử dụng kiến trúc Hybrid: **React (Frontend)** và **Flask (Backend Proxy)**.

1. **Backend (Render):** (Đã được deploy)
2. **Frontend (Vercel):** Kết nối repo. Thiết lập `VITE_API_URL` (URL của backend) và `VITE_CLIENT_KEY`.

### 🧩 Kiến trúc HDVO: Trí tuệ dựa trên Giả thiết
Hệ thống là minh chứng thực tế cho kiến trúc **HDVO**, giúp xóa bỏ **Khoảng cách Ngữ nghĩa** giữa dữ liệu thô và thực tế vận hành:

* **L1 (Lớp Giả thiết):** Mã hóa tri thức thủy văn thành các hàm JavaScript có thể thực thi.
* **L2 (Lớp Thực thi):** Cầu nối dữ liệu từ Python/Firebase sang môi trường chạy Logic.
* **L3 (Lớp Tối ưu hóa):** Vòng lặp phản hồi dựa trên sai số thực tế và **Gợi ý từ con người**.

### 💡 Tại sao chọn HDVO? Lợi thế "Tự phục vụ"
Các hệ thống phi AI thường bị "đóng cứng" về logic. Kiến trúc HDVO cho phép người dùng **"tự gãi lưng cho chính mình"** trước những biến động tức thời:

* **Tình huống:** Một máy bơm tại trạm bất ngờ hỏng. Các cảm biến sẽ mất nhiều giờ mới ghi nhận được sự thay đổi mực nước.
* **Giải pháp AI:** Người dùng chỉ cần nhập: *"Máy bơm số 2 đang dừng để sửa chữa."*
* **Kết quả:** AI lập tức viết lại logic dự báo (L1) để phản ánh việc ngừng khai thác nước, đưa ra dự đoán chính xác ngay lập tức mà không cần đợi dữ liệu cập nhật hay sửa mã nguồn.

### 🧠 Tính năng AI Nổi bật
- **Logic tự tổng hợp (Self-Synthesizing):** AI tự phân tích lỗi và cải thiện hàm dự báo ngay cả khi không có sự can thiệp của con người.
- **AI có khả năng giải thích:** AI cung cấp trường `theory` (Học thuyết) để giải thích cơ sở khoa học đằng sau mỗi điều chỉnh logic.

---
## 🛠 Tech Stack / Công nghệ sử dụng
- **Frontend:** React, Recharts, Firebase SDK.
- **Backend:** Python Flask, Gemini 1.5 Flash API.
- **Methodology:** HDVO (Human-in-the-loop Hypothesis-Driven Validation and Optimization).

---
*Developed for sustainable water resource management. Open for contributions!*
