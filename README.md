# 🌊 Groundwater Monitoring and Prediction System
### *AI-Driven Hydrological Intelligence Framework (HDVO Architecture)*

An advanced system for monitoring and predicting groundwater levels (GWL) using Artificial Intelligence, built on the **HDVO (Human-in-the-Loop Hypothesis-Driven Validation and Optimization)** architecture.

---

## 🇺🇸 ENGLISH VERSION

### 🚀 Deployment Guide
This project uses a Hybrid architecture: **React (Frontend)** and **Flask (Backend Proxy)**.

#### 1. Backend (Flask) on Render
1. Upload backend source code (containing `app.py` and `requirements.txt`) to GitHub.
2. Create a **Web Service** on [Render.com](https://render.com/).
3. Set **Environment Variables**: `GEMINI_API_KEY` and `CLIENT_SECRET_KEY_FETCH`.

#### 2. Frontend (Vercel)
1. Connect your repo to [Vercel.com](https://vercel.com/).
2. **Project Settings:**
   - **Framework Preset:** `Vite`
   - **Build Command:** `npm run build`
   - **Install Command:** `npm install`
3. **Environment Variables:**
   - `VITE_API_URL`: Your Render Backend URL.
   - `VITE_CLIENT_KEY`: Must match `CLIENT_SECRET_KEY_FETCH`.
   - Firebase variables: `VITE_FIREBASE_API_KEY`, etc.

### 💡 The HDVO Advantage: "Scratch Your Own Itch"
Traditional systems are "blind" to sudden context changes (e.g., a pump breaks down). HDVO allows users to **"scratch their own itch"** by providing immediate semantic hints:

* **Scenario:** A local pump fails unexpectedly. Data hasn't reflected this yet.
* **Human-in-the-loop:** User inputs: *"Pump #2 is offline for repairs."*
* **AI Action:** The AI immediately rewrites the prediction logic (L1 Hypothesis) to account for the lack of extraction, providing an accurate forecast long before the raw data "catches up."

### 🧩 Architecture Layers
* **L1 (Hypothesis):** Encodes knowledge into executable JS prediction functions.
* **L2 (Execution):** Seamlessly bridges data to the logic engine.
* **L3 (Optimization):** Validates logic against **Ground Truth** and **Human Insight** (The system works autonomously but thrives with human guidance).

---

## 🇻🇳 TIẾNG VIỆT

### 🚀 Hướng dẫn Triển khai
Dự án sử dụng kiến trúc Hybrid: **React (Frontend)** và **Flask (Backend Proxy)**.

#### 1. Backend (Flask) trên Render
1. Đẩy mã backend lên GitHub.
2. Tạo **Web Service** trên [Render.com](https://render.com/).
3. Thiết lập **Biến môi trường**: `GEMINI_API_KEY` và `CLIENT_SECRET_KEY_FETCH`.

#### 2. Frontend (Vercel)
1. Kết nối repo với [Vercel.com](https://vercel.com/).
2. **Cấu hình dự án:**
   - **Framework Preset:** `Vite`
   - **Build Command:** `npm run build`
   - **Install Command:** `npm install`
3. **Biến môi trường:**
   - `VITE_API_URL`: URL Backend từ Render.
   - `VITE_CLIENT_KEY`: Khớp với `CLIENT_SECRET_KEY_FETCH`.
   - Các biến Firebase: `VITE_FIREBASE_API_KEY`, v.v.

### 💡 Lợi thế HDVO: "Tự gãi lưng cho chính mình"
Các hệ thống phi AI thường bị "đóng cứng". Kiến trúc HDVO cho phép người dùng **"tự gãi lưng cho chính mình"** trước những biến động tức thời mà cảm biến chưa kịp ghi nhận:

* **Tình huống:** Một máy bơm bất ngờ bị hỏng. Dữ liệu số chưa kịp phản ánh sự thay đổi.
* **Sự tham gia của con người:** Người dùng nhập: *"Máy bơm số 2 đang dừng để sửa chữa."*
* **Hành động AI:** AI lập tức viết lại logic dự báo (Lớp Giả thiết L1) để phản ánh việc ngừng khai thác nước, đưa ra dự đoán chính xác ngay lập tức mà không cần đợi cập nhật mã nguồn hay dữ liệu cảm biến.

### 🧩 Các lớp kiến trúc
* **L1 (Giả thiết):** Mã hóa tri thức thành các hàm JS có thể thực thi.
* **L2 (Thực thi):** Cầu nối dữ liệu sang môi trường chạy Logic.
* **L3 (Tối ưu hóa):** Kiểm định logic dựa trên **Dữ liệu thực tế** và **Gợi ý từ con người** (Hệ thống tự phân tích lỗi ngay cả khi không có gợi ý, nhưng trở nên cực kỳ linh hoạt khi có con người tham gia).

---
*Developed for sustainable water resource management.*
