# 🌊 Groundwater Monitoring and Prediction System
### *AI-Driven Hydrological Intelligence Framework (HDVO Architecture)*

An advanced system for monitoring and predicting groundwater levels (GWL) using Artificial Intelligence, built on the **HDVO (Human-in-the-Loop Hypothesis-Driven Validation and Optimization)** architecture.

---

## 🇺🇸 ENGLISH VERSION

### 🚀 Configuration & Deployment Guide

#### 1. Firebase Setup (Required for Cloud Features)
To use the "Cloud Storage" mode, follow these steps:
1.  **Create Project:** Go to [Firebase Console](https://console.firebase.google.com/) and create a new project.
2.  **Authentication:** Enable **Anonymous** provider in the *Build > Authentication > Sign-in method* section.
3.  **Firestore Database:** Create a database in **Production Mode**. 
4.  **Security Rules:** Update your Firestore Rules to allow anonymous access:
    ```javascript
    service cloud.firestore {
      match /databases/{database}/documents {
        match /{document=**} {
          allow read, write: if request.auth != null;
        }
      }
    }
    ```
5.  **Environment Variables:** Copy your Web App SDK config into your `.env` file (see below).

#### 2. Backend (Flask) on Render
- Upload your backend to GitHub (or use my backend at [https://hdvo-backend.onrender.com](https://hdvo-backend.onrender.com)).
- Set **Environment Variables**: `GEMINI_API_KEY` and `CLIENT_SECRET_KEY_FETCH`.

#### 3. Frontend (Vercel)
- **Framework Preset:** `Vite` | **Build Command:** `npm run build` | **Install Command:** `npm install`
- **Environment Variables:**
  - `VITE_API_URL`: Your Render Backend URL.
  - `VITE_CLIENT_KEY`: Shared secret key.
  - `VITE_FIREBASE_API_KEY`, `VITE_FIREBASE_AUTH_DOMAIN`, `VITE_FIREBASE_PROJECT_ID`, etc.

### 🛡 System Resilience: Graceful Degradation
- **Automatic Fallback:** If Firebase config is missing or fails, the system automatically disables "Cloud Mode" and switches to "Local File Mode".
- **Dynamic UI:** Firebase-related buttons are automatically disabled to prevent crashes.

### 💡 The HDVO Advantage: "Scratch Your Own Itch"
Traditional systems are "blind" to sudden context changes. HDVO allows users to **"scratch their own itch"**:
* **Scenario:** A local pump fails unexpectedly.
* **Human-in-the-loop:** User inputs: *"Pump #2 is offline for repairs."*
* **AI Result:** The AI immediately rewrites the prediction logic (L1) to reflect this, providing accurate forecasts long before sensors catch up.

---

## 🇻🇳 TIẾNG VIỆT

### 🚀 Hướng dẫn Cấu hình & Triển khai

#### 1. Thiết lập Firebase (Bắt buộc cho tính năng Cloud)
Để sử dụng chế độ "Lưu trữ đám mây", bạn cần thực hiện:
1.  **Tạo dự án:** Tại [Firebase Console](https://console.firebase.google.com/).
2.  **Xác thực:** Kích hoạt phương thức **Anonymous** (Ẩn danh) trong phần *Authentication*.
3.  **Firestore Database:** Tạo cơ sở dữ liệu ở **Production Mode**.
4.  **Cấu hình Quyền (Rules):** Cho phép người dùng ẩn danh đọc/ghi:
    ```javascript
    service cloud.firestore {
      match /databases/{database}/documents {
        match /{document=**} {
          allow read, write: if request.auth != null;
        }
      }
    }
    ```
5.  **Biến môi trường:** Sao chép các thông số SDK vào file `.env`.

#### 2. Triển khai Backend (Render) & Frontend (Vercel)
- **Backend:** Chạy Flask trên Render, thiết lập `GEMINI_API_KEY`. Bỏ qua bước này nếu bạn không sử dụng backend của riêng mình. 
- **Frontend:** Sử dụng Preset `Vite` trên Vercel, cấu hình đầy đủ các biến `VITE_FIREBASE_...`.

### 🛡 Tính bền vững: Hạ cấp mượt mà
Hệ thống có cơ chế tự bảo vệ: Nếu cấu hình Firebase lỗi, ứng dụng sẽ tự động khóa các tính năng Cloud và chuyển sang chế độ sử dụng Tệp cục bộ, đảm bảo không bị crash giữa chừng.

### 💡 Lợi thế HDVO: "Tự gãi lưng cho chính mình"
Cho phép người dùng can thiệp vào logic dự báo bằng ngôn ngữ tự nhiên:
* **Tình huống:** Trạm bơm bị hỏng bất ngờ.
* **Can thiệp:** Người dùng nhập gợi ý: *"Máy bơm số 2 đang bảo trì."*
* **Kết quả:** AI viết lại hàm logic dự báo ngay lập tức để thích ứng với bối cảnh mới mà không cần cập nhật dữ liệu số hay sửa mã nguồn.

---
*Developed for sustainable water resource management. Open for contributions!*
