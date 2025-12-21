// tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  // Quan trọng: Thêm đường dẫn đến tất cả các tệp JSX/JS/TSX của bạn
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
