import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { initializeApp, getApps, getApp } from 'firebase/app';
import { getAuth, signInAnonymously, signInWithCustomToken, onAuthStateChanged } from 'firebase/auth';
import { getFirestore, doc, setDoc, getDoc } from 'firebase/firestore';

import { addDays, formatISO } from 'date-fns';

// Import necessary components from Recharts
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    Legend, ReferenceLine, ScatterChart, Scatter, BarChart, Bar
} from 'recharts';

// --- Constants and Utility Functions ---
const TRAINING_PERIOD = 14;
const LEADING_PERIOD = 7;
const PREDICTING_PERIOD = 7;
// Khai báo biến môi trường mới
const appId = "ground-water_firestore-app"; // typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';
const FETCH_API_URL = import.meta.env.VITE_HDVO_API_URL;
const FETCH_CLIENT_KEY = import.meta.env.VITE_FETCH_CLIENT_KEY;
const firebaseConfig = {
    apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
    authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
    projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID}; // typeof __firebase_config !== 'undefined' ? JSON.parse(__firebase_config) : {};
const initialAuthToken = (typeof window !== 'undefined' && window.__initial_auth_token) 
                         ? window.__initial_auth_token 
                         : null;

// Hàm mặc định cho AI để dự đoán mực nước ngầm
// Output: Một MẢNG 7 số duy nhất đại diện cho GWL dự đoán trong 7 ngày tới.
const DEFAULT_PREDICTION_FUNCTION_BODY = `
// Hàm dự đoán mực nước ngầm (GWL) trong ${PREDICTING_PERIOD} ngày tới
// Inputs:
// - historicalGroundwaterData: [{ wellId: string, timestamp: string, gwl: number, ec: number }]
// - historicalWaterQualityData: [{ wellId: string, timestamp: string, ph: number, do: number, turbidity: number }]\\
// - historicalWeatherForecast: [{ wellId: string, timestamp: string, precipitation: number, temperature: number }]
// - historicalWaterUsage: [{ wellId: string, timestamp: string, pumping: number, consumption: number }]
// Output: Một MẢNG ${PREDICTING_PERIOD} số duy nhất đại diện cho GWL dự đoán trong ${PREDICTING_PERIOD} ngày tới.

// Đây là một hàm mặc định đơn giản. AI sẽ cải thiện nó.
// Chỉ lấy giá trị GWL cuối cùng làm dự đoán và lặp lại ${PREDICTING_PERIOD} lần.
if (historicalGroundwaterData.length === 0) {
    const ret = [];
    for (let i = 0; i < ${PREDICTING_PERIOD}; i++) {
        ret.push(0);
    }
    return ret; // Không có dữ liệu, trả về mảng 0
}
const latestGwl = historicalGroundwaterData[historicalGroundwaterData.length - 1].gwl;
return Array(${PREDICTING_PERIOD}).fill(latestGwl); // Dự đoán ${PREDICTING_PERIOD} ngày tới bằng giá trị cuối cùng
`;

const SIGMA_FACTOR_95_PERCENT_CI = 1.96; // For 95% Confidence Interval in a normal distribution

// NEW: Performance Metric Functions
const calculateRMSE = (predictions, actuals) => {
    if (predictions.length === 0 || actuals.length === 0 || predictions.length !== actuals.length) return Infinity;
    const squaredErrors = predictions.map((pred, i) => Math.pow(pred - actuals[i], 2));
    const meanSquaredError = squaredErrors.reduce((sum, val) => sum + val, 0) / squaredErrors.length;
    return Math.sqrt(meanSquaredError);
};

const calculateMSE = (predictions, actuals) => {
    if (predictions.length === 0 || actuals.length === 0 || predictions.length !== actuals.length) return Infinity;
    const squaredErrors = predictions.map((pred, i) => Math.pow(pred - actuals[i], 2));
    return squaredErrors.reduce((sum, val) => sum + val, 0) / squaredErrors.length;
};

const calculateMAE = (predictions, actuals) => {
    if (predictions.length === 0 || actuals.length === 0 || predictions.length !== actuals.length) return Infinity;
    const absoluteErrors = predictions.map((pred, i) => Math.abs(pred - actuals[i]));
    return absoluteErrors.reduce((sum, val) => sum + val, 0) / absoluteErrors.length;
};

const performanceMetricsCalculators = {
    'rmse': {
        name: 'RMSE (Sai số trung bình bình phương gốc)',
        calculate: calculateRMSE,
        unit: 'm'
    },
    'mse': {
        name: 'MSE (Sai số bình phương trung bình)',
        calculate: calculateMSE,
        unit: 'm²'
    },
    'mae': {
        name: 'MAE (Sai số tuyệt đối trung bình)',
        calculate: calculateMAE,
        unit: 'm'
    }
};

// NEW: Function to render the AI function signature
const renderAiFunctionSignature = () => {
    return `function predictGroundwaterLevel(historicalGroundwaterData, historicalWaterQualityData, historicalWeatherForecast, historicalWaterUsage) {`;
};

// Helper function to analyze malformed AI responses or runtime errors
const analyzeMalformedAiResponse = async (errorDetails, contextHint) => {
    const p_body = JSON.stringify({
        errorDetails: errorDetails,
        contextHint: contextHint
    });
    try {
        const apiKey = ""; // Canvas will provide API key
        const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=${apiKey}`;
        const response = await fetch(`${FETCH_API_URL}/api/v2/ai1_fetch/raw_text`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json', 
                'X-Client-Key': FETCH_CLIENT_KEY
            },
            body: JSON.stringify({
                errorDetails: errorDetails, 
                contextHint: contextHint
            })
        });
        const rawText = await response.text();
        let result;
        result = JSON.parse(rawText);
        if (result.candidates && result.candidates.length > 0 && result.candidates[0].content && result.candidates[0].content.parts && result.candidates[0].content.parts.length > 0) {
            return result.candidates[0].content.parts[0].text;
        }
        return "Cannot analyze malformed AI response.";
    } catch (error) {
        console.error("Error analyzing malformed AI response:", error);
        return `Error analyzing malformed AI response: ${error.message}`;
    }
};

// Function to convert Markdown to HTML
const renderMarkdownToHtml = (markdownText) => {
    if (!markdownText) return '';
    let html = markdownText;
    html = html.replace(/^###### (.*$)/gim, '<h6>$1</h6>');
    html = html.replace(/^##### (.*$)/gim, '<h5>$1</h5>');
    html = html.replace(/^#### (.*$)/gim, '<h4>$1</h4>');
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    html = html.replace(/_(.*?)_/g, '<em>$1</em>');
    html = html.replace(/^- (.*$)/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*?<\/li>\s*)+/gs, '<ul>$&</ul>');
    html = html.replace(/(\r\n|\n|\r)/gm, '<br/>'); // Convert newlines to <br/>
    return html;
};

// Utility function to truncate text to a specific number of lines
const truncateToLines = (text, maxLines) => {
    if (!text) return '';
    const lines = text.split('\n').filter(line => line.trim() !== ''); // Split by newline and remove empty lines
    if (lines.length <= maxLines) {
        return text;
    }
    // Join the first 'maxLines' lines and add "..."
    return lines.slice(0, maxLines).join('\n') + '...';
};

// Statistical Helper Functions (placed outside App component)
// Function to calculate mean
const calculateMean = (arr) => arr.length ? arr.reduce((sum, val) => sum + val, 0) / arr.length : 0;

// Function to calculate standard deviation
const calculateStandardDeviation = (arr) => {
    if (arr.length < 2) return 0;
    const mean = calculateMean(arr);
    const variance = arr.reduce((sum, val) => sum + (val - mean) ** 2, 0) / (arr.length - 1); // Sample standard deviation
    return Math.sqrt(variance);
};

// Function to calculate skewness (third standardized moment)
const calculateSkewness = (arr) => {
    if (arr.length < 3) return 0; // Skewness requires at least 3 data points
    const mean = calculateMean(arr);
    const stdDev = calculateStandardDeviation(arr);
    if (stdDev === 0) return 0;
    const n = arr.length;
    let sum = 0;
    for (let i = 0; i < n; i++) {
        sum += ((arr[i] - mean) / stdDev) ** 3;
    }
    // Corrected to use sample skewness formula: sum / ((n-1) * (n-2) * stdDev^3) * n
    // For simplicity and common visual interpretation, using population skewness here
    return sum / n;
};

// Function to calculate kurtosis (fourth standardized moment)
const calculateKurtosis = (arr) => {
    if (arr.length < 4) return 0; // Kurtosis requires at least 4 data points
    const mean = calculateMean(arr);
    const stdDev = calculateStandardDeviation(arr);
    if (stdDev === 0) return 0;
    const n = arr.length;
    let sum = 0;
    for (let i = 0; i < n; i++) {
        sum += ((arr[i] - mean) / stdDev) ** 4;
    }
    // Corrected to use sample kurtosis formula: ([n(n+1)]/[(n-1)(n-2)(n-3)])*sum(z^4) - [3(n-1)^2]/[(n-2)(n-3)]
    // For simplicity and common visual interpretation, using excess kurtosis (population) here
    return (sum / n) - 3;
};

// Function to calculate Autocorrelation Function (ACF)
const calculateACF = (data, maxLag) => {
    const n = data.length;
    if (n === 0) return [];
    const mean = calculateMean(data);
    let variance = 0;
    for (let i = 0; i < n; i++) {
        variance += (data[i] - mean) ** 2;
    }
    if (variance === 0) return Array(maxLag + 1).fill({ lag: 0, value: 0, ciUpper: 0, ciLower: 0 }); // Avoid division by zero

    const acfValues = [];
    // Confidence interval bound for visual comparison (approximate 95% CI)
    const ciBound = 1.96 / Math.sqrt(n); // For large N, CI is approx +/- 1.96/sqrt(N)

    for (let k = 0; k <= maxLag; k++) {
        if (n - k <= 0) break; // Ensure there are enough points for the lag
        let numerator = 0;
        for (let i = 0; i < n - k; i++) {
            numerator += (data[i] - mean) * (data[i + k] - mean);
        }
        const value = numerator / variance;
        acfValues.push({ lag: k, value: value, ciUpper: ciBound, ciLower: -ciBound });
    }
    return acfValues;
};

// Function to calculate QQ Plot data
const calculateQQPlotData = (data) => {
    if (data.length === 0) return [];
    const sortedData = [...data].sort((a, b) => a - b);
    const n = sortedData.length;
    const qqData = [];

    // Simple approach for theoretical quantiles:
    // Generate evenly spaced points that would be expected from a normal distribution.
    // This is a common visual approximation for QQ plots in charting libraries.
    // For more robust statistical QQ plots, one would use inverse CDF of standard normal distribution.
    const minVal = sortedData[0];
    const maxVal = sortedData[n - 1];
    const range = maxVal - minVal;

    for (let i = 0; i < n; i++) {
        // Linear interpolation for theoretical quantiles
        const theoreticalQuantile = (i / (n - 1)) * range + minVal;
        qqData.push({ observed: sortedData[i], theoretical: theoreticalQuantile });
    }
    return qqData;
};

// Function to calculate Histogram Bins
const calculateHistogramBins = (data, numBins) => {
    if (data.length === 0 || numBins <= 0) return [];

    const minVal = Math.min(...data);
    const maxVal = Math.max(...data);
    const range = maxVal - minVal;

    if (range === 0) { // All data points are the same, create a single bin
        return [{ bin: minVal.toFixed(2), count: data.length, midPoint: minVal }];
    }

    const binWidth = range / numBins;
    const bins = Array(numBins).fill(0);
    const binLabels = [];

    for (let i = 0; i < data.length; i++) {
        let binIndex = Math.floor((data[i] - minVal) / binWidth);
        // Handle the case where maxVal falls exactly into the last bin's upper bound
        if (binIndex === numBins) {
            binIndex--;
        }
        if (binIndex >= 0 && binIndex < numBins) {
            bins[binIndex]++;
        }
    }

    for (let i = 0; i < numBins; i++) {
        const lowerBound = minVal + i * binWidth;
        const upperBound = minVal + (i + 1) * binWidth;
        binLabels.push({
            bin: `${lowerBound.toFixed(2)} - ${upperBound.toFixed(2)}`, // Display range for bin
            count: bins[i],
            midPoint: (lowerBound + upperBound) / 2
        });
    }
    return binLabels;
};

// Hàm để tính Khoảng Dự đoán (Prediction Intervals) bằng phương pháp Bootstrapping của Phần dư.
// Đã điều chỉnh để mô phỏng đường dẫn toàn phần (full path simulation) và tích lũy lỗi.
const calculateBootstrapPI = (residuals, predictions, startStep=0, numSimulations = 1000, confidenceLevel = 0.95) => {
    if (!predictions || predictions.length === 0) {
        console.warn("calculateBootstrapPI: Không có dữ liệu dự đoán. Trả về mảng rỗng.");
        return [];
    }
    if (!residuals || residuals.length === 0) {
        console.warn("calculateBootstrapPI: Không có phần dư lịch sử. Trả về PI với giá trị null.");
        return predictions.map(() => ({ upper: null, lower: null }));
    }

    const alpha = 1 - confidenceLevel;
    const lowerPercentile = alpha / 2 * 100;
    const upperPercentile = (1 - alpha / 2) * 100;

    // Tính toán trung bình phần dư để hiệu chỉnh thiên lệch (bias correction)
    const meanResidual = calculateMean(residuals);
    // Tạo tập hợp các phần dư đã hiệu chỉnh thiên lệch
    const biasCorrectedResiduals = residuals.map(r => r - meanResidual);

    // simulationDistributions sẽ chứa tất cả các giá trị dự đoán mô phỏng cho mỗi bước thời gian
    const simulationDistributions = Array(predictions.length).fill(null).map(() => []);

    // Thực hiện số lần mô phỏng Bootstrapping đã chỉ định
    for (let s = 0; s < numSimulations; s++) {
        let cumulativeError = 0; // Lỗi tích lũy cho đường dẫn mô phỏng hiện tại
        // Với mỗi lần mô phỏng, chúng ta tạo một đường dẫn dự đoán tương lai
        // Đường dẫn này được tạo bằng cách thêm các phần dư được lấy mẫu vào các dự đoán điểm,
        // CÁC PHẦN DƯ NÀY SẼ ĐƯỢC TÍCH LŨY.
        for (let i = 0; i < startStep; i++) {
            // Lấy mẫu MỘT phần dư ngẫu nhiên từ tập hợp các phần dư ĐÃ HIỆU CHỈNH THIÊN LỆCH
            const randomIndex = Math.floor(Math.random() * biasCorrectedResiduals.length);
            const sampledResidual = biasCorrectedResiduals[randomIndex];

            // Tích lũy lỗi. Lỗi cho bước i sẽ bao gồm lỗi ngẫu nhiên của bước đó cộng dồn từ các bước trước.
            cumulativeError += sampledResidual; 
            
            // Giá trị mô phỏng cho bước này là dự đoán điểm ban đầu cộng với lỗi tích lũy.
            // Điều này tái tạo cách lỗi tích lũy trong các mô hình chuỗi thời gian như ARIMA.
            const simulatedFutureValue = predictions[i] + cumulativeError;
            
            // Đẩy giá trị mô phỏng vào phân phối chính xác cho điểm dự đoán này.
            simulationDistributions[i].push(simulatedFutureValue);
        }
    }

    // Sau khi chạy tất cả các mô phỏng, tính toán các giới hạn Khoảng Dự đoán (PI) cho từng điểm thời gian
    const piBounds = [];
    for (let i = 0; i < predictions.length; i++) {
        if (simulationDistributions[i].length > 0) { // Đảm bảo có dữ liệu để tính
            simulationDistributions[i].sort((a, b) => a - b);

            const lowerIndex = Math.floor(simulationDistributions[i].length * lowerPercentile / 100);
            const upperIndex = Math.ceil(simulationDistributions[i].length * upperPercentile / 100) - 1;

            piBounds.push({
                lower: simulationDistributions[i][lowerIndex],
                upper: simulationDistributions[i][upperIndex]
            });
        } else {
            piBounds.push({ lower: null, upper: null }); // Không có dữ liệu để tính PI
        }
    }
    return piBounds;
};


// --- Reusable Components ---

// Modal Component for general messages/confirmations
const MessageModal = ({ isOpen, onClose, title, message, type = 'info' }) => {
    if (!isOpen) return null;

    let bgColor = 'bg-blue-100';
    let borderColor = 'border-blue-400';
    let textColor = 'text-blue-700';

    if (type === 'success') {
        bgColor = 'bg-green-100';
        borderColor = 'border-green-400';
        textColor = 'text-green-700';
    } else if (type === 'error') {
        bgColor = 'bg-red-100';
        borderColor = 'border-red-400';
        textColor = 'text-red-700';
    } else if (type === 'warning') {
        bgColor = 'bg-yellow-100';
        borderColor = 'border-yellow-400';
        textColor = 'text-yellow-700';
    }

    return (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-75 flex items-center justify-center z-[1000] p-4">
            <div className={`bg-white p-8 rounded-lg shadow-xl max-w-sm w-full ${bgColor} border ${borderColor}`}>
                <h2 className={`text-2xl font-bold mb-4 ${textColor}`}>{title}</h2>
                <p className={`mb-6 ${textColor}`}>{message}</p>
                <div className="flex justify-end">
                    <button
                        className="bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                        onClick={onClose}
                    >
                        Đóng
                    </button>
                </div>
            </div>
        </div>
    );
};

// Confirm Modal Component
const ConfirmModal = ({ isOpen, onClose, onConfirm, title, message, type = 'warning' }) => {
    if (!isOpen) return null;

    let bgColor = 'bg-yellow-100';
    let borderColor = 'border-yellow-400';
    let textColor = 'text-yellow-700';

    if (type === 'error') {
        bgColor = 'bg-red-100';
        borderColor = 'border-red-400';
        textColor = 'text-red-700';
    }

    return (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-75 flex items-center justify-center z-[1000] p-4">
            <div className="bg-white p-8 rounded-lg shadow-xl max-w-sm w-full ${bgColor} border ${borderColor}">
                <h2 className={`text-2xl font-bold mb-4 ${textColor}`}>{title}</h2>
                <p className={`mb-6 ${textColor}`}>{message}</p>
                <div className="flex justify-end gap-4">
                    <button
                        className="bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                        onClick={onClose}
                    >
                        Hủy
                    </button>
                    <button
                        className={`bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline`}
                        onClick={onConfirm}
                    >
                        Xác nhận
                    </button>
                </div>
            </div>
        </div>
    );
};


// General JSON input modal
const JsonInputModal = React.memo(({ isOpen, onClose, onSubmit, title, jsonInput, onJsonInputChange, selectedFileName, onFileChange, errorMessage, fileInputKey }) => {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-75 flex items-center justify-center z-[1000] p-4">
            <div className="bg-white p-8 rounded-lg shadow-xl max-w-lg w-full">
                <h2 className="text-2xl font-bold mb-4 text-slate-800">{title}</h2>
                {errorMessage && (
                    <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4" role="alert">
                        <strong className="font-bold">Lỗi:</strong>
                        <span className="block sm:inline"> {errorMessage}</span>
                    </div>
                )}
                <div className="mb-4">
                    <label htmlFor="json-textarea" className="block text-slate-700 text-sm font-bold mb-2">
                        Dán dữ liệu JSON vào đây:
                    </label>
                    <textarea
                        id="json-textarea"
                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline h-40 font-mono text-sm"
                        value={jsonInput}
                        onChange={onJsonInputChange}
                        placeholder="[{ 'timestamp': '2023-01-01T10:00:00Z', 'gwl': 10.5, 'ec': 500 }]"
                    ></textarea>
                </div>
                <div className="mb-4">
                    <label htmlFor="json-file-upload" className="block text-slate-700 text-sm font-bold mb-2">
                        Hoặc tải lên từ tệp JSON:
                    </label>
                    <input
                        key={fileInputKey} // Use key to reset file input
                        type="file"
                        id="json-file-upload"
                        accept=".json"
                        onChange={onFileChange}
                        className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                    />
                    {selectedFileName && <p className="mt-2 text-sm text-gray-600">Đã chọn tệp: <span className="font-semibold">{selectedFileName}</span></p>}
                </div>
                <div className="flex justify-end gap-4">
                    <button
                        className="bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                        onClick={onSubmit}
                    >
                        Xác nhận
                    </button>
                    <button
                        className="bg-indigo-500 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                        onClick={onClose}
                    >
                        Hủy
                    </button>
                </div>
            </div>
        </div>
    );
});

// Collapsible Section Component (Enhanced to accept initialOpen and custom colors)
const CollapsibleSection = React.memo(({ title, children, initialOpen = true, bgColor = 'bg-blue-50', borderColor = 'border-blue-200', textColor = 'text-blue-800', headerBgColor = 'bg-blue-100', headerTextColor = 'text-blue-900' }) => {
    const [isOpen, setIsOpen] = useState(initialOpen);

    return (
        <div className={`mt-4 ${bgColor} p-4 rounded-lg border ${borderColor} ${textColor} text-sm`}>
            <button
                className={`w-full flex justify-between items-center text-left font-bold ${headerTextColor} mb-2 cursor-pointer focus:outline-none`}
                onClick={() => setIsOpen(!isOpen)}
            >
                <span>{title}</span>
                <span>
                    {isOpen ? (
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 01-1.414 0l-4-4a1 1 010-1.414z" clipRule="evenodd" />
                        </svg>
                    ) : (
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 01-1.414-1.414l4-4a1 1 011.414 0l4 4a1 1 010 1.414z" clipRule="evenodd" />
                        </svg>
                    )}
                </span>
            </button>
            <div className={`overflow-hidden transition-all duration-300 ease-in-out ${isOpen ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'}`}>
                <div className="p-2 space-y-3 overflow-y-auto max-h-80">
                    {children}
                </div>
            </div>
        </div>
    );
});

// Define a generic MarkdownRendererComponent to encapsulate display logic for Markdown content
const MarkdownRendererComponent = React.memo(({ content, placeholderText }) => {
    return (
        <div className="bg-gray-100 rounded-lg p-4 prose max-w-none text-slate-800 mb-6">
            {content ? (
                <div dangerouslySetInnerHTML={{ __html: renderMarkdownToHtml(content) }} />
            ) : (
                <p className="text-slate-600">{placeholderText}</p>
            )}
        </div>
    );
});


// Recharts specific components (existing ones)
const GroundwaterChart = ({ data }) => (
    <ResponsiveContainer width="100%" height="100%">
        <LineChart
            data={data.map(d => ({ ...d, timestamp: new Date(d.timestamp).toLocaleDateString('vi-VN') }))}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis label={{ value: 'Mực nước ngầm (m bgs)', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="gwl" stroke="#3b82f6" activeDot={{ r: 8 }} name="Mực nước ngầm (GWL)" />
        </LineChart>
    </ResponsiveContainer>
);

const EcChart = ({ data }) => (
    <ResponsiveContainer width="100%" height="100%">
        <LineChart
            data={data.map(d => ({ ...d, timestamp: new Date(d.timestamp).toLocaleDateString('vi-VN') }))}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis label={{ value: 'Độ dẫn điện (µS/cm)', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="ec" stroke="#10b981" activeDot={{ r: 8 }} name="Độ dẫn điện (EC)" />
        </LineChart>
    </ResponsiveContainer>
);

const WaterQualityChart = ({ data }) => (
    <ResponsiveContainer width="100%" height="100%">
        <LineChart
            data={data.map(d => ({ ...d, timestamp: new Date(d.timestamp).toLocaleDateString('vi-VN') }))}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis label={{ value: 'Giá trị', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="ph" stroke="#f59e0b" activeDot={{ r: 8 }} name="pH" />
            <Line type="monotone" dataKey="do" stroke="#06b6d4" activeDot={{ r: 8 }} name="DO (mg/L)" />
            <Line type="monotone" dataKey="turbidity" stroke="#6366f1" activeDot={{ r: 8 }} name="Độ đục (NTU)" />
        </LineChart>
    </ResponsiveContainer>
);

const WeatherChart = ({ data }) => (
    <ResponsiveContainer width="100%" height="100%">
        <LineChart
            data={data.map(d => ({ ...d, timestamp: new Date(d.timestamp).toLocaleDateString('vi-VN') }))}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis label={{ value: 'Giá trị', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="precipitation" stroke="#22c55e" activeDot={{ r: 8 }} name="Lượng mưa (mm)" />
            <Line type="monotone" dataKey="temperature" stroke="#ef4444" activeDot={{ r: 8 }} name="Nhiệt độ (°C)" />
        </LineChart>
    </ResponsiveContainer>
);

const UsageChart = ({ data }) => (
    <ResponsiveContainer width="100%" height="100%">
        <LineChart
            data={data.map(d => ({ ...d, timestamp: new Date(d.timestamp).toLocaleDateString('vi-VN') }))}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis label={{ value: 'Lượng (m³/ngày)', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="pumping" stroke="#a855f7" activeDot={{ r: 8 }} name="Lượng bơm (m³/ngày)" />
            <Line type="monotone" dataKey="consumption" stroke="#ec4899" activeDot={{ r: 8 }} name="Tiêu thụ (m³/ngày)" />
        </LineChart>
    </ResponsiveContainer>
);

// UPDATED PredictionRecharts component (This is used in prediction-optimization tab, so keep it)
const PredictionRecharts = ({ data }) => (
    <ResponsiveContainer width="100%" height="100%">
        <LineChart
            data={data}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" tickFormatter={(tickItem) => new Date(tickItem).toLocaleDateString('vi-VN')} />
            {/* Left Y-Axis for Actual and Predicted GWL */}
            <YAxis yAxisId="left" label={{ value: 'Mực nước ngầm (m bgs)', angle: -90, position: 'insideLeft' }} />
            {/* Right Y-Axis for Error */}
            <YAxis yAxisId="right" orientation="right" stroke="#8884d8" label={{ value: 'Sai số (m)', angle: 90, position: 'insideRight' }} />
            <Tooltip />
            <Legend />
            <Line yAxisId="left" type="monotone" dataKey="actualGwl" stroke="#3b82f6" activeDot={{ r: 8 }} name="Mực nước ngầm thực tế" />
            <Line yAxisId="left" type="monotone" dataKey="predictedGwl" stroke="#f59e0b" strokeDasharray="5 5" activeDot={{ r: 8 }} name="GWL dự đoán" />
            <Line yAxisId="right" type="monotone" dataKey="predictionError" stroke="#8884d8" activeDot={{ r: 8 }} name="Sai số" />
            {/* New Lines for Prediction Intervals (PI) */}
            <Line yAxisId="left" type="monotone" dataKey="predictedGwlUpperPI" stroke="#f59e0b" strokeDasharray="2 2" strokeWidth={1} name="PI trên" dot={false} />
            <Line yAxisId="left" type="monotone" dataKey="predictedGwlLowerPI" stroke="#f59e0b" strokeDasharray="2 2" strokeWidth={1} name="PI dưới" dot={false} />
        </LineChart>
    </ResponsiveContainer>
);


// Components for Statistical Validation Tab
const ResidualsVsTimeChart = ({ data }) => (
    <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" tickFormatter={(tickItem) => new Date(tickItem).toLocaleDateString('vi-VN')} />
            <YAxis label={{ value: 'Phần dư (m)', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <ReferenceLine y={0} stroke="#8884d8" strokeDasharray="3 3" label={{ value: 'Zero Residual', position: 'insideTopRight' }} />
            <Line type="monotone" dataKey="error" stroke="#82ca9d" name="Sai số dự đoán" dot={false} />
        </LineChart>
    </ResponsiveContainer>
);

const ResidualsHistogramChart = ({ data }) => (
    <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bin" />
            <YAxis label={{ value: 'Tần suất', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Bar dataKey="count" fill="#8884d8" name="Tần suất Phần dư" />
        </BarChart>
    </ResponsiveContainer>
);

const QQPlotChart = ({ data }) => (
    <ResponsiveContainer width="100%" height={300}>
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid />
            <XAxis type="number" dataKey="theoretical" name="Phân vị Lý thuyết Chuẩn" unit="" domain={['auto', 'auto']} />
            <YAxis type="number" dataKey="observed" name="Phân vị Thực tế Phần dư" unit="m" domain={['auto', 'auto']} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Legend />
            <Scatter name="Phần dư" data={data} fill="#8884d8" />
            {/* Reference line for normal distribution (y=x) */}
            <ReferenceLine x1={Math.min(...data.map(d => d.theoretical))} y1={Math.min(...data.map(d => d.theoretical))} 
                           x2={Math.max(...data.map(d => d.theoretical))} y2={Math.max(...data.map(d => d.theoretical))} 
                           stroke="red" strokeDasharray="3 3" label={{ value: 'Lý thuyết Chuẩn', position: 'insideBottomRight' }} />
        </ScatterChart>
    </ResponsiveContainer>
);

const ACFPlotChart = ({ data }) => (
    <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="lag" label={{ value: 'Độ trễ', position: 'insideBottom', offset: 0 }} />
            <YAxis label={{ value: 'Hệ số tự tương quan', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Bar dataKey="value" fill="#82ca9d" name="ACF" />
            {/* Confidence Interval Lines */}
            {data.length > 0 && (
                <>
                    <ReferenceLine y={data[0]?.ciUpper || 0} stroke="blue" strokeDasharray="3 3" label={{ value: '95% CI Trên', position: 'insideTopRight' }} />
                    <ReferenceLine y={data[0]?.ciLower || 0} stroke="blue" strokeDasharray="3 3" label={{ value: '95% CI Dưới', position: 'insideBottomRight' }} />
                </>
            )}
        </BarChart>
    </ResponsiveContainer>
);

const RawGroundwaterDataChart = ({ data }) => (
    <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data.map(d => ({ ...d, timestamp: new Date(d.timestamp).toLocaleDateString('vi-VN') }))}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis label={{ value: 'Mực nước ngầm (m bgs)', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="gwl" stroke="#3b82f6" activeDot={{ r: 8 }} name="Mực nước ngầm (GWL)" />
        </LineChart>
    </ResponsiveContainer>
);

// New Well Management Component
const WellManagement = ({ dataStorageMode, db, userId, showMessage, showConfirm, closeConfirmModal, wellLocations, setWellLocations, setSelectedWellId, selectedWellId, setGroundwaterData, setWaterQualityData, setWeatherForecast, setWaterUsage, setAllWellForecasts }) => {
    const [wellId, setWellId] = useState('');
    const [wellName, setWellName] = useState('');
    const [wellLat, setWellLat] = useState('');
    const [wellLon, setWellLon] = useState('');
    const [editMode, setEditMode] = useState(false);
    const [currentWellToEdit, setCurrentWellToEdit] = useState(null);

    // Removed basePath for public data collections

    const handleAddOrUpdateWell = useCallback(async () => {
        if (!wellId || !wellName || wellLat === '' || wellLon === '') {
            showMessage('Lỗi', 'Vui lòng điền đầy đủ thông tin giếng.', 'error');
            return;
        }
        if (isNaN(parseFloat(wellLat)) || isNaN(parseFloat(wellLon))) {
            showMessage('Lỗi', 'Vĩ độ và Kinh độ phải là số hợp lệ.', 'error');
            return;
        }

        const wellData = {
            id: wellId,
            name: wellName,
            lat: parseFloat(wellLat),
            lon: parseFloat(wellLon)
        };

        try {
            // When adding/updating well, update the local state directly.
            // When saving/loading session, the entire wellLocations array will be managed.
            setWellLocations(prev => {
                const existingIndex = prev.findIndex(w => w.id === wellId);
                if (existingIndex > -1) {
                    return prev.map((w, idx) => idx === existingIndex ? wellData : w);
                } else {
                    return [...prev, wellData];
                }
            });
            showMessage('Thành công', `Giếng '${wellName}' đã được ${editMode ? 'cập nhật' : 'thêm mới'}.`, 'success');
            setWellId('');
            setWellName('');
            setWellLat('');
            setWellLon('');
            setEditMode(false);
            setCurrentWellToEdit(null);
            setSelectedWellId(wellId); // Automatically select the newly added/updated well
        } catch (error) {
            showMessage('Lỗi', `Không thể ${editMode ? 'cập nhật' : 'thêm'} giếng: ${error.message}`, 'error');
            console.error("Error adding/updating well:", error);
        }
    }, [wellId, wellName, wellLat, wellLon, showMessage, editMode, setSelectedWellId, setWellLocations]);

    const handleDeleteWell = useCallback(async (id, wellName) => {
        showConfirm(
            'Xác nhận Xóa',
            `Bạn có chắc chắn muốn xóa giếng '${wellName}' (ID: ${id})? Thao tác này không thể hoàn tác và sẽ xóa tất cả dữ liệu liên quan đến giếng này.`,
            async () => {
                try {
                    // Update local state directly. Session save/load will handle persistence.
                    setWellLocations(prev => prev.filter(well => well.id !== id));
                    setGroundwaterData(prev => prev.filter(data => data.wellId !== id));
                    setWaterQualityData(prev => prev.filter(data => data.wellId !== id));
                    setWeatherForecast(prev => prev.filter(data => data.wellId !== id));
                    setWaterUsage(prev => prev.filter(data => data.wellId !== id));
                    setAllWellForecasts(prev => { // Clear predictions for this well
                        const newState = { ...prev };
                        delete newState[id];
                        return newState;
                    });


                    showMessage('Thành công', `Giếng '${wellName}' và tất cả dữ liệu liên quan đã được xóa.`, 'success');
                    if (selectedWellId === id) {
                        setSelectedWellId(''); // Deselect if the current well is deleted
                    }
                } catch (error) {
                    showMessage('Lỗi', `Không thể xóa giếng: ${error.message}`, 'error');
                    console.error("Error deleting well:", error);
                } finally {
                    closeConfirmModal();
                }
            },
            () => {
                closeConfirmModal();
                showMessage('Thông báo', 'Thao tác xóa đã bị hủy.', 'info');
            }
        );
    }, [showMessage, selectedWellId, setSelectedWellId, showConfirm, closeConfirmModal, setWellLocations, setGroundwaterData, setWaterQualityData, setWeatherForecast, setWaterUsage, setAllWellForecasts]);

    const handleEditWellClick = useCallback((well) => {
        setWellId(well.id);
        setWellName(well.name);
        setWellLat(well.lat.toString());
        setWellLon(well.lon.toString());
        setEditMode(true);
        setCurrentWellToEdit(well);
    }, []);

    const handleCancelEdit = useCallback(() => {
        setWellId('');
        setWellName('');
        setWellLat('');
        setWellLon('');
        setEditMode(false);
        setCurrentWellToEdit(null);
    }, []);

    return (
        <div className="mt-8 p-6 bg-white rounded-xl shadow-lg border border-slate-200">
            <h3 className="text-2xl font-bold mb-4 text-slate-800">Quản lý Giếng</h3>
            <p className="text-slate-600 mb-4">Thêm, chỉnh sửa hoặc xóa thông tin về các giếng quan trắc.</p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                <div>
                    <label htmlFor="well-id" className="block text-slate-700 text-sm font-bold mb-2">ID Giếng:</label>
                    <input
                        type="text"
                        id="well-id"
                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                        value={wellId}
                        onChange={(e) => setWellId(e.target.value)}
                        placeholder="VD: GW_WELL_006"
                        disabled={editMode}
                    />
                </div>
                <div>
                    <label htmlFor="well-name" className="block text-slate-700 text-sm font-bold mb-2">Tên Giếng:</label>
                    <input
                        type="text"
                        id="well-name"
                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                        value={wellName}
                        onChange={(e) => setWellName(e.target.value)}
                        placeholder="VD: Giếng công nghiệp mới"
                    />
                </div>
                <div>
                    <label htmlFor="well-lat" className="block text-slate-700 text-sm font-bold mb-2">Vĩ độ:</label>
                    <input
                        type="number"
                        id="well-lat"
                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                        value={wellLat}
                        onChange={(e) => setWellLat(e.target.value)}
                        placeholder="VD: 10.80"
                        step="0.01"
                    />
                </div>
                <div>
                    <label htmlFor="well-lon" className="block text-slate-700 text-sm font-bold mb-2">Kinh độ:</label>
                    <input
                        type="number"
                        id="well-lon"
                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                        value={wellLon}
                        onChange={(e) => setWellLon(e.target.value)}
                        placeholder="VD: 106.75"
                        step="0.01"
                    />
                </div>
            </div>
            <div className="flex gap-4">
                <button
                    className="px-6 py-2 bg-blue-600 text-white font-bold rounded-lg shadow-md hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    onClick={handleAddOrUpdateWell}
                    disabled={dataStorageMode === 'firestore' && (!db || !userId)} // Disable for Firestore if not authenticated
                >
                    {editMode ? 'Cập nhật Giếng' : 'Thêm Giếng Mới'}
                </button>
                {editMode && (
                    <button
                        className="px-6 py-2 bg-gray-500 text-white font-bold rounded-lg shadow-md hover:bg-gray-600 transition-colors"
                        onClick={handleCancelEdit}
                    >
                        Hủy
                    </button>
                )}
            </div>

            <CollapsibleSection title="Danh sách Giếng hiện có">
                {wellLocations.length > 0 ? (
                    <div className="overflow-x-auto">
                        <table className="min-w-full bg-white border border-gray-200 rounded-lg">
                            <thead>
                                <tr>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">ID Giếng</th>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">Tên Giếng</th>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">Vĩ độ</th>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">Kinh độ</th>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">Hành động</th>
                                </tr>
                            </thead>
                            <tbody>
                                {wellLocations.map(well => (
                                    <tr key={well.id}>
                                        <td className="py-2 px-4 border-b text-sm text-gray-800">{well.id}</td>
                                        <td className="py-2 px-4 border-b text-sm text-gray-800">{well.name}</td>
                                        <td className="py-2 px-4 border-b text-sm text-gray-800">{well.lat.toFixed(4)}</td>
                                        <td className="py-2 px-4 border-b text-sm text-gray-800">{well.lon.toFixed(4)}</td>
                                        <td className="py-2 px-4 border-b text-sm text-gray-800 flex gap-2">
                                            <button
                                                className="text-indigo-600 hover:text-indigo-900 font-medium"
                                                onClick={() => handleEditWellClick(well)}
                                            >
                                                Sửa
                                            </button>
                                            <button
                                                className="text-red-600 hover:text-red-900 font-medium"
                                                onClick={() => handleDeleteWell(well.id, well.name)}
                                            >
                                                Xóa
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                ) : (
                    <p className="text-slate-600">Chưa có giếng nào được thêm vào.</p>
                )}
            </CollapsibleSection>
        </div>
    );
};

// Consolidated Data Table Component (SSOT Bird-Eye View)
const ConsolidatedDataTable = ({ groundwaterData, waterQualityData, weatherForecast, waterUsage, wellLocations }) => {
    const [filterWellId, setFilterWellId] = useState('');
    const [filterStartDate, setFilterStartDate] = useState('');
    const [filterEndDate, setFilterEndDate] = useState('');

    const allData = useMemo(() => {
        const combined = {};

        // Use nullish coalescing to ensure data arrays are treated as empty arrays if undefined/null
        (groundwaterData ?? []).forEach(d => {
            const key = `${d.wellId}-${d.timestamp}`;
            combined[key] = { ...combined[key], wellId: d.wellId, timestamp: d.timestamp, gwl: d.gwl, ec: d.ec };
        });
        (waterQualityData ?? []).forEach(d => {
            const key = `${d.wellId}-${d.timestamp}`;
            combined[key] = { ...combined[key], wellId: d.wellId, timestamp: d.timestamp, ph: d.ph, do: d.do, turbidity: d.turbidity };
        });
        (weatherForecast ?? []).forEach(d => {
            const key = `${d.wellId}-${d.timestamp}`;
            combined[key] = { ...combined[key], wellId: d.wellId, timestamp: d.timestamp, precipitation: d.precipitation, temperature: d.temperature };
        });
        (waterUsage ?? []).forEach(d => {
            const key = `${d.wellId}-${d.timestamp}`;
            combined[key] = { ...combined[key], wellId: d.wellId, timestamp: d.timestamp, pumping: d.pumping, consumption: d.consumption };
        });

        const result = Object.values(combined).sort((a, b) => {
            const dateA = new Date(a.timestamp);
            const dateB = new Date(b.timestamp);
            if (dateA - dateB !== 0) return dateA - dateB;
            // Safely compare wellIds, treating undefined/null as empty strings for sorting
            const wellIdA = a.wellId || '';
            const wellIdB = b.wellId || '';
            return wellIdA.localeCompare(wellIdB);
        });
        return result;
    }, [groundwaterData, waterQualityData, weatherForecast, waterUsage]);

    const filteredAllData = useMemo(() => {
        return allData.filter(item => {
            const itemDate = new Date(item.timestamp);
            const start = filterStartDate ? new Date(filterStartDate) : null;
            const end = filterEndDate ? new Date(filterEndDate) : null;

            const matchesWellId = filterWellId ? (item.wellId || '').toLowerCase().includes(filterWellId.toLowerCase()) : true;
            const matchesStartDate = start ? itemDate >= start : true;
            const matchesEndDate = end ? itemDate >= end : true; // Changed from <= to >= for end date filter to be inclusive of the end of the day

            return matchesWellId && matchesStartDate && matchesEndDate;
        });
    }, [allData, filterWellId, filterStartDate, filterEndDate]);

    return (
        <div className="mt-8 p-6 bg-white rounded-xl shadow-lg border border-slate-200">
            <h3 className="text-2xl font-bold mb-4 text-slate-800">Dữ liệu Tổng hợp (SSOT Bird-Eye View)</h3>
            <p className="text-slate-600 mb-4">
                Xem tất cả dữ liệu từ các nguồn khác nhau được hợp nhất. Sử dụng các bộ lọc để tùy chỉnh khung nhìn của bạn.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div>
                    <label htmlFor="filter-well-id" className="block text-slate-700 text-sm font-bold mb-2">Lọc theo ID Giếng:</label>
                    <input
                        type="text"
                        id="filter-well-id"
                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                        value={filterWellId}
                        onChange={(e) => setFilterWellId(e.target.value)}
                        placeholder="Nhập ID giếng..."
                    />
                </div>
                <div>
                    <label htmlFor="filter-start-date" className="block text-slate-700 text-sm font-bold mb-2">Từ ngày:</label>
                    <input
                        type="date"
                        id="filter-start-date"
                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                        value={filterStartDate}
                        onChange={(e) => setFilterStartDate(e.target.value)}
                    />
                </div>
                <div>
                    <label htmlFor="filter-end-date" className="block text-slate-700 text-sm font-bold mb-2">Đến ngày:</label>
                    <input
                        type="date"
                        id="filter-end-date"
                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                        value={filterEndDate}
                        onChange={(e) => setFilterEndDate(e.target.value)}
                    />
                </div>
            </div>

            <CollapsibleSection title="Bảng Dữ liệu Tổng hợp">
                {filteredAllData.length > 0 ? (
                    <div className="overflow-x-auto">
                        <table className="min-w-full bg-white border border-gray-200 rounded-lg">
                            <thead>
                                <tr>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">ID Giếng</th>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">Thời gian</th>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">GWL (m bgs)</th>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">EC (µS/cm)</th>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">pH</th>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">DO (mg/L)</th>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">Độ đục (NTU)</th>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">Lượng mưa (mm)</th>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">Nhiệt độ (°C)</th>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">Lượng bơm (m³/ngày)</th>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">Tiêu thụ (m³/ngày)</th>
                                </tr>
                            </thead>
                            <tbody>
                                {filteredAllData.map((item, index) => (
                                    <tr key={`${item.wellId}-${item.timestamp}-${index}`}>
                                        <td className="py-2 px-4 border-b text-sm text-gray-800">{item.wellId}</td>
                                        <td className="py-2 px-4 border-b text-sm text-gray-800">{new Date(item.timestamp).toLocaleString('vi-VN')}</td>
                                        <td className="py-2 px-4 border-b text-sm text-gray-800">{item.gwl !== undefined ? item.gwl.toFixed(2) : 'N/A'}</td>
                                        <td className="py-2 px-4 border-b text-sm text-gray-800">{item.ec !== undefined ? item.ec.toFixed(0) : 'N/A'}</td>
                                        <td className="py-2 px-4 border-b text-sm text-gray-800">{item.ph !== undefined ? item.ph.toFixed(1) : 'N/A'}</td>
                                        <td className="py-2 px-4 border-b text-sm text-gray-800">{item.do !== undefined ? item.do.toFixed(1) : 'N/A'}</td>
                                        <td className="py-2 px-4 border-b text-sm text-gray-800">{item.turbidity !== undefined ? item.turbidity.toFixed(0) : 'N/A'}</td>
                                        <td className="py-2 px-4 border-b text-sm text-gray-800">{item.precipitation !== undefined ? item.precipitation.toFixed(1) : 'N/A'}</td>
                                        <td className="py-2 px-4 border-b text-sm text-gray-800">{item.temperature !== undefined ? item.temperature.toFixed(1) : 'N/A'}</td>
                                        <td className="py-2 px-4 border-b text-sm text-gray-800">{item.pumping !== undefined ? item.pumping.toFixed(0) : 'N/A'}</td>
                                        <td className="py-2 px-4 border-b text-sm text-gray-800">{item.consumption !== undefined ? item.consumption.toFixed(0) : 'N/A'}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                ) : (
                    <p className="text-slate-600">Không có dữ liệu tổng hợp nào để hiển thị. Hãy thử tạo dữ liệu khả tín!</p>
                )}
            </CollapsibleSection>
        </div>
    );
};

// DashboardTab
const DashboardTab = ({ filteredGroundwaterData, sevenDayGroundwaterPrediction, predictionErrors, selectedPredictionModel, handleRefreshDashboard, aiDashboardRecommendation }) => {

    const latestGwl = filteredGroundwaterData.length > 0 ? filteredGroundwaterData[filteredGroundwaterData.length - 1]?.gwl : undefined;
    const latestEc = filteredGroundwaterData.length > 0 ? filteredGroundwaterData[filteredGroundwaterData.length - 1]?.ec : undefined;
    // Ensure firstPredictedGwl is only taken if the prediction array is not empty
    const firstPredictedGwl = sevenDayGroundwaterPrediction.length > 0 ? sevenDayGroundwaterPrediction[0] : undefined;

    const combinedGwlData = useMemo(() => {
        const historicalPoints = filteredGroundwaterData.map(d => ({
            timestamp: new Date(d.timestamp).getTime(), // Use getTime for sorting
            actualGwl: d.gwl, // Corrected key to match PredictionRecharts
            predictedGwl: null // No predicted value for historical actuals
        }));

        const lastActualDate = historicalPoints.length > 0 ? new Date(historicalPoints[historicalPoints.length - 1].timestamp) : new Date();

        const predictionPoints = sevenDayGroundwaterPrediction.map((val, i) => ({
            timestamp: addDays(lastActualDate, i + 1).getTime(),
            actualGwl: null, // No actual value for predictions
            predictedGwl: val
        }));

        const combined = [...historicalPoints, ...predictionPoints].sort((a, b) => a.timestamp - b.timestamp);

        // Map back to date string for display in Recharts if needed, or keep as timestamp for XAxis
        return combined.map(d => ({
            timestamp: new Date(d.timestamp).toLocaleDateString('vi-VN'), // For display
            actualGwl: d.actualGwl,
            predictedGwl: d.predictedGwl
        }));
    }, [filteredGroundwaterData, sevenDayGroundwaterPrediction]);


    const ecData = useMemo(() => {
        return filteredGroundwaterData.map(d => ({
            timestamp: new Date(d.timestamp).toLocaleDateString('vi-VN'),
            ec: d.ec
        }));
    }, [filteredGroundwaterData]);

    // Determine the summary message for the collapsible section
    const summaryMessage = useMemo(() => {
        if (sevenDayGroundwaterPrediction.length > 0) {
            if (predictionErrors.length > 0 && predictionErrors.reduce((sum, err) => sum + Math.abs(err.error), 0) > 0.5) {
                return "Cảnh báo: Sai số dự đoán GWL của AI cao. Cần xem xét lại dữ liệu hoặc tinh chỉnh hàm AI.";
            } else {
                return "Tốt: Sai số dự đoán của AI đang ở mức chấp nhận được.";
            }
        } else {
            return "Thông báo: Chưa có dự đoán AI nào được tạo. Vui lòng tạo hàm dự đoán và kiểm tra.";
        }
    }, [sevenDayGroundwaterPrediction, predictionErrors]);

    // Determine the text color for the summary message
    const statusTextColorClass = useMemo(() => {
        if (sevenDayGroundwaterPrediction.length > 0) {
            if (predictionErrors.length > 0 && predictionErrors.reduce((sum, err) => sum + Math.abs(err.error), 0) > 0.5) {
                return "text-red-700";
            } else {
                return "text-green-700";
            }
        } else {
            return "text-gray-700";
        }
    }, [sevenDayGroundwaterPrediction, predictionErrors]);


    return (
        <div className="p-6 bg-white rounded-xl shadow-lg border border-slate-200">
            <h2 className="text-3xl font-bold mb-6 text-slate-800">Tổng quan Dashboard</h2>
            <div className="flex justify-end mb-4">
                <button
                    className="px-4 py-2 bg-gray-200 text-gray-700 font-bold rounded-lg shadow-md hover:bg-gray-300 transition-colors"
                    onClick={handleRefreshDashboard}
                >
                    🔄 Làm mới Dashboard
                </button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                    <h3 className="text-xl font-semibold text-blue-800 mb-2">Mực nước ngầm (GWL)</h3>
                    <p className="text-3xl font-bold text-blue-600">
                        {latestGwl !== undefined ? `${latestGwl.toFixed(2)} m bgs` : 'N/A'}
                    </p>
                    <p className="text-sm text-blue-700 mt-2">Giá trị gần nhất</p>
                </div>
                <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                    <h3 className="text-xl font-semibold text-green-800 mb-2">Độ dẫn điện (EC)</h3>
                    <p className="text-3xl font-bold text-green-600">
                        {latestEc !== undefined ? `${latestEc.toFixed(0)} µS/cm` : 'N/A'}
                    </p>
                    <p className="text-sm text-green-700 mt-2">Giá trị gần nhất</p>
                </div>
                <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
                    <h3 className="text-xl font-semibold text-yellow-800 mb-2">Dự báo GWL AI (Ngày 1)</h3>
                    <p className="text-3xl font-bold text-yellow-600">
                        {/* Check if firstPredictedGwl is a number beforetoFixed */}
                        {firstPredictedGwl !== undefined && !isNaN(firstPredictedGwl) ? `${firstPredictedGwl.toFixed(2)} m bgs` : 'N/A'}
                    </p>
                    <p className="text-sm text-yellow-700 mt-2">Dự báo ${PREDICTING_PERIOD} ngày tới (giá trị ngày đầu tiên)</p>
                    <p className="text-xs text-yellow-700 mt-1">Mô hình: {selectedPredictionModel === 'general' ? 'AI Tổng quát' : selectedPredictionModel.toUpperCase()}</p>
                </div>
                {/* Collapsible section for Warnings & Recommendations */}
                <div className="bg-red-50 p-4 rounded-lg border border-red-200 col-span-full">
                    <h3 className="text-xl font-semibold text-red-800 mb-2">Cảnh báo & Đề xuất</h3>
                    <CollapsibleSection
                        title={summaryMessage}
                        initialOpen={true} // Make it opened by default as requested
                        bgColor="transparent" // Remove default background
                        borderColor="transparent" // Remove default border
                        textColor={statusTextColorClass} // Set text color from summary
                        headerBgColor="transparent" // Make header background transparent
                        headerTextColor={statusTextColorClass} // Set header text color
                    >
                        <div className="text-sm text-gray-600 mt-2">
                            {/* Render as Markdown */}
                            {aiDashboardRecommendation ? (
                                <div dangerouslySetInnerHTML={{ __html: renderMarkdownToHtml(aiDashboardRecommendation) }} />
                            ) : (
                                <p>AI có thể cung cấp các đề xuất tối ưu hóa ở đây.</p>
                            )}
                        </div>
                    </CollapsibleSection>
                </div>
            </div>
            <div className="mt-8">
                <h3 className="text-2xl font-bold mb-4 text-slate-800">Biểu đồ tổng quan</h3>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="bg-slate-50 p-4 rounded-lg shadow-inner border border-slate-200 h-80">
                        {combinedGwlData.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart
                                    data={combinedGwlData}
                                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                                >
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="timestamp" />
                                    <YAxis label={{ value: 'Mực nước ngầm (m bgs)', angle: -90, position: 'insideLeft' }} />
                                    <Tooltip />
                                    <Legend />
                                    {/* Updated dataKey to actualGwl and predictedGwl */}
                                    <Line type="monotone" dataKey="actualGwl" stroke="#3b82f6" dot={{ r: 3 }} name="Mực nước ngầm thực tế" />
                                    <Line type="monotone" dataKey="predictedGwl" stroke="#f59e0b" strokeDasharray="5 5" dot={{ r: 3 }} name={`GWL dự đoán ${PREDICTING_PERIOD} ngày tới`} />
                                </LineChart>
                            </ResponsiveContainer>
                        ) : (
                            <p className="text-center text-gray-600 py-8">Không có dữ liệu mực nước ngầm để hiển thị biểu đồ.</p>
                        )}
                    </div>
                    <div className="bg-slate-50 p-4 rounded-lg shadow-inner border border-slate-200 h-80">
                        {ecData.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart
                                    data={ecData}
                                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                                >
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="timestamp" />
                                    <YAxis label={{ value: 'Độ dẫn điện (µS/cm)', angle: -90, position: 'insideLeft' }} />
                                    <Tooltip />
                                    <Legend />
                                    <Line type="monotone" dataKey="ec" stroke="#10b981" activeDot={{ r: 8 }} name="Độ dẫn điện (EC)" />
                                </LineChart>
                            </ResponsiveContainer>
                        ) : (
                            <p className="text-center text-gray-600 py-8">Không có dữ liệu độ dẫn điện để hiển thị biểu đồ.</p>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

const DataManagementTab = ({
  dataStorageMode, db, userId, showMessage, showConfirm, closeConfirmModal, wellLocations, setWellLocations,
  selectedWellId, setSelectedWellId, filteredGroundwaterData, filteredWaterQualityData,
  filteredWeatherForecast, handleSuggestPlausibleData, isGeneratingPlausibleData,
  handleGenerateDataSchemaExplanation, aiDataSchemaExplanation,
  groundwaterData, waterQualityData, weatherForecast, waterUsage, openImportModal,
  setGroundwaterData, setWaterQualityData, setWeatherForecast, setWaterUsage,
  followUpSchemaQuestion, setFollowUpSchemaQuestion, isGeneratingDataSchemaExplanation,
  setAllWellForecasts, filteredWaterUsage // Pass this down
}) => {
  return (
    <div className="p-6 bg-white rounded-xl shadow-lg border border-slate-200">
      <h2 className="text-3xl font-bold mb-6 text-slate-800">Quản lý Dữ liệu Sensor & Môi trường</h2>
      <p className="text-slate-600 mb-6">Nhập hoặc xem dữ liệu mô phỏng từ các sensor và yếu tố môi trường.</p>

        <div className="mb-6">
            <label htmlFor="well-select" className="block text-slate-700 text-sm font-bold mb-2">
                Chọn giếng để xem dữ liệu:
            </label>
            <select
                id="well-select"
                className="shadow border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                value={selectedWellId}
                onChange={(e) => setSelectedWellId(e.target.value)}
            >
                {wellLocations.length === 0 ? (
                    <option value="">Không có giếng nào</option>
                ) : (
                    wellLocations.map(well => (
                        <option key={well.id} value={well.id}>{well.name} ({well.id})</option>
                    ))
                )}
            </select>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                <h3 className="text-xl font-semibold text-blue-800 mb-3">Dữ liệu Nước ngầm (GWL & EC)</h3>
                <button
                    className="px-4 py-2 bg-blue-600 text-white font-bold rounded-lg shadow-md hover:bg-blue-700 transition-colors w-full mb-4"
                    onClick={() => openImportModal('groundwater')}
                >
                    Nhập Dữ liệu Nước ngầm (JSON)
                </button>
                <div className="bg-slate-50 p-4 rounded-lg shadow-inner border border-slate-200 h-80">
                    {filteredGroundwaterData.length > 0 ? (
                        <GroundwaterChart data={filteredGroundwaterData} />
                    ) : (
                        <p className="text-center text-gray-600 py-8">Không có dữ liệu mực nước ngầm để hiển thị biểu đồ.</p>
                    )}
                </div>
                <div className="bg-slate-50 p-4 rounded-lg shadow-inner border border-slate-200 h-80 mt-4">
                    {filteredGroundwaterData.length > 0 ? (
                        <EcChart data={filteredGroundwaterData} />
                    ) : (
                        <p className="text-center text-gray-600 py-8">Không có dữ liệu độ dẫn điện để hiển thị biểu đồ.</p>
                    )}
                </div>
                <CollapsibleSection 
                    title="Dữ liệu thô (Nước ngầm)"
                    initialOpen={false}>
                    <pre className="text-xs text-gray-800 overflow-auto max-h-40 bg-gray-100 p-2 rounded-md">
                        {JSON.stringify(filteredGroundwaterData, null, 2)}
                    </pre>
                </CollapsibleSection>
            </div>

            <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                <h3 className="text-xl font-semibold text-green-800 mb-3">Dữ liệu Chất lượng Nước</h3>
                <button
                    className="px-4 py-2 bg-green-600 text-white font-bold rounded-lg shadow-md hover:bg-green-700 transition-colors w-full mb-4"
                    onClick={() => openImportModal('waterQuality')}
                >
                    Nhập Dữ liệu Chất lượng Nước (JSON)
                </button>
                <div className="bg-slate-50 p-4 rounded-lg shadow-inner border border-slate-200 h-80">
                    {filteredWaterQualityData.length > 0 ? (
                        <WaterQualityChart data={filteredWaterQualityData} />
                    ) : (
                        <p className="text-center text-gray-600 py-8">Không có dữ liệu chất lượng nước để hiển thị biểu đồ.</p>
                    )}
                </div>
                <CollapsibleSection title="Dữ liệu thô (Chất lượng nước)">
                    <pre className="text-xs text-gray-800 overflow-auto max-h-40 bg-gray-100 p-2 rounded-md">
                        {JSON.stringify(filteredWaterQualityData, null, 2)}
                    </pre>
                </CollapsibleSection>
            </div>

            <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
                <h3 className="text-xl font-semibold text-yellow-800 mb-3">Dữ liệu Dự báo Thời tiết</h3>
                <button
                    className="px-4 py-2 bg-yellow-600 text-white font-bold rounded-lg shadow-md hover:bg-yellow-700 transition-colors w-full mb-4"
                    onClick={() => openImportModal('weather')}
                >
                    Nhập Dữ liệu Thời tiết (JSON)
                </button>
                <div className="bg-slate-50 p-4 rounded-lg shadow-inner border border-slate-200 h-80">
                    {filteredWeatherForecast.length > 0 ? (
                        <WeatherChart data={filteredWeatherForecast} />
                    ) : (
                        <p className="text-center text-gray-600 py-8">Không có dữ liệu dự báo thời tiết để hiển thị biểu đồ.</p>
                    )}
                </div>
                <CollapsibleSection 
                    title="Dữ liệu thô (Thời tiết)"
                    initialOpen={false}>
                    <pre className="text-xs text-gray-800 overflow-auto max-h-40 bg-gray-100 p-2 rounded-md">
                        {JSON.stringify(filteredWeatherForecast, null, 2)}
                    </pre>
                </CollapsibleSection>
            </div>

            <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                <h3 className="text-xl font-semibold text-purple-800 mb-3">Dữ liệu Sử dụng Nước</h3>
                <button
                    className="px-4 py-2 bg-purple-600 text-white font-bold rounded-lg shadow-md hover:bg-purple-700 transition-colors w-full mb-4"
                    onClick={() => openImportModal('usage')}
                >
                    Nhập Dữ liệu Sử dụng Nước (JSON)
                </button>
                <div className="bg-slate-50 p-4 rounded-lg shadow-inner border border-slate-200 h-80">
                    {filteredWaterUsage.length > 0 ? (
                        <UsageChart data={filteredWaterUsage} />
                    ) : (
                        <p className="text-center text-gray-600 py-8">Không có dữ liệu sử dụng nước để hiển thị biểu đồ.</p>
                    )}
                </div>
                <CollapsibleSection 
                    title="Dữ liệu thô (Sử dụng nước)"
                    initialOpen={false}>
                    <pre className="text-xs text-gray-800 overflow-auto max-h-40 bg-gray-100 p-2 rounded-md">
                        {JSON.stringify(filteredWaterUsage, null, 2)}
                    </pre>
                </CollapsibleSection>
            </div>
        </div>

        <div className="mt-6 text-center">
            <button
                className={`px-6 py-3 bg-indigo-600 text-white font-bold rounded-lg shadow-md hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${isGeneratingPlausibleData ? 'animate-pulse' : ''}`}
                onClick={handleSuggestPlausibleData}
                disabled={isGeneratingPlausibleData || !selectedWellId}
            >
                {isGeneratingPlausibleData ? (
                    <div className="flex items-center justify-center space-x-2">
                        <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <span>Đang tạo dữ liệu...</span>
                    </div>
                ) : (
                    'Đề xuất dữ liệu khả tín (bởi AI) cho giếng đã chọn'
                )}
            </button>
        </div>

        <WellManagement
            dataStorageMode={dataStorageMode}
            db={db}
            userId={userId}
            showMessage={showMessage}
            showConfirm={showConfirm}
            closeConfirmModal={closeConfirmModal}
            wellLocations={wellLocations}
            setWellLocations={setWellLocations}
            setSelectedWellId={setSelectedWellId}
            selectedWellId={selectedWellId}
            setGroundwaterData={setGroundwaterData}
            setWaterQualityData={setWaterQualityData}
            setWeatherForecast={setWeatherForecast}
            setWaterUsage={setWaterUsage}
            setAllWellForecasts={setAllWellForecasts}
        />

        <ConsolidatedDataTable
            groundwaterData={groundwaterData}
            waterQualityData={waterQualityData}
            weatherForecast={weatherForecast}
            waterUsage={waterUsage}
            wellLocations={wellLocations}
        />

        <div className="mt-8 p-6 bg-white rounded-xl shadow-lg border border-slate-200">
            <h3 className="text-2xl font-bold mb-4 text-slate-800">Lược đồ Dữ liệu & Liên kết Ngữ nghĩa (SSOT)</h3>
            <p className="text-slate-600 mb-4">
                Để hiểu rõ hơn về cấu trúc dữ liệu tổng thể và cách các loại dữ liệu khác nhau được liên kết, bạn có thể yêu cầu AI giải thích.
            </p>
            <div className="mb-4">
                <label htmlFor="follow-up-schema-question" className="block text-slate-700 text-sm font-bold mb-2">
                    Nhập câu hỏi của bạn về lược đồ dữ liệu hoặc để trống để yêu cầu giải thích tổng thể:
                </label>
                <textarea
                    id="follow-up-schema-question"
                    className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline h-20"
                    value={followUpSchemaQuestion}
                    onChange={(e) => setFollowUpSchemaQuestion(e.target.value)}
                    placeholder="VD: 'Giải thích thêm về trường 'timestamp' và định dạng của nó.', 'Mối quan hệ giữa GWL và lượng mưa là gì?'"
                ></textarea>
            </div>
            <button
                className={`px-6 py-3 bg-teal-600 text-white font-bold rounded-lg shadow-md hover:bg-teal-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${isGeneratingDataSchemaExplanation ? 'animate-pulse' : ''}`}
                onClick={handleGenerateDataSchemaExplanation} // Consolidated function
                disabled={isGeneratingDataSchemaExplanation}
            >
                {isGeneratingDataSchemaExplanation ? (
                    <div className="flex items-center justify-center space-x-2">
                        <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <span>Đang xử lý...</span>
                    </div>
                ) : (
                    followUpSchemaQuestion.trim() ? 'Hỏi AI về lược đồ dữ liệu' : 'Yêu cầu AI giải thích lược đồ dữ liệu'
                )}
            </button>
            {aiDataSchemaExplanation && (
                <CollapsibleSection title="Giải thích Lược đồ Dữ liệu bởi AI">
                    <div className="prose max-w-none text-slate-800">
                        <div dangerouslySetInnerHTML={{ __html: renderMarkdownToHtml(aiDataSchemaExplanation) }} />
                    </div>
                </CollapsibleSection>
            )}
            <div className="mt-8">
                <h3 className="text-2xl font-bold mb-4 text-slate-800">Thông tin Mô hình AI & Phản ứng</h3>
                <p className="text-slate-600 mb-4">
                    Các mô hình AI trong ứng dụng này (như mô hình dự đoán mực nước ngầm) học hỏi từ các tập dữ liệu được thu thập (nước ngầm, chất lượng nước, thời tiết, sử dụng nước) để đưa ra dự báo và đề xuất.
                    Chúng sử dụng các mẫu lịch sử và mối quan hệ giữa các yếu tố để đưa ra dự đoán.
                </p>
                <p className="text-slate-600 mb-4">
                    Để xem kiến trúc, tham số chi tiết và lịch sử học tập của mô hình, vui lòng chuyển đến tab "Dự báo & Tối ưu hóa AI" và "Học tập & Tri thức AI". Tại đó, bạn có thể tương tác trực tiếp với hàm AI và theo dõi hiệu suất của nó.
                </p>
            </div>
        </div>
    </div>
  );
};

const PredictionOptimizationTab = ({
    filteredGroundwaterData,
    selectedPerformanceMetric, setSelectedPerformanceMetric, // NEW
    selectedPredictionModel, setSelectedPredictionModel, arimaParams, setArimaParams,
    gpKernelType, setGpKernelType, promptMode, setPromptMode, userHint, setUserHint,
    isAISuggestingHint, setIsAISuggestingHint, aiPredictionFunctionBody, setAiPredictionFunctionBody,
    isGeneratingAiFunction, aiFunctionError, isAiFunctionChecked, handleGenerateAiFunction,
    handleCheckPrediction, handleRevertToLastValidAiFunction, // lastValidAiPredictionFunctionBody removed
    aiIterationCount,
    aiTheoryHistory, aiTheory, setAiTheory, aiNaturalLanguageExplanation, setAiNaturalLanguageExplanation,
    handleAnalyzeCurrentMismatches, isAnalyzingCurrentMismatches, showMessage, selectedWellId,
    filteredWaterQualityData, filteredWeatherForecast, filteredWaterUsage,
    allWellForecasts, setAllWellForecasts,
    aiModelSpecificData, setAiModelSpecificData,
    setAiFunctionError, setIsAiFunctionChecked,
    db, userId, appId, // Pass db, userId, appId for Firestore operations
    futureCiBounds, bootstrapStartStep, handleBootstrapStartStepChange
}) => {
    // Derive current well's predictions for display
    const sevenDayGroundwaterPrediction = useMemo(() => {
        return allWellForecasts[selectedWellId]?.predictions || [];
    }, [allWellForecasts, selectedWellId]);

    const sevenDayGroundwaterPredictionDates = useMemo(() => {
        return allWellForecasts[selectedWellId]?.dates || [];
    }, [allWellForecasts, selectedWellId]);

    const predictionErrors = useMemo(() => {
        return allWellForecasts[selectedWellId]?.errors || [];
    }, [allWellForecasts, selectedWellId]);

    const currentWellMetrics = useMemo(() => {
        return allWellForecasts[selectedWellId]?.metrics || {};
    }, [allWellForecasts, selectedWellId]);

    // Combine actual and predicted data for display on prediction chart
    const predictionChartData = useMemo(() => {
        const chartDataMap = new Map();

        // 1. Add ALL historical groundwater data (for actual GWL line)
        filteredGroundwaterData.forEach(d => {
            const originalTimestamp = d.timestamp; // Keep original ISO string or whatever it is
            chartDataMap.set(originalTimestamp, {
                timestamp: originalTimestamp,
                actualGwl: d.gwl, // Corrected key to match PredictionRecharts
                predictedGwl: null,
                predictionError: null
            });
        });

        // 2. Overlay historical predictions and errors from predictionErrors
        predictionErrors.forEach(err => {
            const originalTimestamp = err.timestamp; // Use original timestamp for error points
            const existingEntry = chartDataMap.get(originalTimestamp) || {};
            chartDataMap.set(originalTimestamp, {
                ...existingEntry,
                timestamp: originalTimestamp, // Ensure original timestamp is kept for sorting
                predictedGwl: err.predicted,
                predictionError: err.error
            });
        });

        // 3. Add future predictions
        const lastHistoricalDate = filteredGroundwaterData.length > 0
            ? new Date(filteredGroundwaterData[filteredGroundwaterData.length - 1].timestamp)
            : new Date();

        sevenDayGroundwaterPrediction.forEach((val, i) => {
            const futureDate = addDays(lastHistoricalDate, i + 1);
            const futureTimestampISO = formatISO(futureDate, { representation: 'complete' }); // Use ISO string for consistency
            const piBoundsForDay = futureCiBounds[i]; // Get PI bounds for this specific day
            
            chartDataMap.set(futureTimestampISO, {
                timestamp: futureTimestampISO,
                actualGwl: null,
                predictedGwl: val,
                predictionError: null,
                predictedGwlUpperPI: piBoundsForDay?.upper || null, // Change to PI
                predictedGwlLowerPI: piBoundsForDay?.lower || null  // Change to PI
            });
        });

        // Convert map values to array and sort by original timestamp
        const sortedData = Array.from(chartDataMap.values()).sort((a, b) => {
            return new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime(); // Use getTime for robust sorting
        });

        return sortedData;
    }, [filteredGroundwaterData, sevenDayGroundwaterPrediction, predictionErrors]);

    // NEW: Handle model change - load data specific to the new model
    const handleModelChange = useCallback((newModel) => {
        setSelectedPredictionModel(newModel);
        // Load AI function, theory, and explanation specific to the new model
        // We now rely on `aiPredictionFunctionBody`, `aiTheory`, `aiNaturalLanguageExplanation` being derived
        // directly from `aiModelSpecificData[selectedPredictionModel]` based on the updated `selectedPredictionModel`
        
        // When changing model, ensure the parameters for that model are reflected
        const newModelData = aiModelSpecificData[newModel];
        if (newModel === 'arima' && newModelData?.arimaParams) {
            setArimaParams(newModelData.arimaParams);
        } else if (newModel === 'gaussian_process' && newModelData?.gpKernelType) {
            setGpKernelType(newModelData.gpKernelType);
        }
        
        setAiFunctionError(null);
        setIsAiFunctionChecked(false); // Mark as unchecked to re-evaluate
        // No need to clear allWellForecasts here, it's specific to the well not model.
        // It will be updated when handleCheckPrediction is called.
    }, [aiModelSpecificData, setSelectedPredictionModel, setArimaParams, setGpKernelType, setAiFunctionError, setIsAiFunctionChecked]);


    const handleSelectBestModel = useCallback(async () => {
        if (aiTheoryHistory.length === 0) {
            showMessage('Thông báo', 'Không có lịch sử học thuyết AI để chọn mô hình tốt nhất.', 'info');
            return;
        }

        // Find the entry with the minimum error for the currently selected performance metric
        const bestModelEntry = aiTheoryHistory.reduce((best, current) => {
            const currentMetricValue = current.metrics ? current.metrics[selectedPerformanceMetric] : Infinity;
            const bestMetricValue = best.metrics ? best.metrics[selectedPerformanceMetric] : Infinity;

            if (currentMetricValue === Infinity || currentMetricValue === null || isNaN(currentMetricValue)) {
                return best;
            }
                
            if (bestMetricValue === Infinity || bestMetricValue === null || isNaN(bestMetricValue) || currentMetricValue < bestMetricValue) {
                return current;
            }
            return best; // Ensure to return 'best' if 'current' is not better
        }, { metrics: { [selectedPerformanceMetric]: Infinity } }); // Initialize with Infinity error for the selected metric

        if (bestModelEntry.metrics[selectedPerformanceMetric] === Infinity) {
            showMessage('Thông báo', `Không tìm thấy mô hình hợp lệ nào trong lịch sử với chỉ số ${performanceMetricsCalculators[selectedPerformanceMetric].name} để chọn làm tốt nhất.`, 'info');
            return;
        }

        // Apply the best model's function and parameters to the editor states
        // This implicitly updates the aiModelSpecificData for the relevant model type
        setAiPredictionFunctionBody(bestModelEntry.functionBody || DEFAULT_PREDICTION_FUNCTION_BODY);
        setSelectedPredictionModel(bestModelEntry.modelType || 'general'); // Ensure the UI radio button updates
        setAiTheory(bestModelEntry.theory);
        setAiNaturalLanguageExplanation(bestModelEntry.explanation || '');
        
        // Only set ARIMA/GP params if they were relevant for the best model
        if (bestModelEntry.modelType === 'arima' && bestModelEntry.arimaParams) {
            setArimaParams(bestModelEntry.arimaParams);
        } else if (bestModelEntry.modelType === 'gaussian_process' && bestModelEntry.gpKernelType) {
            setGpKernelType(bestModelEntry.gpKernelType);
        } else {
            // Reset to defaults if the best model is general or parameters are missing
            setArimaParams({ p: 1, d: 1, q: 1 });
            setGpKernelType('RBF');
        }
        
        setAiFunctionError(null);
        setIsAiFunctionChecked(false); // Mark as unchecked to re-evaluate

        // Load the future forecast associated with this best model for the currently selected well
        if (selectedWellId) {
            setAllWellForecasts(prev => ({
                ...prev,
                [selectedWellId]: {
                    ...prev[selectedWellId], // Preserve other data like errors if needed
                    predictions: bestModelEntry.sevenDayGroundwaterForecast || [],
                    dates: sevenDayGroundwaterPredictionDates, // Reuse existing dates for this well, or re-calculate if needed
                    metrics: bestModelEntry.metrics || {} // Update with the metrics from the best model entry
                }
            }));
        }

        showMessage('Thông báo', `Đã chọn mô hình tốt nhất từ lần lặp ${bestModelEntry.iteration} với ${performanceMetricsCalculators[selectedPerformanceMetric].name} là ${bestModelEntry.metrics[selectedPerformanceMetric] !== null ? bestModelEntry.metrics[selectedPerformanceMetric].toFixed(2) : 'N/A'}. Đang kiểm tra hiệu suất trên dữ liệu gần nhất...`, 'success');

        // Automatically run handleCheckPrediction to update historical errors based on the newly loaded best function
        await handleCheckPrediction();

        // The second message will be shown after handleCheckPrediction completes.
        // The first message is just a "loading" indicator for the check.
    }, [aiTheoryHistory, selectedPerformanceMetric, performanceMetricsCalculators, setAiPredictionFunctionBody, setSelectedPredictionModel, setAiTheory, setAiNaturalLanguageExplanation, setArimaParams, setGpKernelType, showMessage, handleCheckPrediction, selectedWellId, setAllWellForecasts, sevenDayGroundwaterPredictionDates, setAiFunctionError, setIsAiFunctionChecked]);

    const explanationContent = `
- **Bootstrapping (Khoảng Dự đoán - PI):** Phương pháp mô phỏng phi tham số để xây dựng khoảng dự đoán. Bằng cách lấy mẫu ngẫu nhiên lặp lại từ các phần dư lịch sử của mô hình (được coi là các "đổi mới" - innovations), chúng ta tạo ra một tập hợp lớn các đường dẫn dự đoán tương lai có thể có. Điều **quan trọng** là, mỗi giá trị trong đường dẫn mô phỏng được tạo ra bằng cách cộng dồn các lỗi ngẫu nhiên được lấy mẫu vào dự đoán điểm ban đầu. Khi dự đoán càng xa về tương lai, càng nhiều lỗi ngẫu nhiên được cộng dồn, làm cho phương sai của các giá trị mô phỏng tăng lên và dẫn đến khoảng PI mở rộng một cách tự nhiên và đúng đắn về mặt xác suất. Phương pháp này cung cấp độ tin cậy cao hơn, đặc biệt khi phân phối sai số không chuẩn, nhưng đòi hỏi nhiều tính toán.

- **Factor-based (Khoảng Dự đoán - PI):** Phương pháp đơn giản hóa, tính toán khoảng dự đoán dựa trên độ lệch chuẩn của phần dư lịch sử, nhân với một hệ số (ví dụ: 1.96 cho PI 95% nếu sai số phân phối chuẩn). Khoảng này cũng được mở rộng theo thời gian dự báo (ví dụ, nhân thêm $\\sqrt{\\text{chân trời}}$) để phản ánh sự không chắc chắn tăng lên khi dự báo xa hơn vào tương lai.

**Kiểm soát Khoảng Dự đoán (PI) bằng Thanh trượt:**
Thanh trượt cho phép bạn điều chỉnh cách kết hợp giữa hai phương pháp trên để tính toán Prediction Interval (PI) cho ${PREDICTING_PERIOD} ngày dự đoán tới:
- **Bước 0 (Hoàn toàn Factor-based PI):** Khi thanh trượt ở 0, toàn bộ ${PREDICTING_PERIOD} ngày dự đoán sẽ sử dụng phương pháp Factor-based PI. Đây là cách tiếp cận nhanh chóng nhưng có thể ít chính xác hơn nếu sai số không tuân theo giả định chuẩn.
- **Bước ${PREDICTING_PERIOD} (Hoàn toàn Bootstrapping PI):** Khi thanh trượt ở ${PREDICTING_PERIOD} (ví dụ: 7), toàn bộ ${PREDICTING_PERIOD} ngày dự đoán sẽ sử dụng phương pháp Bootstrapping PI với cơ chế tích lũy lỗi. Đây là phương pháp mạnh mẽ hơn về mặt thống kê và không yêu cầu giả định phân phối sai số chuẩn, nhưng tốn kém tính toán hơn.
- **Các bước trung gian (Hybrid):** Khi thanh trượt ở một giá trị trung gian (ví dụ: 3), các ngày dự đoán từ ngày 1 đến ngày 3 sẽ sử dụng Bootstrapping PI (tích lũy lỗi), và các ngày còn lại (ngày 4 đến ngày 7) sẽ chuyển sang sử dụng Factor-based PI. Điều này cho phép tận dụng ưu điểm của Bootstrapping cho các dự báo ngắn hạn và sự đơn giản của Factor-based cho các dự báo dài hạn.
    `;
    return (
        <React.Fragment>
            <div className="p-6 bg-white rounded-xl shadow-lg border border-slate-200">
                <h2 className="text-3xl font-bold mb-6 text-slate-800">Dự báo & Tối ưu hóa AI</h2>
                <p className="text-slate-600 mb-6">
                    AI sẽ học cách dự báo mực nước ngầm và đề xuất các hành động quản lý bền vững dựa trên dữ liệu sensor.
                </p>

                <div className="mb-6">
                    <div className="mb-4 flex flex-wrap gap-4 items-center">
                        <button
                            className="px-6 py-2 bg-blue-500 text-white font-bold rounded-lg shadow-md hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            onClick={handleSelectBestModel}
                            // Button is enabled if there's at least one valid model (error not Infinity/null)
                            disabled={aiTheoryHistory.every(entry => !entry.metrics || isNaN(entry.metrics[selectedPerformanceMetric]) || entry.metrics[selectedPerformanceMetric] === Infinity) || aiTheoryHistory.length === 0}
                        >
                            Chọn Mô hình Tốt nhất
                        </button>
                        <div className="flex items-center space-x-2">
                            <label htmlFor="performance-metric-select" className="block text-slate-700 text-sm font-bold">
                                Chọn chỉ số tối ưu hóa:
                            </label>
                            <select
                                id="performance-metric-select"
                                className="mt-1 block w-auto rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50 text-gray-700"
                                value={selectedPerformanceMetric}
                                onChange={(e) => setSelectedPerformanceMetric(e.target.value)}
                            >
                                {Object.keys(performanceMetricsCalculators).map(key => (
                                    <option key={key} value={key}>
                                        {performanceMetricsCalculators[key].name}
                                    </option>
                                ))}
                            </select>
                        </div>
                    </div>

                    <h3 className="text-lg font-bold mb-2 text-slate-700">Chọn Mô hình Dự đoán:</h3>
                    <div className="flex flex-wrap gap-4">
                        <label className="inline-flex items-center">
                            <input
                                type="radio"
                                className="form-radio text-green-600"
                                name="predictionModel"
                                value="general"
                                checked={selectedPredictionModel === 'general'}
                                onChange={() => handleModelChange('general')} // UPDATED
                            />
                            <span className="ml-2 text-slate-700">Tổng quát (AI tự do)</span>
                        </label>
                        <label className="inline-flex items-center">
                            <input
                                type="radio"
                                className="form-radio text-green-600"
                                name="predictionModel"
                                value="arima"
                                checked={selectedPredictionModel === 'arima'}
                                onChange={() => handleModelChange('arima')} // UPDATED
                            />
                            <span className="ml-2 text-slate-700">ARIMA (AI hướng dẫn)</span>
                        </label>
                        <label className="inline-flex items-center">
                            <input
                                type="radio"
                                className="form-radio text-green-600"
                                name="predictionModel"
                                value="gaussian_process"
                                checked={selectedPredictionModel === 'gaussian_process'}
                                onChange={() => handleModelChange('gaussian_process')} // UPDATED
                            />
                            <span className="ml-2 text-slate-700">Gaussian Process (AI hướng dẫn)</span>
                        </label>
                    </div>

                    {selectedPredictionModel === 'arima' && (
                        <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
                            <h4 className="text-md font-semibold text-gray-700 mb-2">Tham số ARIMA (p, d, q):</h4>
                            <div className="flex gap-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700">p:</label>
                                    <input
                                        type="number"
                                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
                                        value={arimaParams.p}
                                        onChange={(e) => setArimaParams(prev => ({ ...prev, p: parseInt(e.target.value) || 0 }))}
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700">d:</label>
                                    <input
                                        type="number"
                                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
                                        value={arimaParams.d}
                                        onChange={(e) => setArimaParams(prev => ({ ...prev, d: parseInt(e.target.value) || 0 }))}
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700">q:</label>
                                    <input
                                        type="number"
                                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
                                        value={arimaParams.q}
                                        onChange={(e) => setArimaParams(prev => ({ ...prev, q: parseInt(e.target.value) || 0 }))}
                                    />
                                </div>
                            </div>
                            <p className="text-xs text-gray-500 mt-2">
                                (p: bậc tự hồi quy, d: bậc sai phân tích hợp, q: bậc trung bình trượt)
                            </p>
                        </div>
                    )}

                    {selectedPredictionModel === 'gaussian_process' && (
                        <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
                            <h4 className="text-md font-semibold text-gray-700 mb-2">Loại Kernel Gaussian Process:</h4>
                            <select
                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
                                value={gpKernelType}
                                onChange={(e) => setGpKernelType(e.target.value)}
                            >
                                <option value="RBF">RBF Kernel (Radial Basis Function)</option>
                                <option value="Linear">Linear Kernel</option>
                                <option value="Polynomial">Polynomial Kernel</option>
                            </select>
                        </div>
                    )}

                    <h3 className="text-lg font-bold mt-6 mb-2 text-slate-700">Chế độ Prompt AI:</h3>
                    <div className="flex flex-wrap gap-4">
                        <label className="inline-flex items-center">
                            <input
                                type="radio"
                                className="form-radio text-purple-600"
                                name="promptMode"
                                value="high-end"
                                checked={promptMode === 'high-end'}
                                onChange={() => setPromptMode('high-end')}
                            />
                            <span className="ml-2 text-slate-700">High-End (Phân tích sâu, học hỏi mạnh mẽ)</span>
                        </label>
                        <label className="inline-flex items-center">
                            <input
                                type="radio"
                                className="form-radio text-purple-600"
                                name="promptMode"
                                value="mid-end"
                                checked={promptMode === 'mid-end'}
                                onChange={() => setPromptMode('mid-end')}
                            />
                            <span className="ml-2 text-slate-700">Mid-End (Phân tích cân bằng, cải thiện)</span>
                        </label>
                        <label className="inline-flex items-center">
                            <input
                                type="radio"
                                className="form-radio text-purple-600"
                                name="promptMode"
                                value="low-end"
                                checked={promptMode === 'low-end'}
                                onChange={() => setPromptMode('low-end')}
                            />
                            <span className="ml-2 text-slate-700">Low-End (Đơn giản, tập trung sửa lỗi)</span>
                        </label>
                    </div>
                </div>

                <div className="mb-6">
                    <label htmlFor="user-hint" className="block text-slate-700 text-sm font-bold mb-2">
                        Gợi ý của Người dùng (cho AI):
                    </label>
                    <textarea
                        id="user-hint"
                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline h-24"
                        value={userHint}
                        onChange={(e) => setUserHint(e.target.value)}
                        placeholder="VD: 'AI nên xem xét mối mối quan hệ giữa lượng mưa và mực nước ngầm.', 'Tập trung vào các xu hướng dài hạn thay vì biến động ngắn hạn.'"
                    ></textarea>
                    {/* AI Hint Suggestion Toggle */}
                    <div className="flex items-center mt-2">
                        <label className="flex items-center cursor-pointer">
                            <div className="relative">
                                <input
                                    type="checkbox"
                                    className="sr-only peer"
                                    checked={isAISuggestingHint}
                                    onChange={(e) => setIsAISuggestingHint(e.target.checked)}
                                />
                                <div className="block bg-gray-300 w-14 h-8 rounded-full transition-all duration-300 peer-checked:bg-blue-600"></div>
                                <div className="dot absolute left-1 top-1 bg-white w-6 h-6 rounded-full transition-all duration-300 peer-checked:translate-x-full"></div>
                            </div>
                            <span className="ml-3 text-gray-700 font-medium">AI phân tích và đưa ra gợi ý người dùng (dựa trên lỗi hiện tại)</span>
                        </label>
                    </div>
                </div>

                <div className="flex gap-4 mb-6">
                    <button
                        className={`px-6 py-3 bg-indigo-600 text-white font-bold rounded-lg shadow-md hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${isGeneratingAiFunction ? 'animate-pulse' : ''}`}
                        onClick={handleGenerateAiFunction}
                        disabled={isGeneratingAiFunction || !selectedWellId}
                    >
                        {isGeneratingAiFunction ? (
                            <div className="flex items-center justify-center space-x-2">
                                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                <span>AI đang tạo hàm...</span>
                            </div>
                        ) : (
                            `AI Tạo/Cải thiện Hàm Dự đoán (Lần lặp ${aiIterationCount + 1})`
                        )}
                    </button>
                    <button
                        className="px-6 py-3 bg-green-600 text-white font-bold rounded-lg shadow-md hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        onClick={handleCheckPrediction}
                        disabled={isGeneratingAiFunction || !selectedWellId}
                    >
                        Kiểm tra Dự đoán
                    </button>
                    <button
                        className="px-6 py-3 bg-gray-500 text-white font-bold rounded-lg shadow-md hover:bg-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        onClick={handleRevertToLastValidAiFunction}
                        disabled={!aiModelSpecificData[selectedPredictionModel]?.lastValidState || isGeneratingAiFunction}
                    >
                        Hoàn tác về Hàm hợp lệ cuối cùng
                    </button>
                </div>

                {aiFunctionError && (
                    <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4" role="alert">
                        <strong className="font-bold">Lỗi Hàm AI:</strong>
                        <span className="block sm:inline"> {aiFunctionError}</span>
                    </div>
                )}

                <h3 className="text-2xl font-bold mb-4 text-slate-800">Hàm Dự đoán AI (JavaScript)</h3>
                <p className="text-slate-600 mb-4">
                    AI sẽ tự động tạo và tối ưu hóa hàm JavaScript này để dự đoán mực nước ngầm.
                </p>
                <div className="bg-gray-800 rounded-lg p-4 font-mono text-sm text-white overflow-x-auto mb-6">
                    <pre><code>{renderAiFunctionSignature()}<br/>
                        <textarea
                            className="w-full h-64 bg-gray-800 text-white font-mono text-sm resize-y"
                            value={aiPredictionFunctionBody}
                            onChange={(e) => setAiPredictionFunctionBody(e.target.value)}
                            spellCheck="false"
                        ></textarea>
                    <br/>{'}'}</code></pre>
                </div>

                <h3 className="text-2xl font-bold mb-4 text-slate-800">Giải thích AI (Ngôn ngữ tự nhiên)</h3>
                <MarkdownRendererComponent
                    content={aiNaturalLanguageExplanation}
                    placeholderText="AI sẽ cung cấp giải thích về hàm và học thuyết của nó ở đây."
                />

                <h3 className="text-2xl font-bold mb-4 text-slate-800">Học thuyết AI</h3>
                <MarkdownRendererComponent
                    content={aiTheory}
                    placeholderText="Học thuyết cốt lõi của AI về hành vi mực nước ngầm sẽ xuất hiện ở đây."
                />

                <h3 className="text-2xl font-bold mb-4 text-slate-800">Sai số Dự đoán</h3>
                <p className="text-slate-600 mb-4">
                    Theo dõi sai số giữa giá trị thực tế và giá trị dự đoán của AI.
                </p>
                <div className="mt-4 p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl shadow-md border border-blue-200">
                    <label htmlFor="bootstrap-step-slider" className="block text-xl font-bold text-blue-800 mb-3">
                        Kiểm soát Khoảng Dự đoán (PI)
                    </label>
                    <div className="flex items-center space-x-4 mb-3">
                        <span className="text-2xl font-semibold text-indigo-700 w-24">
                            Bước {bootstrapStartStep}
                        </span>
                        <input
                            type="range"
                            id="bootstrap-step-slider"
                            min="0"
                            max={`${PREDICTING_PERIOD}`} // Ví dụ: max là 7 cho 7 ngày dự đoán
                            value={bootstrapStartStep}
                            onChange={handleBootstrapStartStepChange}
                            className="w-full h-3 bg-blue-300 rounded-full appearance-none cursor-pointer
                                       [&::-webkit-slider-thumb]:w-6 [&::-webkit-slider-thumb]:h-6
                                       [&::-webkit-slider-thumb]:bg-blue-600 [&::-webkit-slider-thumb]:rounded-full
                                       [&::-webkit-slider-thumb]:shadow-lg [&::-webkit-slider-thumb]:transition-transform
                                       [&::-webkit-slider-thumb]:duration-200 [&::-webkit-slider-thumb]:ease-in-out
                                       hover:[&::-webkit-slider-thumb]:scale-110
                                       [&::-moz-range-thumb]:w-6 [&::-moz-moz-range-thumb]:h-6
                                       [&::-moz-range-thumb]:bg-blue-600 [&::-moz-range-thumb]:rounded-full
                                       [&::-moz-range-thumb]:shadow-lg [&::-moz-range-thumb]:transition-transform
                                       [&::-moz-range-thumb]:duration-200 [&::-moz-range-thumb]:ease-in-out
                                       hover:[&::-moz-range-thumb]:scale-110"
                        />
                    </div>
                    <div className="flex justify-between text-sm text-blue-700 mt-2 font-medium">
                        <span>0 (Hoàn toàn Factor PI)</span>
                        <span>{PREDICTING_PERIOD} (Hoàn toàn Bootstrapping)</span>
                    </div>
                    <CollapsibleSection title="Giải thích" initialOpen={false}>
                        <MarkdownRendererComponent
                            content={explanationContent}
                            className="text-blue-700 text-sm" // Thêm className để giữ nguyên style
                        />
                    </CollapsibleSection>
                </div>
                <div className="bg-slate-50 p-4 rounded-lg shadow-inner border border-slate-200 h-80 mb-6">
                    {predictionChartData.length > 0 ? (
                        <PredictionRecharts data={predictionChartData} />
                    ) : (
                        <p className="text-center text-gray-600 py-8">Không có dữ liệu dự đoán để hiển thị biểu đồ.</p>
                    )}
                </div>

                {currentWellMetrics && Object.keys(currentWellMetrics).length > 0 && (
                    <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200 text-blue-800 text-sm">
                        <h4 className="font-bold mb-2">Chỉ số Hiệu suất Hiện tại cho Giếng {selectedWellId}:</h4>
                        <ul>
                            {Object.entries(currentWellMetrics).map(([key, value]) => (
                                <li key={key}>
                                    <strong>{performanceMetricsCalculators[key]?.name || key}:</strong> {value !== Infinity && value !== null ? `${value.toFixed(4)} ${performanceMetricsCalculators[key]?.unit || ''}` : 'N/A'}
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
                
                <CollapsibleSection title="Lịch sử Sai số Dự đoán">
                    {predictionErrors.length > 0 ? (
                        <div className="overflow-x-auto">
                            <table className="min-w-full bg-white border border-gray-200 rounded-lg">
                                <thead>
                                    <tr>
                                        <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">Thời gian</th>
                                        <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">Thực tế (m bgs)</th>
                                        <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">Dự đoán (m bgs)</th>
                                        <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">Sai số (m)</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {predictionErrors.map((error, index) => (
                                        <tr key={index}>
                                            <td className="py-2 px-4 border-b text-sm text-gray-800">{new Date(error.timestamp).toLocaleString('vi-VN')}</td>
                                            <td className="py-2 px-4 border-b text-sm text-gray-800">{error.actual.toFixed(2)}</td>
                                            <td className="py-2 px-4 border-b text-sm text-gray-800">{error.predicted.toFixed(2)}</td>
                                            <td className={`py-2 px-4 border-b text-sm font-semibold ${Math.abs(error.error) > 0.5 ? 'text-red-600' : 'text-green-600'}`}>
                                                {error.error.toFixed(2)}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    ) : (
                        <p className="text-slate-600">Chưa có sai số dự đoán nào để hiển thị. Hãy tạo và kiểm tra hàm AI!</p>
                    )}
                </CollapsibleSection>
            </div>
        </React.Fragment>
    );
};

const SustainabilityComplianceTab = ({ filteredGroundwaterData, filteredWaterQualityData, filteredWaterUsage, showMessage, selectedWellId,
    onGenerateSustainabilityInsights, sustainabilityInsights, isGeneratingSustainabilityInsights,
    sevenDayGroundwaterPrediction, // This will now be derived from allWellForecasts
    sevenDayGroundwaterPredictionDates // This will now be derived from allWellForecasts
}) => {
    // Placeholder for sustainability thresholds
    const MIN_GWL_THRESHOLD = 10; // meters below ground surface
    const MAX_EC_THRESHOLD = 1000; // µS/cm
    const MIN_PH_THRESHOLD = 6.5;
    const MAX_PH_THRESHOLD = 8.5;

    // Calculate average GWL for the selected well
    const averageGwl = useMemo(() => {
        if (filteredGroundwaterData.length === 0) return 'N/A';
        const totalGwl = filteredGroundwaterData.reduce((sum, d) => sum + d.gwl, 0);
        return (totalGwl / filteredGroundwaterData.length).toFixed(2);
    }, [filteredGroundwaterData]);

    // --- UPDATED: Check compliance to return null if no data ---
    const isGwlCompliant = useMemo(() => {
        if (filteredGroundwaterData.length === 0) return null; // Return null if no data
        return filteredGroundwaterData.every(d => d.gwl >= MIN_GWL_THRESHOLD);
    }, [filteredGroundwaterData]);

    const isEcCompliant = useMemo(() => {
        if (filteredGroundwaterData.length === 0) return null; // Return null if no data
        return filteredGroundwaterData.every(d => d.ec <= MAX_EC_THRESHOLD);
    }, [filteredGroundwaterData]);

    const isPhCompliant = useMemo(() => {
        if (filteredWaterQualityData.length === 0) return null; // Return null if no data
        return filteredWaterQualityData.every(d => d.ph >= MIN_PH_THRESHOLD && d.ph <= MAX_PH_THRESHOLD);
    }, [filteredWaterQualityData]);

    // Helper to render compliance status text
    const renderComplianceStatus = (isCompliant) => {
        if (isCompliant === null) {
            return 'N/A (Không đủ dữ liệu)';
        } else if (isCompliant) {
            return 'ĐẠT';
        } else {
            return 'KHÔNG ĐẠT';
        }
    };

    const renderComplianceTextColor = (isCompliant) => {
        if (isCompliant === null) {
            return 'text-gray-600';
        } else if (isCompliant) {
            return 'text-green-600';
        } else {
            return 'text-red-600';
        }
    };


    const handleGenerateComplianceReport = useCallback(() => {
        if (!selectedWellId) {
            showMessage('Thông báo', 'Vui lòng chọn một giếng để tạo báo cáo tuân thủ.', 'info');
            return;
        }
        if (filteredGroundwaterData.length === 0 && filteredWaterQualityData.length === 0 && filteredWaterUsage.length === 0) {
            showMessage('Thông báo', 'Không có đủ dữ liệu để tạo báo cáo tuân thủ.', 'info');
            return;
        }


        let reportContent = `# Báo cáo Tuân thủ Bền vững cho Giếng: ${selectedWellId}\n\n`;
        reportContent += `Ngày báo cáo: ${new Date().toLocaleDateString('vi-VN')}\n\n`;
        reportContent += `## 1. Mực nước ngầm (GWL)\n`;
        reportContent += `- Mực nước ngầm trung bình: ${averageGwl} m bgs\n`;
        reportContent += `- Ngưỡng tối thiểu được khuyến nghị: ${MIN_GWL_THRESHOLD} m bgs\n`;
        reportContent += `- Trạng thái tuân thủ: **${renderComplianceStatus(isGwlCompliant)}**\n\n`;

        reportContent += `## 2. Độ dẫn điện (EC)\n`;
        reportContent += `- Ngưỡng tối đa được khuyến nghị: ${MAX_EC_THRESHOLD} µS/cm\n`;
        reportContent += `- Trạng thái tuân thủ: **${renderComplianceStatus(isEcCompliant)}**\n\n`;

        reportContent += `## 3. Độ pH Chất lượng Nước\n`;
        reportContent += `- Ngưỡng pH được khuyến nghị: ${MIN_PH_THRESHOLD} - ${MAX_PH_THRESHOLD}\n`;
        reportContent += `- Trạng thái tuân thủ: **${renderComplianceStatus(isPhCompliant)}**\n\n`;

        reportContent += `## 4. Đề xuất & Nhận xét\n`;

        let hasComplianceIssues = false;
        if (isGwlCompliant === false) { // Explicitly check for false, not null
            reportContent += `- **Cảnh báo GWL:** Mực nước ngầm đang thấp hơn ngưỡng an toàn. Cần xem xét giảm lượng bơm hoặc tìm kiếm các biện pháp nạp nước.\n`;
            hasComplianceIssues = true;
        }
        if (isEcCompliant === false) { // Explicitly check for false, not null
            reportContent += `- **Cảnh báo EC:** Độ dẫn điện vượt quá ngưỡng an toàn, cho thấy khả năng nhiễm mặn hoặc ô nhiễm khác. Cần điều tra thêm.\n`;
            hasComplianceIssues = true;
        }
        if (isPhCompliant === false) { // Explicitly check for false, not null
            reportContent += `- **Cảnh báo pH:** Độ pH nằm ngoài phạm vi an toàn. Cần phân tích nguyên nhân và có biện pháp xử lý.\n`;
            hasComplianceIssues = true;
        }

        if (!hasComplianceIssues && (isGwlCompliant !== null || isEcCompliant !== null || isPhCompliant !== null)) {
            // Only show "Đạt" message if there was *some* data and no issues.
            // If all are null, it means no data to assess at all.
            reportContent += `- Giếng đang hoạt động trong giới hạn bền vững và tuân thủ (dựa trên dữ liệu hiện có). Tiếp tục theo dõi.\n`;
        } else if (isGwlCompliant === null && isEcCompliant === null && isPhCompliant === null) {
            reportContent += `- Không có đủ dữ liệu để đưa ra đánh giá toàn diện về tuân thủ. Vui lòng nhập thêm dữ liệu.\n`;
        }

        // Simulate downloading the report
        const blob = new Blob([reportContent], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Bao_cao_tuan_thu_gieng_${selectedWellId}_${formatISO(new Date(), { representation: 'date' })}.md`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        showMessage('Thành công', 'Báo cáo tuân thủ đã được tạo và tải xuống dưới dạng tệp Markdown.', 'success');
    }, [selectedWellId, filteredGroundwaterData, filteredWaterQualityData, filteredWaterUsage, averageGwl, isGwlCompliant, isEcCompliant, isPhCompliant, showMessage, renderComplianceStatus]);


    return (
        <div className="p-6 bg-white rounded-xl shadow-lg border border-slate-200">
            <h2 className="text-3xl font-bold mb-6 text-slate-800">Bền vững & Tuân thủ</h2>
            <p className="text-slate-600 mb-6">
                Đánh giá hiệu suất giếng dựa trên các chỉ số bền vững và tạo báo cáo tuân thủ.
            </p>

            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200 mb-6">
                <h3 className="text-xl font-semibold text-blue-800 mb-3">Chỉ số Bền vững Chính</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <p className="text-slate-700">
                            <span className="font-bold">Mực nước ngầm trung bình (GWL):</span> {averageGwl} m bgs
                        </p>
                        <p className="text-sm text-slate-600">
                            Ngưỡng tối thiểu: {MIN_GWL_THRESHOLD} m bgs
                        </p>
                        <p className={`font-bold ${renderComplianceTextColor(isGwlCompliant)}`}>
                            Trạng thái GWL: {renderComplianceStatus(isGwlCompliant)}
                        </p>
                    </div>
                    <div>
                        <p className="text-slate-700">
                            <span className="font-bold">Độ dẫn điện (EC):</span> {filteredGroundwaterData.length > 0 ? filteredGroundwaterData[filteredGroundwaterData.length - 1]?.ec.toFixed(0) : 'N/A'} µS/cm (Giá trị gần nhất)
                        </p>
                        <p className="text-sm text-slate-600">
                            Ngưỡng tối đa: {MAX_EC_THRESHOLD} µS/cm
                        </p>
                        <p className={`font-bold ${renderComplianceTextColor(isEcCompliant)}`}>
                            Trạng thái EC: {renderComplianceStatus(isEcCompliant)}
                        </p>
                    </div>
                    <div>
                        <p className="text-slate-700">
                            <span className="font-bold">Độ pH:</span> {filteredWaterQualityData.length > 0 ? filteredWaterQualityData[filteredWaterQualityData.length - 1]?.ph.toFixed(1) : 'N/A'} (Giá trị gần nhất)
                        </p>
                        <p className="text-sm text-slate-600">
                            Ngưỡng pH: {MIN_PH_THRESHOLD} - ${MAX_PH_THRESHOLD}
                        </p>
                        <p className={`font-bold ${renderComplianceTextColor(isPhCompliant)}`}>
                            Trạng thái pH: {renderComplianceStatus(isPhCompliant)}
                        </p>
                    </div>
                </div>
            </div>

            <div className="mt-6 flex gap-4">
                <button
                    className="px-6 py-3 bg-teal-600 text-white font-bold rounded-lg shadow-md hover:bg-teal-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    onClick={handleGenerateComplianceReport}
                    disabled={!selectedWellId || (filteredGroundwaterData.length === 0 && filteredWaterQualityData.length === 0 && filteredWaterUsage.length === 0)}
                >
                    Tạo Báo cáo Tuân thủ (Markdown)
                </button>
                <button
                    className={`px-6 py-3 bg-indigo-600 text-white font-bold rounded-lg shadow-md hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${isGeneratingSustainabilityInsights ? 'animate-pulse' : ''}`}
                    onClick={() => onGenerateSustainabilityInsights(
                        filteredGroundwaterData,
                        filteredWaterQualityData,
                        filteredWaterUsage,
                        isGwlCompliant, // Pass the explicit compliance state (true/false/null)
                        isEcCompliant,  // Pass the explicit compliance state (true/false/null)
                        isPhCompliant,  // Pass the explicit compliance state (true/false/null)
                        sevenDayGroundwaterPrediction, // Pass sevenDayGroundwaterPrediction
                        sevenDayGroundwaterPredictionDates // Pass sevenDayGroundwaterPredictionDates
                    )}
                    disabled={isGeneratingSustainabilityInsights || !selectedWellId || (filteredGroundwaterData.length === 0 && filteredWaterQualityData.length === 0 && filteredWaterUsage.length === 0)}
                >
                    {isGeneratingSustainabilityInsights ? (
                        <div className="flex items-center justify-center space-x-2">
                            <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            <span>Đang tạo thông tin...</span>
                        </div>
                    ) : (
                        'Tạo Chi tiết & Đề xuất Tối ưu hóa (bởi AI)'
                    )}
                </button>
            </div>

            <CollapsibleSection title="Chi tiết Tuân thủ">
                <MarkdownRendererComponent
                    content={sustainabilityInsights?.details}
                    placeholderText="AI sẽ cung cấp thông tin chi tiết về tình hình tuân thủ ở đây."
                />
            </CollapsibleSection>

            <CollapsibleSection title="Đề xuất Tối ưu hóa">
                <MarkdownRendererComponent
                    content={sustainabilityInsights?.recommendations}
                    placeholderText="AI sẽ đưa ra các đề xuất tối ưu hóa ở đây."
                />
            </CollapsibleSection>
        </div>
    );
};

const AILearningInsightsTab = ({ aiTheoryHistory, aiTheory, aiNaturalLanguageExplanation, predictionErrors, selectedPerformanceMetric }) => {
    return (
        <div className="p-6 bg-white rounded-xl shadow-lg border border-slate-200">
            <h2 className="text-3xl font-bold mb-6 text-slate-800">Học tập & Tri thức AI</h2>
            <p className="text-slate-600 mb-6">
                Theo dõi quá trình học tập của AI, các học thuyết được tích lũy và những hiểu biết được rút ra từ dữ liệu.
            </p>

            <CollapsibleSection title="Học thuyết AI hiện tại">
                <MarkdownRendererComponent
                    content={aiTheory}
                    placeholderText="Học thuyết cốt lõi của AI về hành vi mực nước ngầm sẽ xuất hiện ở đây."
                />
            </CollapsibleSection>

            <CollapsibleSection title="Giải thích AI hiện tại">
                <MarkdownRendererComponent
                    content={aiNaturalLanguageExplanation}
                    placeholderText="AI sẽ cung cấp giải thích về hàm và học thuyết của nó ở đây."
                />
            </CollapsibleSection>

            <CollapsibleSection title="Lịch sử Học tập AI">
                {aiTheoryHistory.length > 0 ? (
                    <div className="overflow-x-auto">
                        <table className="min-w-full bg-white border border-gray-200 rounded-lg">
                            <thead>
                                <tr>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">Lần lặp</th>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700"> {selectedPerformanceMetric == 'rmse' ? `RMSE (m)` : selectedPerformanceMetric == 'mse' ? `MSE (m²)` : `MAE (m)`}</th>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">Mô hình</th>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">Học thuyết (Tóm tắt)</th>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">Giải thích (Tóm tắt)</th>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">Tham số ARIMA</th>
                                    <th className="py-2 px-4 border-b text-left text-sm font-semibold text-gray-700">Kernel GP</th>
                                </tr>
                            </thead>
                            <tbody>
                                {aiTheoryHistory.map((entry, index) => {
                                    const metric = entry.metrics ? entry.metrics[selectedPerformanceMetric] : null;
                                    const hasError = metric === null || metric === Infinity || isNaN(metric);

                                    return (
                                        <tr key={index}>
                                            <td className="py-2 px-4 border-b text-sm text-gray-800">{entry.iteration}</td>
                                            <td className={`py-2 px-4 border-b text-sm font-semibold ${hasError ? 'text-red-600' : (Math.abs(metric) > 0.5 ? 'text-yellow-600' : 'text-green-600')}`}>
                                                {hasError ? 'Lỗi' : metric.toFixed(2)}
                                            </td>
                                            <td className="py-2 px-4 border-b text-sm text-gray-800">{entry.modelType === 'general' ? 'Tổng quát' : entry.modelType.toUpperCase()}</td>
                                            <td className="py-2 px-4 border-b text-sm text-gray-800">
                                                {entry.theory ? `${truncateToLines(entry.theory, 2)}` : 'N/A'}
                                            </td>
                                            <td className="py-2 px-4 border-b text-sm text-gray-800">
                                                {entry.explanation ? `${truncateToLines(entry.explanation, 2)}` : 'N/A'}
                                            </td>
                                            <td className="py-2 px-4 border-b text-sm text-gray-800">
                                                {entry.modelType === 'arima' && entry.arimaParams ?
                                                    `p:${entry.arimaParams.p}, d:${entry.arimaParams.d}, q:${entry.arimaParams.q}` :
                                                    'N/A'
                                                }
                                            </td>
                                            <td className="py-2 px-4 border-b text-sm text-gray-800">
                                                {entry.modelType === 'gaussian_process' && entry.gpKernelType ?
                                                    entry.gpKernelType :
                                                    'N/A'
                                                }
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                ) : (
                    <p className="text-slate-600">Chưa có lịch sử học tập AI nào để hiển thị. Hãy tạo và kiểm tra hàm AI!</p>
                )}
            </CollapsibleSection>
        </div>
    );
};

const GISTab = ({ wellLocations, groundwaterData, waterQualityData, weatherForecast, waterUsage, selectedWellId, setSelectedWellId, showMessage }) => {
    const mapRef = useRef(null);
    const tileLayerRef = useRef(null);
    const markersRef = useRef([]);
    const [isMapInitialized, setIsMapInitialized] = useState(false); // New state to track map initialization

    // Load Leaflet CSS
    useEffect(() => {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = 'https://unpkg.com/leaflet@1.7.1/dist/leaflet.css';
        document.head.appendChild(link);
        return () => {
            document.head.removeChild(link);
        };
    }, []);

    // Load Leaflet JS
    useEffect(() => {
        const script = document.createElement('script');
        script.src = 'https://unpkg.com/leaflet@1.7.1/dist/leaflet.js';
        script.async = true;
        document.body.appendChild(script);

        script.onload = () => {
            if (!mapRef.current) {
                // Initialize map only once
                const map = L.map('map').setView([10.76, 106.70], 13); // Default to a central location in Vietnam

                tileLayerRef.current = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                }).addTo(map);

                mapRef.current = map;
                setIsMapInitialized(true); // Set map as initialized
            }
        };

        return () => {
            if (mapRef.current) {
                mapRef.current.remove();
                mapRef.current = null;
                tileLayerRef.current = null;
            }
            // Check if script still exists before attempting to remove
            const existingScript = document.querySelector('script[src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"]');
            if (existingScript) {
                document.body.removeChild(existingScript);
            }
        };
    }, []);

    // Effect to add/update markers and center map
    useEffect(() => {
        if (!isMapInitialized || !mapRef.current || !wellLocations) return;

        // Clear existing markers
        markersRef.current.forEach(marker => marker.remove());
        markersRef.current = [];

        // Add new markers
        wellLocations.forEach(well => {
            const latestGwl = groundwaterData.filter(d => d.wellId === well.id)
                                             .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))[0]?.gwl;
            const latestEc = groundwaterData.filter(d => d.wellId === well.id)
                                            .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))[0]?.ec;
            const latestPh = waterQualityData.filter(d => d.wellId === well.id)
                                             .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))[0]?.ph;

            let popupContent = `<b>${well.name} (${well.id})</b><br>`;
            popupContent += `Vĩ độ: ${well.lat.toFixed(4)}, Kinh độ: ${well.lon.toFixed(4)}<br>`;
            popupContent += `GWL gần nhất: ${latestGwl !== undefined ? `${latestGwl.toFixed(2)} m bgs` : 'N/A'}<br>`;
            popupContent += `EC gần nhất: ${latestEc !== undefined ? `${latestEc.toFixed(0)} µS/cm` : 'N/A'}<br>`;
            popupContent += `pH gần nhất: ${latestPh !== undefined ? `${latestPh.toFixed(1)}` : 'N/A'}`;

            const marker = L.marker([well.lat, well.lon])
                .addTo(mapRef.current)
                .bindPopup(popupContent);

            marker.on('click', () => {
                setSelectedWellId(well.id);
                showMessage('Thông tin Giếng', `Đã chọn giếng: ${well.name} (${well.id})`, 'info');
            });

            markersRef.current.push(marker);
        });

        // Center map on selected well if available, otherwise on the first well if any
        if (selectedWellId) {
            const well = wellLocations.find(w => w.id === selectedWellId);
            if (well) {
                mapRef.current.setView([well.lat, well.lon], 15); // Zoom in on selected well
            }
        } else if (wellLocations.length > 0) {
            // If no well selected, but wells exist, center on the first one and select it
            mapRef.current.setView([wellLocations[0].lat, wellLocations[0].lon], 13);
            setSelectedWellId(wellLocations[0].id); // Ensure the first well is also selected in the state
        }

    }, [isMapInitialized, wellLocations, groundwaterData, waterQualityData, weatherForecast, waterUsage, selectedWellId, setSelectedWellId, showMessage]);

    return (
        <div className="p-6 bg-white rounded-xl shadow-lg border border-slate-200">
            <h2 className="text-3xl font-bold mb-6 text-slate-800">GIS & Bản đồ</h2>
            <p className="text-slate-600 mb-6">
                Trực quan hóa vị trí các giếng và dữ liệu liên quan trên bản đồ tương tác.
            </p>

            <div className="mb-6">
                <label htmlFor="well-select-map" className="block text-slate-700 text-sm font-bold mb-2">
                    Chọn giếng để tập trung trên bản đồ:
                </label>
                <select
                    id="well-select-map"
                    className="shadow border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                    value={selectedWellId}
                    onChange={(e) => setSelectedWellId(e.target.value)}
                >
                    {wellLocations.length === 0 ? (
                        <option value="">Không có giếng nào</option>
                    ) : (
                        wellLocations.map(well => (
                            <option key={well.id} value={well.id}>{well.name} ({well.id})</option>
                        ))
                    )}
                </select>
            </div>

            <div id="map" className="w-full h-96 rounded-lg shadow-md border border-gray-200"></div>
        </div>
    );
};

// New component for Statistical Validation Tab
const StatisticalValidationTab = React.memo(({
    selectedPredictionModel,
    historicalPredictionResults, // Array of {timestamp, actualGwl, predictedGwl, error} for residual vs time plot
    residuals, // Array of just error values for other plots
    meanResidual,
    stdDevResiduals,
    skewnessResiduals,
    kurtosisResiduals,
    acfResidualsData,
    qqPlotData,
    histogramBinsData,
    rawGroundwaterAcfData,
    aiStatisticalAnalysis,
    isCalculatingStatistics,
    filteredGroundwaterData // Pass raw groundwater data for its chart
}) => {
    return (
        <div className="p-6 bg-white rounded-xl shadow-lg border border-slate-200">
            <h2 className="text-3xl font-bold mb-6 text-slate-800">
                Kiểm định Thống kê cho Mô hình: {selectedPredictionModel === 'general' ? 'Tổng quát' : selectedPredictionModel === 'arima' ? 'ARIMA' : 'Gaussian Process'}
            </h2>

            {isCalculatingStatistics && (
                <div className="text-center text-blue-600 font-semibold mb-4">
                    Đang tính toán và phân tích thống kê...
                </div>
            )}

            {!isCalculatingStatistics && (
                <>
                    <CollapsibleSection title="Thống kê Phần dư Cơ bản" initialOpen={true}>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-slate-700">
                            <div>
                                <span className="font-semibold">Trung bình Phần dư:</span> {meanResidual.toFixed(4)}
                            </div>
                            <div>
                                <span className="font-semibold">Độ lệch chuẩn Phần dư:</span> {stdDevResiduals.toFixed(4)}
                            </div>
                            <div>
                                <span className="font-semibold">Độ xiên Phần dư:</span> {skewnessResiduals.toFixed(4)}
                            </div>
                            <div>
                                <span className="font-semibold">Độ nhọn Phần dư:</span> {kurtosisResiduals.toFixed(4)}
                            </div>
                        </div>
                    </CollapsibleSection>

                    <CollapsibleSection title="Biểu đồ Phần dư (Residuals Diagnostics)" initialOpen={true}>
                        <div className="space-y-8">
                            <div>
                                <h3 className="text-lg font-semibold mb-2 text-slate-700">1. Phần dư theo Thời gian (Residuals vs. Time)</h3>
                                {historicalPredictionResults.length > 0 ? (
                                    <ResidualsVsTimeChart data={historicalPredictionResults} />
                                ) : (
                                    <p className="text-slate-600">Không có dữ liệu phần dư để hiển thị. Hãy thử kiểm tra dự đoán AI.</p>
                                )}
                            </div>
                            <div>
                                <h3 className="text-lg font-semibold mb-2 text-slate-700">2. Phân phối Phần dư (Histogram)</h3>
                                {histogramBinsData.length > 0 ? (
                                    <ResidualsHistogramChart data={histogramBinsData} />
                                ) : (
                                    <p className="text-slate-600">Không có dữ liệu tần suất phần dư để hiển thị.</p>
                                )}
                            </div>
                            <div>
                                <h3 className="text-lg font-semibold mb-2 text-slate-700">3. Biểu đồ QQ Plot của Phần dư</h3>
                                {qqPlotData.length > 0 ? (
                                    <QQPlotChart data={qqPlotData} />
                                ) : (
                                    <p className="text-slate-600">Không có dữ liệu QQ Plot của phần dư để hiển thị.</p>
                                )}
                            </div>
                            <div>
                                <h3 className="text-lg font-semibold mb-2 text-slate-700">4. Tự tương quan của Phần dư (ACF Plot)</h3>
                                {acfResidualsData.length > 0 ? (
                                    <ACFPlotChart data={acfResidualsData} />
                                ) : (
                                    <p className="text-slate-600">Không có dữ liệu tự tương quan của phần dư để hiển thị.</p>
                                )}
                            </div>
                        </div>
                    </CollapsibleSection>

                    {(selectedPredictionModel === 'arima') && (
                        <CollapsibleSection title="Kiểm định Dữ liệu Gốc (Raw Data Diagnostics - Tính dừng)" initialOpen={true}>
                            <div className="space-y-8">
                                <div>
                                    <h3 className="text-lg font-semibold mb-2 text-slate-700">1. Chuỗi thời gian Dữ liệu Nước ngầm Gốc</h3>
                                    {filteredGroundwaterData.length > 0 ? (
                                        <RawGroundwaterDataChart data={filteredGroundwaterData} />
                                    ) : (
                                        <p className="text-slate-600">Không có dữ liệu nước ngầm gốc để hiển thị.</p>
                                    )}
                                </div>
                                <div>
                                    <h3 className="text-lg font-semibold mb-2 text-slate-700">2. Tự tương quan của Dữ liệu Nước ngầm Gốc (ACF Plot)</h3>
                                    {rawGroundwaterAcfData.length > 0 ? (
                                        <ACFPlotChart data={rawGroundwaterAcfData} />
                                    ) : (
                                    <p className="text-slate-600">Không có dữ liệu tự tương quan của dữ liệu nước ngầm gốc để hiển thị.</p>
                                )}
                            </div>
                            </div>
                        </CollapsibleSection>
                    )}

                    <CollapsibleSection title="Giải thích và Suy luận của AI" initialOpen={true} bgColor="bg-green-50" borderColor="border-green-200" textColor="text-green-800" headerBgColor="bg-green-100" headerTextColor="text-green-900">
                        <MarkdownRendererComponent
                            content={aiStatisticalAnalysis}
                            placeholderText="AI sẽ phân tích các kiểm định thống kê và đưa ra suy luận tại đây..."
                        />
                    </CollapsibleSection>
                </>
            )}
        </div>
    );
});


function App() {
    const [isFirebaseEnabled, setIsFirebaseEnabled] = useState(false);
    // --- State Variables ---
    const [currentTab, setCurrentTab] = useState('dashboard'); // 'dashboard', 'data-management', 'prediction-optimization', 'sustainability-compliance', 'ai-learning-insights', 'gis', 'statistical-validation'

    // Firebase related states
    const [db, setDb] = useState(null);
    const [auth, setAuth] = useState(null);
    const [userId, setUserId] = useState("09943658684084029073"); // Firebase user ID
    const [dataStorageMode, setDataStorageMode] = useState('local'); // 'local' or 'firestore'

    // Data States
    const [wellLocations, setWellLocations] = useState([
        // Default well for local mode
        { id: 'WELL_001', name: 'Giếng 1', lat: 10.7712, lon: 106.6975 }
    ]);
    const [groundwaterData, setGroundwaterData] = useState([]); // GWL, EC
    const [waterQualityData, setWaterQualityData] = useState([]); // pH, DO, Turbidity
    const [weatherForecast, setWeatherForecast] = useState([]); // Precipitation, Temperature
    const [waterUsage, setWaterUsage] = useState([]); // Pumping, Consumption

    const [selectedWellId, setSelectedWellId] = useState('');

    // AI Prediction & Optimization States
    // New state to store predictions per well
    const [allWellForecasts, setAllWellForecasts] = useState({}); // { wellId: { predictions: number[], dates: string[], errors: object[], metrics: object, futureCiBounds: object[] } }
    
    const [isGeneratingAiFunction, setIsGeneratingAiFunction] = useState(false);
    const [aiFunctionError, setAiFunctionError] = useState(null);
    const [isAiFunctionChecked, setIsAiFunctionChecked] = useState(false); // Track if current AI function has been checked
    const [aiIterationCount, setAiIterationCount] = useState(0);

    // NEW: State for model-specific AI data
    const [aiModelSpecificData, setAiModelSpecificData] = useState({
        general: {
            functionBody: DEFAULT_PREDICTION_FUNCTION_BODY,
            theory: '',
            explanation: '',
            lastValidState: null, // Stores function, theory, explanation, and best performance for this model
        },
        arima: {
            functionBody: DEFAULT_PREDICTION_FUNCTION_BODY, // Or a specific ARIMA default
            theory: '',
            explanation: '',
            lastValidState: null, // Stores function, theory, explanation, and best performance for this model
            arimaParams: { p: 1, d: 1, q: 1 },
        },
        gaussian_process: {
            functionBody: DEFAULT_PREDICTION_FUNCTION_BODY, // Or a specific GP default
            theory: '',
            explanation: '',
            lastValidState: null, // Stores function, theory, explanation, and best performance for this model
            gpKernelType: 'RBF',
        }
    });

    const [selectedPredictionModel, setSelectedPredictionModel] = useState('general'); // 'general', 'arima', 'gaussian_process'
    const [selectedPerformanceMetric, setSelectedPerformanceMetric] = useState('rmse'); // 'rmse', 'mse', 'mae'

    // Derived states based on selectedPredictionModel from aiModelSpecificData
    const aiPredictionFunctionBody = aiModelSpecificData[selectedPredictionModel]?.functionBody || DEFAULT_PREDICTION_FUNCTION_BODY;
    const aiTheory = aiModelSpecificData[selectedPredictionModel]?.theory || '';
    const aiNaturalLanguageExplanation = aiModelSpecificData[selectedPredictionModel]?.explanation || '';
    const arimaParams = aiModelSpecificData.arima?.arimaParams || { p: 1, d: 1, q: 1 };
    const gpKernelType = aiModelSpecificData.gaussian_process?.gpKernelType || 'RBF';

    // Setter functions for model-specific data
    const setAiPredictionFunctionBody = useCallback((body) => {
        setAiModelSpecificData(prev => ({
            ...prev,
            [selectedPredictionModel]: { ...prev[selectedPredictionModel], functionBody: body }
        }));
    }, [selectedPredictionModel]);

    const setAiTheory = useCallback((theory) => {
        setAiModelSpecificData(prev => ({
            ...prev,
            [selectedPredictionModel]: { ...prev[selectedPredictionModel], theory: theory }
        }));
    }, [selectedPredictionModel]);

    const setAiNaturalLanguageExplanation = useCallback((explanation) => {
        setAiModelSpecificData(prev => ({
            ...prev,
            [selectedPredictionModel]: { ...prev[selectedPredictionModel], explanation: explanation }
        }));
    }, [selectedPredictionModel]);

    // Setter for lastValidState for the current model
    const setLastValidAiModelState = useCallback((state) => {
        setAiModelSpecificData(prev => ({
            ...prev,
            [selectedPredictionModel]: { ...prev[selectedPredictionModel], lastValidState: state }
        }));
    }, [selectedPredictionModel]);


    const setArimaParams = useCallback((params) => {
        setAiModelSpecificData(prev => ({
            ...prev,
            arima: { ...prev.arima, arimaParams: params }
        }));
    }, []); // Not dependent on selectedModel, specifically for arima

    const setGpKernelType = useCallback((type) => {
        setAiModelSpecificData(prev => ({
            ...prev,
            gaussian_process: { ...prev.gaussian_process, gpKernelType: type }
        }));
    }, []); // Not dependent on selectedModel, specifically for gaussian_process


    const [aiTheoryHistory, setAiTheoryHistory] = useState([]); // Stores historical AI theories and performance

    // Data Management States
    const [isGeneratingPlausibleData, setIsGeneratingPlausibleData] = useState(false);
    const [aiDataSchemaExplanation, setAiDataSchemaExplanation] = useState('');
    const [isGeneratingDataSchemaExplanation, setIsGeneratingDataSchemaExplanation] = useState(false);
    const [isAnalyzingCurrentMismatches, setIsAnalyzingCurrentMismatches] = useState(false);
    const [followUpSchemaQuestion, setFollowUpSchemaQuestion] = useState(''); // New state for follow-up question
    // const [isAskingSchemaQuestion, setIsAskingSchemaQuestion] = useState(false); // This state is no longer needed

    const [promptMode, setPromptMode] = useState('mid-end'); // 'low-end', 'mid-end', 'high-end'
    const [userHint, setUserHint] = useState('');
    const [isAISuggestingHint, setIsAISuggestingHint] = useState(false);

    // Modals
    const [isMessageModalOpen, setIsMessageModalOpen] = useState(false);
    const [messageModalContent, setMessageModalContent] = useState({ title: '', message: '', type: 'info' });
    const [isConfirmModalOpen, setIsConfirmModalOpen] = useState(false);
    const [confirmModalContent, setConfirmModalContent] = useState({ title: '', message: '', onConfirm: () => { }, onCancel: () => { } });

    const [isImportModalOpen, setIsImportModalOpen] = useState(false);
    const [importDataType, setImportDataType] = useState(null); // 'groundwater', 'waterQuality', etc.
    const [jsonInput, setJsonInput] = useState('');
    const [importErrorMessage, setImportErrorMessage] = useState('');
    const [selectedFileName, setSelectedFileName] = useState('');
    const fileInputKey = useRef(Date.now()); // Key to reset file input

    // Sustainability & Compliance States
    const [sustainabilityInsights, setSustainabilityInsights] = useState({ details: '', recommendations: '' });
    const [isGeneratingSustainabilityInsights, setIsGeneratingSustainabilityInsights] = useState(false);

    // New state for AI Dashboard Recommendation
    const [aiDashboardRecommendation, setAiDashboardRecommendation] = useState('');

    // NEW: State for statistical validation data
    const [historicalPredictionResults, setHistoricalPredictionResults] = useState([]); // {timestamp, actualGwl, predictedGwl, error}
    const [residuals, setResiduals] = useState([]); // Just the error values for other plots
    const [meanResidual, setMeanResidual] = useState(0);
    const [stdDevResiduals, setStdDevResiduals] = useState(0);
    const [skewnessResiduals, setSkewnessResiduals] = useState(0);
    const [kurtosisResiduals, setKurtosisResiduals] = useState(0);
    const [acfResidualsData, setAcfResidualsData] = useState([]);
    const [qqPlotData, setQqPlotData] = useState([]);
    const [histogramBinsData, setHistogramBinsData] = useState([]);
    const [rawGroundwaterAcfData, setRawGroundwaterAcfData] = useState([]);
    const [aiStatisticalAnalysis, setAiStatisticalAnalysis] = useState('');
    const [isCalculatingStatistics, setIsCalculatingStatistics] = useState(false);


    // --- Helper Functions for Modals ---
    const showMessage = useCallback((title, message, type = 'info') => {
        setMessageModalContent({ title, message, type });
        setIsMessageModalOpen(true);
    }, []);

    const closeMessageModal = useCallback(() => {
        setIsMessageModalOpen(false);
        setMessageModalContent({ title: '', message: '', type: 'info' });
    }, []);

    const showConfirm = useCallback((title, message, onConfirm, onCancel, type = 'warning') => {
        setConfirmModalContent({ title, message, onConfirm, onCancel, type });
        setIsConfirmModalOpen(true);
    }, []);

    const closeConfirmModal = useCallback(() => {
        setIsConfirmModalOpen(false);
        setConfirmModalContent({ title: '', message: '', onConfirm: () => { }, onCancel: () => { } });
    }, []);

    // --- File Change Handler for JSON Input Modal ---
    const handleFileChange = useCallback((e) => {
        const file = e.target.files[0];
        if (file) {
            setSelectedFileName(file.name);
            const reader = new FileReader();
            reader.onload = (event) => {
                try {
                    const content = event.target.result;
                    setJsonInput(content);
                    setImportErrorMessage(''); // Clear previous error on new file selection
                } catch (error) {
                    setImportErrorMessage(`Lỗi đọc tệp: ${error.message}`);
                    setJsonInput(''); // Clear input if error
                }
            };
            reader.onerror = () => {
                setImportErrorMessage('Lỗi khi đọc tệp.');
                setJsonInput('');
            };
            reader.readAsText(file);
        } else {
            setSelectedFileName('');
            setJsonInput('');
        }
    }, []);

    // --- Firebase Initialization & Authentication ---
    useEffect(() => {
        let firebaseApp; // Khai báo biến cục bộ bên trong useEffect
        try {
            // Kiểm tra xem các biến môi trường thiết yếu có tồn tại không
            if (!import.meta.env.VITE_FIREBASE_API_KEY) {
                throw new Error("Missing Firebase Config");
            }
    
            if (!getApps().length) {
                firebaseApp = initializeApp(firebaseConfig);
            } else {
                firebaseApp = getApp();
            }
            const dbInstance = getFirestore(firebaseApp);
            const authInstance = getAuth(firebaseApp);
    
            setDb(dbInstance);
            setAuth(authInstance);
            setIsFirebaseEnabled(true); // Đánh dấu Firebase đã sẵn sàng

            const unsubscribe = onAuthStateChanged(authInstance, async (user) => {
                if (user) {
                    setUserId(user.uid);
                    showMessage('Xác thực Firebase', `${t('header.login.success')}, User ID: ${user.uid}`, 'success');
                } else {
                    try {
    
                        if (initialAuthToken) {
                            await signInWithCustomToken(authInstance, initialAuthToken);
                            currentUserId = authInstance.currentUser?.uid || crypto.randomUUID();
                            setUserId(currentUserId); // Cập nhật userId sau khi đăng nhập
                            showMessage('Xác thực Firebase', `${t('header.login.success')}, User ID: ${currentUserId}`, 'success');
                        } else {
                            // Try to sign in anonymously if no user is authenticated
                            const anonymousUser = await signInAnonymously(authInstance);
                            setUserId(anonymousUser.user.uid);
                            showMessage('Xác thực Firebase', `${t('header.login.success')}, User ID: ${anonymousUser.user.uid}`, 'success');
                        }
                        
                    } catch (error) {
                        setUserId('anonymous-user'); // Fallback if anonymous sign-in fails
                        showMessage('Xác thực Firebase', `Đăng nhập thất bại: ${error.message}. Tiếp tục với User ID ẩn danh.`, 'error');
                        console.error("Firebase Auth error:", error);
                    }
                }
            });
    
            // Initial sign-in with custom token if available (for Canvas environment)
            if (typeof __initial_auth_token !== 'undefined' && __initial_auth_token) {
                signInWithCustomToken(authInstance, __initial_auth_token)
                    .catch(error => {
                        console.error("Error signing in with custom token:", error);
                        // Fallback to anonymous if custom token fails
                        signInAnonymously(authInstance)
                            .then(anonymousUser => {
                                setUserId(anonymousUser.user.uid);
                                showMessage('Xác thực Firebase', `Đăng nhập ẩn danh thành công (Custom token thất bại): ${anonymousUser.user.uid}`, 'warning');
                            })
                            .catch(anonError => {
                                setUserId('anonymous-user-fallback');
                                showMessage('Xác thực Firebase', `Không thể đăng nhập. Vui lòng kiểm tra kết nối.`, 'error');
                                console.error("Anonymous sign-in fallback failed:", anonError);
                            });
                    });
            } else {
                // If no custom token, try anonymous sign-in directly
                signInAnonymously(authInstance)
                    .then(anonymousUser => {
                        setUserId(anonymousUser.user.uid);
                        showMessage('Xác thực Firebase', `${t('header.login.anon')}: ${anonymousUser.user.uid}`, 'success');
                    })
                    .catch(error => {
                        setUserId('anonymous-user-fallback');
                        showMessage('Xác thực Firebase', `Không thể đăng nhập. Vui lòng kiểm tra kết nối.`, 'error');
                        console.error("Anonymous sign-in failed:", error);
                    });
            }
            return () => unsubscribe();
        } catch (error) {
            console.error("Firebase fail-safe triggered:", error);
            setIsFirebaseEnabled(false);
            setDataStorageMode('local');
            showMessage('Hệ thống', 'Firebase chưa được cấu hình. Chế độ Lưu trữ đám mây sẽ bị vô hiệu hóa.', 'warning');
        }
    }, [showMessage]);
    
    // --- Data Filtering for Display ---
    const filteredGroundwaterData = useMemo(() => {
        return groundwaterData.filter(d => d.wellId === selectedWellId)
                              .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    }, [groundwaterData, selectedWellId]);

    const filteredWaterQualityData = useMemo(() => {
        return waterQualityData.filter(d => d.wellId === selectedWellId)
                               .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    }, [waterQualityData, selectedWellId]);

    const filteredWeatherForecast = useMemo(() => {
        return weatherForecast.filter(d => d.wellId === selectedWellId)
                              .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    }, [weatherForecast, selectedWellId]);

    const filteredWaterUsage = useMemo(() => {
        return waterUsage.filter(d => d.wellId === selectedWellId)
                         .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    }, [waterUsage, selectedWellId]);

    // Derive sevenDayGroundwaterPrediction and sevenDayGroundwaterPredictionDates from allWellForecasts
    const sevenDayGroundwaterPrediction = useMemo(() => {
        return allWellForecasts[selectedWellId]?.predictions || [];
    }, [allWellForecasts, selectedWellId]);

    const sevenDayGroundwaterPredictionDates = useMemo(() => {
        return allWellForecasts[selectedWellId]?.dates || [];
    }, [allWellForecasts, selectedWellId]);

    // Derive predictionErrors from allWellForecasts
    const predictionErrors = useMemo(() => {
        return allWellForecasts[selectedWellId]?.errors || [];
    }, [allWellForecasts, selectedWellId]);

    // Derive future CI bounds
    const futureCiBounds = useMemo(() => {
        return allWellForecasts[selectedWellId]?.futureCiBounds || [];
    }, [allWellForecasts, selectedWellId]);

    const bootstrapStartStep = useMemo(() => {
        return allWellForecasts[selectedWellId]?.bootstrapStartStep || 0;
    }, [allWellForecasts, selectedWellId]);


    // --- Load/Save Session Logic ---
    // Moved processLoadedSessionData inside App for access to all setStates
    const processLoadedSessionData = useCallback((data) => {
        if (data) {
            setWellLocations(data.wellLocations || []);
            setGroundwaterData(data.groundwaterData || []);
            setWaterQualityData(data.waterQualityData || []);
            setWeatherForecast(data.weatherForecast || []);
            setWaterUsage(data.waterUsage || []);
            setSelectedWellId(data.selectedWellId || '');
            // Ensure futureCiBounds is loaded as well
            setAllWellForecasts(data.allWellForecasts || {}); // Load new per-well predictions (now includes futureCiBounds)
            setIsAiFunctionChecked(data.isAiFunctionChecked || false);
            setAiIterationCount(data.aiIterationCount || 0);
            setAiTheoryHistory(data.aiTheoryHistory || []);
            setPromptMode(data.promptMode || 'mid-end');
            setUserHint(data.userHint || '');
            setIsAISuggestingHint(data.isAISuggestingHint || false);
            setCurrentTab(data.currentTab ?? 'dashboard'); // Restore last active tab
            setSustainabilityInsights(data.sustainabilityInsights || { details: '', recommendations: '' }); // Load sustainability insights
            setAiDashboardRecommendation(data.aiDashboardRecommendation || ''); // Load AI Dashboard Recommendation

            // NEW: Load aiModelSpecificData
            if (data.aiModelSpecificData) {
                setAiModelSpecificData(data.aiModelSpecificData);
            } else {
                // If old session data doesn't have aiModelSpecificData, initialize it from other states
                // This is a migration step for old saved sessions
                setAiModelSpecificData(prev => ({
                    general: {
                        functionBody: data.aiPredictionFunctionBody || DEFAULT_PREDICTION_FUNCTION_BODY,
                        theory: data.aiTheory || '',
                        explanation: data.aiNaturalLanguageExplanation || '',
                        lastValidState: { // Populate lastValidState from old separate states
                            functionBody: data.lastValidAiPredictionFunctionBody || DEFAULT_PREDICTION_FUNCTION_BODY,
                            theory: data.aiTheory || '',
                            explanation: data.aiNaturalLanguageExplanation || '',
                            performance: null, // Old structure didn't have detailed performance metrics
                            diagnostics: null, // New field, will be null for old sessions
                        },
                    },
                    arima: { // Assume default if not explicitly saved
                        functionBody: DEFAULT_PREDICTION_FUNCTION_BODY,
                        theory: '',
                        explanation: '',
                        lastValidState: null, // Default
                        arimaParams: { p: 1, d: 1, q: 1 },
                    },
                    gaussian_process: { // Assume default if not explicitly saved
                        functionBody: DEFAULT_PREDICTION_FUNCTION_BODY,
                        theory: '',
                        explanation: '',
                        lastValidState: null, // Default
                        gpKernelType: 'RBF',
                    }
                }));
            }
            // NEW: Set selected model from loaded data, or default
            setSelectedPredictionModel(data.selectedPredictionModel || 'general');
            setSelectedPerformanceMetric(data.selectedPerformanceMetric || 'rmse');

            // NEW: Load statistical diagnostics data if available (for statistical tab)
            setHistoricalPredictionResults(data.historicalPredictionResults || []);
            setResiduals(data.residuals || []);
            setMeanResidual(data.meanResidual || 0);
            setStdDevResiduals(data.stdDevResiduals || 0);
            setSkewnessResiduals(data.skewnessResiduals || 0);
            setKurtosisResiduals(data.kurtosisResiduals || 0);
            setAcfResidualsData(data.acfResidualsData || []);
            setQqPlotData(data.qqPlotData || []);
            setHistogramBinsData(data.histogramBinsData || []);
            setRawGroundwaterAcfData(data.rawGroundwaterAcfData || []);
            setAiStatisticalAnalysis(data.aiStatisticalAnalysis || '');


            showMessage('Thành công', 'Đã tải phiên làm việc.', 'success');
        } else {
            showMessage('Thông báo', 'Không có dữ liệu phiên để tải.', 'info');
        }
    }, [showMessage, setWellLocations, setGroundwaterData, setWaterQualityData, setWeatherForecast, setWaterUsage, setSelectedWellId, setAllWellForecasts, setIsAiFunctionChecked, setAiIterationCount, setAiTheoryHistory, setPromptMode, setUserHint, setIsAISuggestingHint, setCurrentTab, setSustainabilityInsights, setAiDashboardRecommendation, setAiModelSpecificData, setSelectedPredictionModel, setSelectedPerformanceMetric, setHistoricalPredictionResults, setResiduals, setMeanResidual, setStdDevResiduals, setSkewnessResiduals, setKurtosisResiduals, setAcfResidualsData, setQqPlotData, setHistogramBinsData, setRawGroundwaterAcfData, setAiStatisticalAnalysis]);

    const handleSaveSession = useCallback(async () => {
        const sessionData = {
            wellLocations,
            groundwaterData,
            waterQualityData,
            weatherForecast,
            waterUsage,
            selectedWellId,
            aiModelSpecificData, // NEW: Save model-specific AI data (which now includes lastValidState and its performance)
            allWellForecasts, // Save new per-well predictions (now includes futureCiBounds)
            isAiFunctionChecked,
            aiIterationCount,
            aiTheoryHistory,
            selectedPredictionModel,
            selectedPerformanceMetric, // NEW: Save selected performance metric
            promptMode,
            userHint,
            isAISuggestingHint,
            currentTab, // Save current active tab
            sustainabilityInsights, // Save sustainability insights
            aiDashboardRecommendation, // Save AI Dashboard Recommendation
            // NEW: Save statistical diagnostics data
            historicalPredictionResults,
            residuals,
            meanResidual,
            stdDevResiduals,
            skewnessResiduals,
            kurtosisResiduals,
            acfResidualsData,
            qqPlotData,
            histogramBinsData,
            rawGroundwaterAcfData,
            aiStatisticalAnalysis,
            timestamp: new Date().toISOString(),
            userId: userId || 'anonymous-user', // Include userId for debugging/tracking
        };

        if (dataStorageMode === 'local') {
            showMessage('Thông báo', 'Đang bắt đầu tải xuống tệp phiên làm việc. Vui lòng kiểm tra thư mục tải xuống của bạn.', 'info');
            const blob = new Blob([JSON.stringify(sessionData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `water_resource_session_${formatISO(new Date(), { representation: 'datetime' })}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } else if (dataStorageMode === 'firestore') {
            if (!db || !userId) {
                showMessage('Lỗi', 'Không thể lưu phiên. Vui lòng đăng nhập hoặc kết nối Firebase.', 'error');
                return;
            }
            try {
                const sessionDocRef = doc(db, `artifacts/${appId}/users/${userId}/sessions/current_session`);
                await setDoc(sessionDocRef, sessionData);
                showMessage('Thành công', 'Phiên làm việc đã được lưu vào Đám mây.', 'success');
            } catch (error) {
                showMessage('Lỗi', `Không thể lưu phiên vào Đám mây: ${error.message}`, 'error');
                console.error("Error saving session to Firestore:", error);
            }
        }
    }, [
        dataStorageMode, db, userId, wellLocations, groundwaterData, waterQualityData, weatherForecast, waterUsage,
        selectedWellId, allWellForecasts,
        isAiFunctionChecked, aiIterationCount, aiTheoryHistory, selectedPredictionModel, selectedPerformanceMetric,
        promptMode, userHint, isAISuggestingHint, currentTab, sustainabilityInsights, aiDashboardRecommendation, showMessage,
        aiModelSpecificData, appId,
        historicalPredictionResults, residuals, meanResidual, stdDevResiduals, skewnessResiduals, kurtosisResiduals,
        acfResidualsData, qqPlotData, histogramBinsData, rawGroundwaterAcfData, aiStatisticalAnalysis
    ]);

    const handleLoadSession = useCallback(async () => {
        if (dataStorageMode === 'local') {
            fileInputKey.current = Date.now(); // Reset file input to allow selecting same file again
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json';
            input.onchange = async (e) => {
                const file = e.target.files[0];
                if (file) {
                    try {
                        const reader = new FileReader();
                        reader.onload = (event) => {
                            try {
                                const loadedData = JSON.parse(event.target.result);
                                processLoadedSessionData(loadedData);
                            } catch (parseError) {
                                showMessage('Lỗi', `Không thể phân tích tệp JSON: ${parseError.message}`, 'error');
                                console.error("Error parsing JSON file:", parseError);
                            }
                        };
                        reader.readAsText(file);
                    } catch (error) {
                        showMessage('Lỗi', `Không thể đọc tệp: ${error.message}`, 'error');
                        console.error("Error reading file:", error);
                    }
                }
            };
            input.click();
        } else if (dataStorageMode === 'firestore') {
            if (!db || !userId) {
                showMessage('Lỗi', 'Không thể tải phiên. Vui lòng đăng nhập hoặc kết nối Firebase.', 'error');
                return;
            }
            try {
                const sessionDocRef = doc(db, `artifacts/${appId}/users/${userId}/sessions/current_session`);
                const sessionSnap = await getDoc(sessionDocRef);
                if (sessionSnap.exists()) {
                    processLoadedSessionData(sessionSnap.data());
                } else {
                    showMessage('Thông báo', 'Không tìm thấy phiên làm việc đã lưu trong Đám mây.', 'info');
                }
            } catch (error) {
                showMessage('Lỗi', `Không thể tải phiên từ Đám mây: ${error.message}`, 'error');
                console.error("Error loading session from Firestore:", error);
            }
        }
    }, [dataStorageMode, db, userId, processLoadedSessionData, showMessage, appId]);

    // --- Automatic Well Selection for GIS Tab and initial load ---
    useEffect(() => {
        if (wellLocations.length > 0 && (!selectedWellId || !wellLocations.some(well => well.id === selectedWellId))) {
            setSelectedWellId(wellLocations[0].id);
        } else if (wellLocations.length === 0) {
            setSelectedWellId(''); // Clear selection if no wells
        }
    }, [wellLocations, selectedWellId, setSelectedWellId]);


    // Memoize current AI prediction function for statistical calculations
    const currentAiFunction = useMemo(() => {
        try {
            return new Function('historicalGroundwaterData', 'historicalWaterQualityData', 'historicalWeatherForecast', 'historicalWaterUsage', aiPredictionFunctionBody);
        } catch (error) {
            console.error("Lỗi khi biên dịch hàm AI:", error);
            return null; // Return null if function body is invalid
        }
    }, [aiPredictionFunctionBody]);


    // NEW: useEffect for Statistical Calculations and AI Analysis
    useEffect(() => {
        const performStatisticalAnalysis = async () => {
            if (!selectedWellId || !currentAiFunction || filteredGroundwaterData.length === 0) {
                setHistoricalPredictionResults([]);
                setResiduals([]);
                setMeanResidual(0);
                setStdDevResiduals(0);
                setSkewnessResiduals(0);
                setKurtosisResiduals(0);
                setAcfResidualsData([]);
                setQqPlotData([]);
                setHistogramBinsData([]);
                setRawGroundwaterAcfData([]);
                setAiStatisticalAnalysis('Không có đủ dữ liệu hoặc hàm AI để thực hiện kiểm định thống kê.');
                setIsCalculatingStatistics(false);
                return;
            }

            setIsCalculatingStatistics(true);
            setAiStatisticalAnalysis(''); // Clear previous analysis

            try {
                const newHistoricalPredictionResults = [];
                const actualsForResiduals = [];
                const predictedForResiduals = [];

                // Calculate errors for the last N historical days by simulating predictions
                // Similar logic to handleCheckPrediction but for all available historical data that can be predicted
                const N_effective_points_for_error = filteredGroundwaterData.length;
                for (let i = 0; i < N_effective_points_for_error; i++) {
                    const actualDataPoint = filteredGroundwaterData[i];
                    
                    // Historical data available *before or including* this actualDataPoint for prediction
                    // For prediction, AI uses data *up to* the point of prediction.
                    // To get residuals, we need to predict *for* the current data point using *past* data.
                    const historicalGroundwaterSubset = filteredGroundwaterData.slice(0, i);
                    const historicalWaterQualitySubset = filteredWaterQualityData.filter(d => new Date(d.timestamp) < new Date(actualDataPoint.timestamp));
                    const historicalWeatherSubset = filteredWeatherForecast.filter(d => new Date(d.timestamp) < new Date(actualDataPoint.timestamp));
                    const historicalWaterUsageSubset = filteredWaterUsage.filter(d => new Date(d.timestamp) < new Date(actualDataPoint.timestamp));

                    if (historicalGroundwaterSubset.length > 0) {
                        try {
                            const simulatedPredictions = currentAiFunction(
                                historicalGroundwaterSubset,
                                historicalWaterQualitySubset,
                                historicalWeatherSubset,
                                historicalWaterUsageSubset
                            );

                            if (Array.isArray(simulatedPredictions) && simulatedPredictions.length >= 1 && typeof simulatedPredictions[0] === 'number' && !isNaN(simulatedPredictions[0])) {
                                const predictedGwlForThisDay = simulatedPredictions[0]; // Take the first day's prediction
                                const error = predictedGwlForThisDay - actualDataPoint.gwl;
                                newHistoricalPredictionResults.push({
                                    timestamp: actualDataPoint.timestamp,
                                    actualGwl: actualDataPoint.gwl,
                                    predictedGwl: predictedGwlForThisDay,
                                    error: error
                                });
                                actualsForResiduals.push(actualDataPoint.gwl);
                                predictedForResiduals.push(predictedGwlForThisDay);
                            } else {
                                console.warn(`Invalid prediction for ${actualDataPoint.timestamp} during statistical analysis.`);
                            }
                        } catch (execError) {
                            console.error(`Error executing AI function for statistical analysis at ${actualDataPoint.timestamp}:`, execError);
                            // Do not break, try to continue with other points
                        }
                    } else {
                         // If no historical data before current point, cannot predict, skip
                         console.warn(`No sufficient historical data before ${actualDataPoint.timestamp} for statistical analysis prediction.`);
                    }
                }

                setHistoricalPredictionResults(newHistoricalPredictionResults);
                const currentResiduals = newHistoricalPredictionResults.map(p => p.error);
                setResiduals(currentResiduals);

                // Calculate all statistical metrics for residuals
                setMeanResidual(calculateMean(currentResiduals));
                setStdDevResiduals(calculateStandardDeviation(currentResiduals));
                setSkewnessResiduals(calculateSkewness(currentResiduals));
                setKurtosisResiduals(calculateKurtosis(currentResiduals));
                setAcfResidualsData(calculateACF(currentResiduals, 7)); // ACF for 7 lags
                setQqPlotData(calculateQQPlotData(currentResiduals));
                setHistogramBinsData(calculateHistogramBins(currentResiduals, 10)); // 10 bins for histogram

                // Calculate ACF for raw groundwater data (for ARIMA)
                const rawGwlValues = filteredGroundwaterData.map(d => d.gwl);
                setRawGroundwaterAcfData(calculateACF(rawGwlValues, 7)); // ACF for raw GWL data, 7 lags

                // Prepare prompt for AI analysis of statistics
                const p_body = JSON.stringify({
                    selectedPredictionModel: selectedPredictionModel, 
                    selectedWellId: selectedWellId, 
                    meanCurrentResiduals: calculateMean(currentResiduals).toFixed(4), 
                    stdCurrentResiduals: calculateStandardDeviation(currentResiduals).toFixed(4), 
                    sknCurrentResiduals: calculateSkewness(currentResiduals).toFixed(4), 
                    ktsCurrentResiduals: calculateKurtosis(currentResiduals).toFixed(4), 
                    acfCurrentResiduals: JSON.stringify(calculateACF(currentResiduals, 3).map(d => ({ lag: d.lag, value: d.value.toFixed(4) }))), 
                    acfRawGwlValues: JSON.stringify(calculateACF(rawGwlValues, 3).map(d => ({ lag: d.lag, value: d.value.toFixed(4) }))),
                    arimaParams: arimaParams
                });

                const apiKey = ""; // Canvas will provide API key
                const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=${apiKey}`;
                const response = await fetch(`${FETCH_API_URL}/api/v2/ai2_fetch/raw_text`, {
                    method: 'POST',
                    headers: { 
                        'Content-Type': 'application/json',
                        'X-Client-Key': FETCH_CLIENT_KEY
                    },
                    body: p_body
                });
                const rawText = await response.text();
                let result;
                result = JSON.parse(rawText);

                if (result.candidates && result.candidates.length > 0 && result.candidates[0].content && result.candidates[0].content.parts && result.candidates[0].content.parts.length > 0) {
                    setAiStatisticalAnalysis(result.candidates[0].content.parts[0].text);
                } else {
                    setAiStatisticalAnalysis('Không thể tạo phân tích thống kê từ AI.');
                }

            } catch (error) {
                console.error("Lỗi khi tính toán thống kê hoặc gọi AI:", error);
                setAiStatisticalAnalysis(`Lỗi khi tạo phân tích thống kê: ${error.message}`);
                showMessage('Lỗi', `Lỗi khi tạo phân tích thống kê: ${error.message}`, 'error');
            } finally {
                setIsCalculatingStatistics(false);
            }
        };

        performStatisticalAnalysis();
    }, [
        selectedWellId,
        filteredGroundwaterData,
        filteredWaterQualityData,
        filteredWeatherForecast,
        filteredWaterUsage,
        currentAiFunction, // Trigger when AI function body changes
        selectedPredictionModel, // Trigger when model changes
        arimaParams, // Trigger for ARIMA params change
        gpKernelType, // Trigger for GP kernel type change
        showMessage,
    ]);


    // NEW: Hàm xử lý thay đổi giá trị của slider bootstrapStartStep
    const handleBootstrapStartStepChange = useCallback((event) => {
        const newStep = parseInt(event.target.value);
        setAllWellForecasts(prev => ({
            ...prev,
            [selectedWellId]: {
                ...prev[selectedWellId],
                bootstrapStartStep: newStep
            }
        }));
        // Sau khi thay đổi giá trị, chúng ta cần chạy lại dự đoán để cập nhật CI
        // Có thể gọi handleCheckPrediction() ở đây nếu muốn phản hồi ngay lập tức,
        // hoặc để người dùng tự nhấn "Kiểm tra AI" để thấy thay đổi.
        // Đối với sự đơn giản và tránh các tính toán không cần thiết, chúng ta sẽ không tự động chạy lại ngay.
        // Người dùng sẽ nhấn "Kiểm tra AI" để thấy CI cập nhật.
    }, [selectedWellId]);


    // --- AI Prediction & Optimization Functions ---
    const handleGenerateAiFunction = useCallback(async () => {
        if (!selectedWellId) {
            showMessage('Cảnh báo', 'Vui lòng chọn một giếng trước khi yêu cầu AI tạo hàm dự đoán.', 'warning');
            return;
        }
        if (filteredGroundwaterData.length === 0) {
            showMessage('Thông báo', 'Không có đủ dữ liệu lịch sử cho giếng đã chọn để AI tạo hàm dự đoán.', 'info');
            return;
        }

        setIsGeneratingAiFunction(true);
        setAiFunctionError(null);
        setIsAiFunctionChecked(false); // Reset check status when generating new function

        // Calculate total absolute error from current prediction errors (if any)
        const currentTotalAbsoluteError = predictionErrors.length > 0 ? predictionErrors.reduce((sum, err) => sum + Math.abs(err.error), 0) : null;
        
        // Truncate aiTheoryHistory for brevity in prompt based on promptMode
        let historyForPrompt = [];
        if (aiTheoryHistory.length > 0) {
            if (promptMode === 'high-end') {
                historyForPrompt = aiTheoryHistory.slice(-2).map(entry => ({
                    iteration: entry.iteration,
                    metrics: entry.metrics, // Include all metrics
                    modelType: entry.modelType,
                    theory: truncateToLines(entry.theory, 3), // More lines for high-end
                    explanation: truncateToLines(entry.explanation, 3), // More lines for high-end
                    arimaParams: entry.arimaParams,
                    gpKernelType: entry.gpKernelType,
                    diagnostics: entry.diagnostics // Include diagnostics
                }));
            } else if (promptMode === 'mid-end') {
                historyForPrompt = aiTheoryHistory.slice(-2).map(entry => ({ // Last 3 entries for mid-end
                    iteration: entry.iteration,
                    metrics: entry.metrics, // Include all metrics
                    modelType: entry.modelType,
                    theory: truncateToLines(entry.theory, 2), // Fewer lines
                    explanation: truncateToLines(entry.explanation, 2), // Fewer lines
                    arimaParams: entry.arimaParams,
                    gpKernelType: entry.gpKernelType,
                    diagnostics: entry.diagnostics // Include diagnostics
                }));
            } else { // low-end
                historyForPrompt = aiTheoryHistory.slice(-1).map(entry => ({ // Only last entry for low-end
                    iteration: entry.iteration,
                    metrics: entry.metrics, // Include all metrics
                    modelType: entry.modelType,
                    theory: truncateToLines(entry.theory, 1), // Minimal lines
                    explanation: truncateToLines(entry.explanation, 1), // Minimal lines
                    arimaParams: entry.arimaParams,
                    gpKernelType: entry.gpKernelType,
                    diagnostics: entry.diagnostics // Include diagnostics
                }));
            }
        }

        const currentModelLastValidState = aiModelSpecificData[selectedPredictionModel]?.lastValidState;

        const p_body = JSON.stringify({
            PREDICTING_PERIOD: PREDICTING_PERIOD,
            selectedWellId: selectedWellId,
            selectedPredictionModel: selectedPredictionModel, // QUAN TRỌNG: Cần gửi biến này
            GWL: JSON.stringify(filteredGroundwaterData.slice(0, -LEADING_PERIOD).slice(-5)),
            waterQuality: JSON.stringify(filteredWaterQualityData.slice(0, -LEADING_PERIOD).slice(-5)),
            weather: JSON.stringify(filteredWeatherForecast.slice(0, -LEADING_PERIOD).slice(-5)),
            waterUsage: JSON.stringify(filteredWaterUsage.slice(0, -LEADING_PERIOD).slice(-5)),
            performanceMetric: performanceMetricsCalculators[selectedPerformanceMetric].name,
            currentTotalAbsoluteError: currentTotalAbsoluteError,
            performanceMetricUnit: performanceMetricsCalculators[selectedPerformanceMetric].unit,
            userHint: userHint,
            historyForPrompt: historyForPrompt,
            aiPredictionFunctionBody: aiPredictionFunctionBody,
            arimaParams: arimaParams,
            gpKernelType: gpKernelType, 
            generationConfig: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: "OBJECT",
                    properties: {
                        "functionBody": { "type": "STRING" },
                        "theory": { "type": "STRING" },
                        "explanation": { "type": "STRING" },
                        "optimalArimaParams": { 
                            "type": "OBJECT",
                            "properties": { "p": { "type": "NUMBER" }, "d": { "type": "NUMBER" }, "q": { "type": "NUMBER" } }
                        },
                        "optimalGpKernelType": { "type": "STRING" }
                    },
                    required: ["functionBody", "theory", "explanation"]
                }
            }
        });

        try {
            const apiKey = ""; // Canvas will provide API key
            const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=${apiKey}`;
            const response = await fetch(`${FETCH_API_URL}/api/v2/ai3_fetch/raw_text`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'X-Client-Key': FETCH_CLIENT_KEY
                },
                body: p_body
            });
            const rawText = await response.text();
            let result;
            result = JSON.parse(rawText);

            if (result.candidates && result.candidates.length > 0 && result.candidates[0].content && result.candidates[0].content.parts && result.candidates[0].content.parts.length > 0) {
                const textResponse = result.candidates[0].content.parts[0].text;
                let parsedResponse;
                try {
                    parsedResponse = JSON.parse(textResponse);
                } catch (jsonError) {
                    console.error("Failed to parse AI JSON response:", textResponse, jsonError);
                    setAiFunctionError(`Lỗi phân tích cú pháp phản hồi AI: ${jsonError.message}. Phản hồi thô: ${textResponse.substring(0, 200)}...`);
                    // --- NEW: AI Suggestion for User Hint based on JSON parsing error ---
                    if (isAISuggestingHint) {
                        try {
                            const analysis = await analyzeMalformedAiResponse(`Lỗi JSON: ${jsonError.message}`, `Phản hồi thô của AI:\n\`\`\`json\n${textResponse}\n\`\`\`\nAI đang tạo một hàm JavaScript để dự đoán mực nước ngầm (GWL) trong ${PREDICTING_PERIOD} ngày tới, cần trả về một mảng ${PREDICTING_PERIOD} số. Đảm bảo phản hồi JSON của bạn tuân thủ đúng schema.`);
                            setUserHint(`AI đề xuất: ${analysis}`);
                            showMessage('Gợi ý AI', 'AI đã phân tích lỗi JSON và đưa ra gợi ý mới.', 'info');
                        } catch (hintError) {
                            console.error("Lỗi khi AI tạo gợi ý người dùng cho lỗi JSON:", hintError);
                        }
                    }
                    // --- END NEW ---
                    setIsGeneratingAiFunction(false); // Stop loading spinner
                    // Crucially, if JSON parsing fails, the prediction function cannot be extracted or run.
                    // So we must clear previous predictions to reflect this broken state.
                    setAllWellForecasts(prev => ({
                        ...prev,
                        [selectedWellId]: { predictions: [], dates: [], errors: [], metrics: {}, futureCiBounds: []}
                    }));
                    return;
                }

                setAiPredictionFunctionBody(parsedResponse.functionBody);
                setAiTheory(parsedResponse.theory);
                setAiNaturalLanguageExplanation(parsedResponse.explanation);
                setAiIterationCount(prev => prev + 1);
                setAiFunctionError(null);

                // Update ARIMA or GP parameters if AI suggests them
                if (parsedResponse.optimalArimaParams && selectedPredictionModel === 'arima') {
                    setArimaParams(parsedResponse.optimalArimaParams);
                }
                if (parsedResponse.optimalGpKernelType && selectedPredictionModel === 'gaussian_process') {
                    setGpKernelType(parsedResponse.optimalGpKernelType); 
                }

                // --- NEW: Generate initial 7-day future prediction with the newly created function ---
                let initialFuturePredictions = [];
                let initialFutureDates = [];
                let initialFutureCiBounds = []; // Initialize CI bounds
                
                try {
                    const executePredictionFunction = new Function('historicalGroundwaterData', 'historicalWaterQualityData', 'historicalWeatherForecast', 'historicalWaterUsage', parsedResponse.functionBody);
                    initialFuturePredictions = executePredictionFunction(filteredGroundwaterData, filteredWaterQualityData, filteredWeatherForecast, filteredWaterUsage);

                    if (!Array.isArray(initialFuturePredictions) || initialFuturePredictions.length !== PREDICTING_PERIOD || initialFuturePredictions.some(isNaN)) {
                        throw new Error(`Hàm AI mới tạo phải trả về một mảng ${PREDICTING_PERIOD} số.`);
                    }

                    // For initial generation, we don't have historical residuals yet for a meaningful CI.
                    // So, we'll initialize CI with a placeholder or based on some default.
                    // Or, more accurately, the CI will be calculated after `handleCheckPrediction`.
                    // Let's initialize with empty bounds for now.
                    initialFutureCiBounds = initialFuturePredictions.map(p => ({ upper: null, lower: null }));


                    const lastHistoricalTimestamp = filteredGroundwaterData.length > 0
                        ? new Date(filteredGroundwaterData[filteredGroundwaterData.length - 1].timestamp)
                        : new Date();
                    for (let i = 0; i < PREDICTING_PERIOD; i++) {
                        initialFutureDates.push(formatISO(addDays(lastHistoricalTimestamp, i + 1), { representation: 'date' }));
                    }

                    // Update allWellForecasts - clear errors for new function generation initially
                    setAllWellForecasts(prev => ({
                        ...prev,
                        [selectedWellId]: {
                            predictions: initialFuturePredictions,
                            dates: initialFutureDates,
                            errors: [], 
                            metrics: {}, // Clear metrics too for a new function
                            futureCiBounds: initialFutureCiBounds, // Initialize CI bounds
                            bootstrapStartStep: prev[selectedWellId] ? prev[selectedWellId].bootstrapStartStep : 0
                        }
                    }));
                    showMessage('Thành công', `Hàm dự đoán AI đã được tạo/cải thiện và dự đoán ${PREDICTING_PERIOD} ngày tới đã được tạo.`, 'success');

                } catch (predictionExecutionError) {
                    const errorMessage = `Lỗi khi tạo dự đoán ban đầu từ hàm AI mới: ${predictionExecutionError.message}. Hàm có thể không chính xác.`;
                    setAiFunctionError(errorMessage);
                    console.error("Error executing newly generated AI function for initial prediction:", predictionExecutionError);
                    
                    // --- NEW: AI Suggestion for User Hint based on runtime error during initial execution ---
                    if (isAISuggestingHint) {
                        try {
                            const analysis = await analyzeMalformedAiResponse(predictionExecutionError.message, `AI đã tạo hàm này:\n\`\`\`javascript\n${parsedResponse.functionBody}\n\`\`\`\nHàm này dự kiến nhận dữ liệu thủy văn và trả về MỘT MẢNG ${PREDICTING_PERIOD} SỐ dự đoán GWL. Vui lòng đảm bảo hàm này thực thi hợp lệ và trả về đúng định dạng.`);
                            setUserHint(`AI đề xuất: ${analysis}`);
                            showMessage('Gợi ý AI', 'AI đã phân tích lỗi runtime và đưa ra gợi ý mới.', 'info');
                        } catch (hintError) {
                            console.error("Lỗi khi AI tạo gợi ý người dùng cho lỗi runtime:", hintError);
                        }
                    }
                    // --- END NEW ---

                    // Clear predictions for this well if invalid
                    setAllWellForecasts(prev => ({
                        ...prev,
                        [selectedWellId]: { predictions: [], dates: [], errors: [], metrics: {}, futureCiBounds: [], bootstrapStartStep: prev[selectedWellId] ? prev[selectedWellId].bootstrapStartStep : 0 }
                    }));
                    showMessage('Lỗi', `Hàm dự đoán AI đã được tạo nhưng kết quả ban đầu không hợp lệ: ${predictionExecutionError.message}`, 'error');
                }
                
                // Set last valid state for the current model if the initial prediction was valid
                if (initialFuturePredictions.length === PREDICTING_PERIOD && initialFuturePredictions.every(val => typeof val === 'number' && !isNaN(val))) {
                    setLastValidAiModelState({
                        functionBody: parsedResponse.functionBody,
                        theory: parsedResponse.theory,
                        explanation: parsedResponse.explanation,
                        arimaParams: selectedPredictionModel === 'arima' ? arimaParams : null,
                        gpKernelType: selectedPredictionModel === 'gaussian_process' ? gpKernelType : null,
                        performance: null, // No historical check done yet, so no performance
                        diagnostics: null // Will be populated after check
                    });
                } else {
                    // If the newly generated function itself is invalid, clear last valid state or revert to a known good default
                    setLastValidAiModelState(null); // Or you could set a default "empty" state here
                }

            } else {
                setAiFunctionError('Không nhận được phản hồi hợp lệ từ AI.');
                showMessage('Lỗi', 'Không nhận được phản hồi hợp lệ từ AI.', 'error');
                console.error("AI response missing candidates or content:", result);
                setAllWellForecasts(prev => ({
                    ...prev,
                    [selectedWellId]: { predictions: [], dates: [], errors: [], metrics: {}, futureCiBounds: [], bootstrapStartStep: prev[selectedWellId] ? prev[selectedWellId].bootstrapStartStep : 0 }
                }));
            }
        } catch (error) {
            setAiFunctionError(`Lỗi khi gọi AI: ${error.message}`);
            showMessage('Lỗi', `Lỗi khi tạo hàm AI: ${error.message}`, 'error');
            console.error("Error generating AI function:", error);
            setAllWellForecasts(prev => ({
                ...prev,
                [selectedWellId]: { predictions: [], dates: [], errors: [], metrics: {}, bootstrapStartStep: prev[selectedWellId] ? prev[selectedWellId].bootstrapStartStep : 0 }
            }));
        } finally {
            setIsGeneratingAiFunction(false);
        }
    }, [selectedWellId, filteredGroundwaterData, filteredWaterQualityData, filteredWeatherForecast, filteredWaterUsage, aiPredictionFunctionBody, promptMode, userHint, selectedPredictionModel, arimaParams, gpKernelType, showMessage, setAiPredictionFunctionBody, setAiTheory, setAiNaturalLanguageExplanation, setAiIterationCount, setAiFunctionError, setAllWellForecasts, setLastValidAiModelState, isAISuggestingHint, predictionErrors, aiTheoryHistory, performanceMetricsCalculators, currentAiFunction, aiModelSpecificData]);

    // --- Check AI Prediction and Calculate Performance ---
    const handleCheckPrediction = useCallback(async () => {
        if (!selectedWellId) {
            showMessage('Cảnh báo', 'Vui lòng chọn một giếng trước khi kiểm tra dự đoán.', 'warning');
            return;
        }
        if (filteredGroundwaterData.length < 1) {
            showMessage('Thông báo', 'Không có đủ dữ liệu lịch sử cho giếng đã chọn để kiểm tra dự đoán.', 'info');
            return;
        }

        setIsGeneratingAiFunction(true); // Reusing this flag for any AI computation
        setAiFunctionError(null);

        try {
            if (!currentAiFunction) {
                throw new Error("Hàm AI không hợp lệ. Vui lòng tạo hàm AI trước.");
            }

            const executePredictionFunction = currentAiFunction; // Use memoized function

            const newPredictionErrors = [];
            const historicalActuals = [];
            const historicalPredictions = [];
            const numHistoricalPoints = filteredGroundwaterData.length;

            // Calculate errors for the last N historical days by simulating predictions
            const N_effective_points_for_error = Math.min(LEADING_PERIOD, numHistoricalPoints);
            for (let i = 0; i < N_effective_points_for_error; i++) {
                const actualDataIndex = numHistoricalPoints - N_effective_points_for_error + i;
                const actualDataPoint = filteredGroundwaterData[actualDataIndex];
                
                // Historical data available *before* this actualDataPoint for prediction
                const historicalGroundwaterSubset = filteredGroundwaterData.slice(0, actualDataIndex);
                const historicalWaterQualitySubset = filteredWaterQualityData.filter(d => new Date(d.timestamp) < new Date(actualDataPoint.timestamp));
                const historicalWeatherSubset = filteredWeatherForecast.filter(d => new Date(d.timestamp) < new Date(actualDataPoint.timestamp));
                const historicalWaterUsageSubset = filteredWaterUsage.filter(d => new Date(d.timestamp) < new Date(actualDataPoint.timestamp));

                if (historicalGroundwaterSubset.length > 0) { // Only predict if there's historical context for the AI function
                    const simulatedPredictions = executePredictionFunction(
                        historicalGroundwaterSubset,
                        historicalWaterQualitySubset,
                        historicalWeatherSubset,
                        historicalWaterUsageSubset
                    );

                    if (Array.isArray(simulatedPredictions) && simulatedPredictions.length >= 1 && typeof simulatedPredictions[0] === 'number' && !isNaN(simulatedPredictions[0])) {
                        const predictedGwlForThisDay = simulatedPredictions[0]; // Take the first day's prediction
                        const error = predictedGwlForThisDay - actualDataPoint.gwl;
                        newPredictionErrors.push({
                            timestamp: actualDataPoint.timestamp,
                            actual: actualDataPoint.gwl,
                            predicted: predictedGwlForThisDay,
                            error: error
                        });
                        historicalActuals.push(actualDataPoint.gwl);
                        historicalPredictions.push(predictedGwlForThisDay);
                    } else {
                        console.warn(`Skipping error calculation for ${actualDataPoint.timestamp} due to invalid simulation prediction (AI returned non-array or NaN).`);
                    }
                } else {
                    // If historicalGroundwaterSubset is empty, it means this is the very first data point being considered.
                    // The AI function might need some history to make a meaningful prediction.
                    console.warn(`Skipping error calculation for ${actualDataPoint.timestamp} because no historical data is available before it for prediction.`);
                }
            }
            newPredictionErrors.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

            // Calculate all performance metrics
            const currentMetrics = {};
            for (const metricKey in performanceMetricsCalculators) {
                currentMetrics[metricKey] = performanceMetricsCalculators[metricKey].calculate(historicalPredictions, historicalActuals);
            }
            
            // Update allWellForecasts with new errors and metrics
            setAllWellForecasts(prev => ({
                ...prev,
                [selectedWellId]: {
                    ...prev[selectedWellId],
                    errors: newPredictionErrors,
                    metrics: currentMetrics, // Store all calculated metrics
                    // Preserve bootstrapStartStep or set default if new well
                    bootstrapStartStep: prev[selectedWellId] ? prev[selectedWellId].bootstrapStartStep : 0
                }
            }));

            // Calculate statistical diagnostics for lastValidAiModelState
            const currentResiduals = newPredictionErrors.map(p => p.error); // Use the residuals generated by this check
            const meanRes = calculateMean(currentResiduals);
            const stdDevOfResiduals = calculateStandardDeviation(currentResiduals); // Std Dev of residuals for PI
            const skewRes = calculateSkewness(currentResiduals);
            const kurtRes = calculateKurtosis(currentResiduals);
            const acfResData = calculateACF(currentResiduals, 1); // Get ACF at lag 1 for prompt
            
            // Add console logs for debugging PI widening
            console.log("--- Debugging PI Widening ---");
            console.log("Std Dev of Residuals:", stdDevOfResiduals.toFixed(5)); // Log with more precision
            console.log("Bootstrap Start Step:", bootstrapStartStep);
            console.log("PREDICTING_PERIOD:", PREDICTING_PERIOD);


            // Generate and set the 7-day future prediction after historical error check
            let latestFuturePredictions = [];
            let latestFutureDates = [];
            let latestFutureCiBounds = []; // Initialize PI bounds here

            if (filteredGroundwaterData.length > 0) {
                latestFuturePredictions = executePredictionFunction(
                    filteredGroundwaterData, // Use full historical data for future prediction
                    filteredWaterQualityData,
                    filteredWeatherForecast,
                    filteredWaterUsage
                );

                if (Array.isArray(latestFuturePredictions) && latestFuturePredictions.length === PREDICTING_PERIOD && latestFuturePredictions.every(val => typeof val === 'number' && !isNaN(val))) {
                    const lastHistoricalTimestamp = new Date(filteredGroundwaterData[filteredGroundwaterData.length - 1].timestamp);
                    
                    // Calculate Prediction Intervals using the corrected Bootstrapping method
                    // We use currentResiduals from the historical check to form the bootstrap samples.
                    const bootstrapPIs = calculateBootstrapPI(currentResiduals, latestFuturePredictions, bootstrapStartStep, 1000, 0.95); // 1000 simulations, 95% confidence

                    for (let i = 0; i < PREDICTING_PERIOD; i++) {
                        latestFutureDates.push(formatISO(addDays(lastHistoricalTimestamp, i + 1), { representation: 'date' }));
                        
                        let lowerPI, upperPI;

                        // Hybrid approach: Use Bootstrapped PI for steps up to bootstrapStartStep,
                        // then transition to Factor-based PI for later steps, modeling increasing uncertainty.
                        // Lấy bootstrapStartStep từ state của giếng hiện tại (allWellForecasts[selectedWellId]?.bootstrapStartStep)
                        // This value is controlled by the slider, allowing dynamic hybrid PI calculation
                        const currentBootstrapStartStep = allWellForecasts[selectedWellId]?.bootstrapStartStep ?? 0;

                        if (i < currentBootstrapStartStep) {
                            // Use Bootstrapped PI if within the specified range (and available)
                            lowerPI = bootstrapPIs[i]?.lower;
                            upperPI = bootstrapPIs[i]?.upper;
                            console.log(`Ngày ${i+1} (Bootstrapping): lower=${lowerPI?.toFixed(3)}, upper=${upperPI?.toFixed(3)}`);
                        } else {
                            // Use Factor-based PI for steps beyond the bootstrap range or if bootstrap isn't used
                            const predictedGwl = latestFuturePredictions[i];
                            // The factor Math.sqrt(i + 1) makes the interval widen further into the future,
                            // reflecting higher uncertainty for longer horizons. This is a common statistical practice
                            // for factor-based PIs in time series.
                            const currentStdDevForPI = stdDevOfResiduals * Math.sqrt(i + 1);
                            upperPI = predictedGwl + (currentStdDevForPI * SIGMA_FACTOR_95_PERCENT_CI);
                            lowerPI = predictedGwl - (currentStdDevForPI * SIGMA_FACTOR_95_PERCENT_CI);
                            console.log(`Ngày ${i+1} (Factor-based): dự đoán=${predictedGwl?.toFixed(3)}, hệ số độ lệch chuẩn=${Math.sqrt(i + 1)?.toFixed(3)}, độ lệch chuẩn hiện tại cho PI=${currentStdDevForPI?.toFixed(3)}, lower=${lowerPI?.toFixed(3)}, upper=${upperPI?.toFixed(3)}`);
                        }
                        latestFutureCiBounds.push({ upper: upperPI, lower: lowerPI });
                    }
                    // Update allWellForecasts with new predictions, dates, and PI bounds
                    setAllWellForecasts(prev => ({
                        ...prev,
                        [selectedWellId]: {
                            ...prev[selectedWellId],
                            predictions: latestFuturePredictions,
                            dates: latestFutureDates,
                            futureCiBounds: latestFutureCiBounds // Store PI bounds
                        }
                    }));

                } else {
                    console.warn("Invalid 7-day future prediction generated by AI function after historical check. Clearing future predictions.");
                    setAllWellForecasts(prev => ({
                        ...prev,
                        [selectedWellId]: { ...prev[selectedWellId], predictions: [], dates: [], futureCiBounds: [] }
                    }));
                }
            } else {
                setAllWellForecasts(prev => ({
                    ...prev,
                    [selectedWellId]: { ...prev[selectedWellId], predictions: [], dates: [], futureCiBounds: [] }
                }));
            }

            // Simple summary for AI based on statistics
            let aiSummary = '';
            if (Math.abs(meanRes) < 0.1) aiSummary += 'Trung bình phần dư gần 0. ';
            if (Math.abs(skewRes) < 0.5 && Math.abs(kurtRes) < 1) aiSummary += 'Phân phối phần dư gần đối xứng và ít nhọn hơn. ';
            if (acfResData.length > 1 && Math.abs(acfResData[1].value) < acfResData[1].ciUpper) aiSummary += 'Phần dư có vẻ độc lập.';
            else if (acfResData.length > 1) aiSummary += 'Phần dư có thể có tự tương quan đáng kể.';

            const currentDiagnostics = {
                meanResidual: meanRes,
                stdDevResiduals: stdDevOfResiduals,
                skewnessResiduals: skewRes,
                kurtosisResiduals: kurtRes,
                acfResidualsData: acfResData,
                aiSummary: aiSummary
            };

            // Save the current valid state of the AI model, including all performance metrics AND diagnostics
            setLastValidAiModelState({
                functionBody: aiPredictionFunctionBody,
                theory: aiTheory,
                explanation: aiNaturalLanguageExplanation,
                arimaParams: selectedPredictionModel === 'arima' ? arimaParams : null,
                gpKernelType: selectedPredictionModel === 'gaussian_process' ? gpKernelType : null,
                performance: currentMetrics, // Save all calculated metrics
                diagnostics: currentDiagnostics // Save statistical diagnostics
            });

            setIsAiFunctionChecked(true); // Mark as checked
            showMessage('Thành công', 'Dự đoán AI đã được kiểm tra và kết quả đã sẵn sàng.', 'success');

            setAiTheoryHistory(prevHistory => {
                const newEntry = {
                    iteration: aiIterationCount,
                    modelType: selectedPredictionModel,
                    functionBody: aiPredictionFunctionBody,
                    theory: aiTheory,
                    explanation: aiNaturalLanguageExplanation,
                    sevenDayGroundwaterForecast: latestFuturePredictions, // Store the current future forecast from this check
                    metrics: currentMetrics, // Store all calculated metrics
                    diagnostics: currentDiagnostics, // Store diagnostics here too
                    arimaParams: selectedPredictionModel === 'arima' ? arimaParams : null,
                    gpKernelType: selectedPredictionModel === 'gaussian_process' ? gpKernelType : null,
                };
                
                // Find if an entry for the current iteration and modelType already exists (e.g., if handleCheckPrediction is run multiple times for the same function)
                const existingIndex = prevHistory.findIndex(entry => entry.iteration === aiIterationCount && entry.modelType === selectedPredictionModel);
                
                if (existingIndex !== -1) {
                    // Update existing entry
                    return prevHistory.map((entry, idx) => idx === existingIndex ? newEntry : entry);
                } else {
                    // Add new entry
                    return [...prevHistory, newEntry];
                }
            });

            // --- AI Suggestion for User Hint (if enabled) ---
            if (isAISuggestingHint) {
                const hintPrompt = `Các sai số trên dữ liệu dự đoán của mực nước ngầm: ${JSON.stringify(newPredictionErrors)}.
                Dữ liệu lịch sử mực nước ngầm (${TRAINING_PERIOD} bản ghi gần đây nhất) cho giếng **${selectedWellId}**: ${JSON.stringify(filteredGroundwaterData.slice(0, -LEADING_PERIOD).slice(-TRAINING_PERIOD))}.
                Dữ liệu lịch sử chất lượng nước (${TRAINING_PERIOD} bản ghi gần đây nhất) cho giếng **${selectedWellId}**: ${JSON.stringify(filteredWaterQualityData.slice(0, -LEADING_PERIOD).slice(-TRAINING_PERIOD))}.
                Dữ liệu lịch sử dự báo thời tiết (${TRAINING_PERIOD} bản ghi gần đây nhất) cho giếng **${selectedWellId}**: ${JSON.stringify(filteredWeatherForecast.slice(0, -LEADING_PERIOD).slice(-TRAINING_PERIOD))}.
                Dữ liệu lịch sử sử dụng nước (${TRAINING_PERIOD} bản ghi gần đây nhất) cho giếng **${selectedWellId}**: ${JSON.stringify(filteredWaterUsage.slice(0, -LEADING_PERIOD).slice(-TRAINING_PERIOD))}.
                
                Tiến hành hiểu sâu sắc dữ liệu đã cho và sau đó tập trung vào các giá trị cụ thể từ các dữ liệu lịch sử (đặc biệt là "Dữ liệu lịch sử mực nước ngầm"), hãy đề xuất một gợi ý ngắn gọn (2-3 câu và theo ngữ cảnh mà dữ liệu lịch sử cho phép, được hỗ trợ bởi các dữ liệu lịch sử hơn là đề cập trực tiếp/cụ thể đến các giá trị cụ thể của các sai số dự đoán), để người dùng có thể cung cấp gợi ý này một cách trực tiếp cho AI, giúp AI có một kênh thông tin để cải thiện hiệu suất của hàm dự đoán AI.
                **ĐẶC BIỆT QUAN TRỌNG**: 
                Nếu bạn phát hiện ra bất kỳ **đặc điểm hay mẫu bất thường nào chỉ xảy ra ở giếng hiện tại (${selectedWellId}) hoặc không nhất quán trên tổng thể dữ liệu (dù có đủ dữ liệu hay không), hãy cố gắng khái quát gợi ý theo hướng hỗ trợ quá trình đề xuất bằng các thông tin dạng cá biệt này.** Tức là, nếu có sự không nhất quán giữa các giếng hoặc dữ liệu giếng hiện tại có vẻ "khác biệt" so với những gì mô hình chung có thể học, hãy hướng đề xuất đến việc xem xét các yếu tố riêng biệt của giếng đó hoặc khả năng dữ liệu không phản ánh đầy đủ tác nhân trong khi tránh để KHÔNG ĐỀ CẬP TRỰC TIẾP đến các giá trị cụ thể của dữ liệu dự đoán có thể dẫn đến overfitting trên dữ liệu dự đoán. 
                Hãy đề xuất như một mô tả phải cụ thể (chứa giá trị cụ thể về dữ liệu lịch sử) nhưng đậm đặc, giàu sắc thái, bằng ngôn ngữ tự nhiên (tối ưu theo ngữ cảnh của mô hình dự đoán hiện tại là \`${selectedPredictionModel}\`) và phải khái quát được toàn bộ những điểm mấu chốt về dữ liệu lịch sử. 
                Thiết kế hiện tại là AI cần thông tin về dữ liệu để cải thiện hiệu suất của hàm dự đoán AI nhưng lại chỉ có thể tiếp cận thông tin dữ liệu lịch sử thông qua gợi ý mà bạn đề xuất. Vì vậy, hãy cụ thể ở những điểm mấu chốt và cố gắng thể hiện chúng bằng các giá trị cụ thể trong lịch sử, thay vì các chi tiết chỉ có trong dữ liệu dự báo, để loại bỏ overfitting. 
                `; 

                try {
                    const apiKey = ""; 
                    const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=${apiKey}`;
                    const response = await fetch(`${FETCH_API_URL}/api/v1/ai_fetch/raw_text`, {
                        method: 'POST',
                        headers: { 
                            'Content-Type': 'application/json',
                            'X-Client-Key': FETCH_CLIENT_KEY
                        },
                        body: JSON.stringify({
                            promptForFunction: hintPrompt
                        })
                    });
                    const rawText = await response.text();
                    let hintResult;
                    hintResult = JSON.parse(rawText);
                    if (hintResult.candidates && hintResult.candidates.length > 0 && hintResult.candidates[0].content && hintResult.candidates[0].content.parts && hintResult.candidates[0].content.parts.length > 0) {
                        setUserHint(hintResult.candidates[0].content.parts[0].text);
                        showMessage('Gợi ý AI', 'AI đã đưa ra gợi ý mới dựa trên sai số dự đoán.', 'info');
                    } else {
                        console.warn("Không nhận được gợi ý hợp lệ từ AI cho User Hint.");
                    }
                } catch (hintError) {
                    console.error("Lỗi khi AI tạo gợi ý người dùng:", hintError);
                }
            }

        } catch (error) {
            const errorMessage = `Lỗi khi thực thi hàm AI hoặc kết quả không hợp lệ: ${error.message}`;
            setAiFunctionError(errorMessage);
            console.error("Error checking prediction:", error);

            // --- NEW: AI Suggestion for User Hint based on runtime error during check ---
            if (isAISuggestingHint) {
                try {
                    const analysis = await analyzeMalformedAiResponse(error.message, `Hàm AI đang được kiểm tra:\n\`\`\`javascript\n${aiPredictionFunctionBody}\n\`\`\`\n`);
                    setUserHint(`AI đề xuất: ${analysis}`);
                    showMessage('Gợi ý AI', 'AI đã phân tích lỗi runtime và đưa ra gợi ý mới.', 'info');
                } catch (hintError) {
                    console.error("Lỗi khi AI tạo gợi ý người dùng cho lỗi runtime:", hintError);
                }
            }
            // --- END NEW ---

            showMessage('Lỗi', `Lỗi khi kiểm tra dự đoán: ${error.message}`, 'error');
        } finally {
            setIsGeneratingAiFunction(false);
        }
    }, [selectedWellId, bootstrapStartStep, filteredGroundwaterData, filteredWaterQualityData, filteredWeatherForecast, filteredWaterUsage, aiPredictionFunctionBody, isAISuggestingHint, showMessage, setUserHint, aiIterationCount, selectedPredictionModel, aiTheory, aiNaturalLanguageExplanation, arimaParams, gpKernelType, setAllWellForecasts, setAiFunctionError, setIsAiFunctionChecked, setLastValidAiModelState, setAiTheoryHistory, performanceMetricsCalculators, selectedPerformanceMetric, currentAiFunction]);

    const handleRevertToLastValidAiFunction = useCallback(() => {
        // Revert to the last valid function body for the currently selected model
        const currentModelLastValidState = aiModelSpecificData[selectedPredictionModel]?.lastValidState;
        
        if (currentModelLastValidState) {
            setAiPredictionFunctionBody(currentModelLastValidState.functionBody || DEFAULT_PREDICTION_FUNCTION_BODY);
            setAiTheory(currentModelLastValidState.theory || '');
            setAiNaturalLanguageExplanation(currentModelLastValidState.explanation || '');
            if (selectedPredictionModel === 'arima' && currentModelLastValidState.arimaParams) {
                setArimaParams(currentModelLastValidState.arimaParams);
            } else if (selectedPredictionModel === 'gaussian_process' && currentModelLastValidState.gpKernelType) {
                setGpKernelType(currentModelLastValidState.gpKernelType);
            }
            
            setAiFunctionError(null);
            setIsAiFunctionChecked(false); // Mark as unchecked to encourage re-checking
            showMessage('Thành công', `Đã hoàn tác về hàm AI hợp lệ cuối cùng cho mô hình ${selectedPredictionModel === 'general' ? 'Tổng quát' : selectedPredictionModel.toUpperCase()}.`, 'success');
        } else {
            showMessage('Thông báo', `Không có hàm AI hợp lệ nào trước đó cho mô hình ${selectedPredictionModel === 'general' ? 'Tổng quát' : selectedPredictionModel.toUpperCase()} để hoàn tác.`, 'info');
        }
    }, [aiModelSpecificData, selectedPredictionModel, showMessage, setAiPredictionFunctionBody, setAiTheory, setAiNaturalLanguageExplanation, setArimaParams, setGpKernelType, setAiFunctionError, setIsAiFunctionChecked]);


    const handleSuggestPlausibleData = useCallback(async () => {
        if (!selectedWellId) {
            showMessage('Cảnh báo', 'Vui lòng chọn một giếng để tạo dữ liệu khả tín.', 'warning');
            return;
        }
        setIsGeneratingPlausibleData(true);

        const prompt = `Bạn là một chuyên gia mô phỏng dữ liệu sensor và môi trường. Dựa trên ID giếng này: ${selectedWellId}, hãy tạo ra dữ liệu khả tín trong 30 ngày qua (tính đến ngày hiện tại) cho các loại dữ liệu sau:
        
        1.  **Dữ liệu mực nước ngầm (Groundwater Data):** gwl (m bgs), ec (µS/cm). Mực nước ngầm nên có xu hướng dao động tự nhiên, có thể giảm nhẹ hoặc tăng nhẹ. EC cũng nên có sự biến động tương ứng với GWL (ví dụ: GWL thấp hơn có thể đi kèm EC cao hơn nếu nhiễm mặn).
        2.  **Dữ liệu chất lượng nước (Water Quality Data):** ph (6.5-8.5), do (mg/L), turbidity (NTU). Giá trị nên nằm trong phạm vi hợp lý cho nước ngầm.
        3.  **Dữ liệu dự báo thời tiết (Weather Forecast):** precipitation (mm), temperature (°C). Nhiệt độ nên dao động theo mùa (nếu có thể mô phỏng), lượng mưa có thể có các đợt ngắn.
        4.  **Dữ liệu sử dụng nước (Water Usage Data):** pumping (m³/ngày), consumption (m³/ngày). Lượng bơm nên tương ứng với nhu cầu và có thể có biến động hàng ngày/tuần.

        Dữ liệu phải có định dạng JSON, là một MẢNG các đối tượng, mỗi đối tượng đại diện cho một bản ghi. Mỗi bản ghi phải có trường \`wellId\` trùng với ID giếng đã cho, và trường \`timestamp\` ở định dạng ISO 8601 (ví dụ: "2023-01-01T10:00:00Z"). Các bản ghi nên được phân bố hợp lý trong 30 ngày, có thể là hàng ngày hoặc cách ngày.

        **LƯU Ý QUAN TRỌNG:** Đảm bảo rằng \`wellId\` và \`timestamp\` là các trường riêng biệt và không được trộn lẫn giá trị. Tất cả các giá trị số cho \`gwl\`, \`ec\`, \`ph\`, \`do\`, \`turbidity\`, \`precipitation\`, và \`consumption\` phải là số hợp lệ và không được để trống.

        Trả về một đối tượng JSON duy nhất chứa các mảng dữ liệu riêng biệt cho mỗi loại:
        \`\`\`json
        {
            "groundwaterData": [],
            "waterQualityData": [],
            "weatherForecast": [],
            "waterUsage": []
        }
        \`\`\`
        `;

        const generationConfig = {
            responseMimeType: "application/json",
            responseSchema: {
                type: "OBJECT",
                properties: {
                    "groundwaterData": {
                        type: "ARRAY",
                        items: {
                            type: "OBJECT",
                            properties: {
                                "wellId": { "type": "STRING" },
                                "timestamp": { "type": "STRING" },
                                "gwl": { "type": "NUMBER" },
                                "ec": { "type": "NUMBER" }
                            }
                        }
                    },
                    "waterQualityData": {
                        type: "ARRAY",
                        items: {
                            type: "OBJECT",
                            properties: {
                                "wellId": { "type": "STRING" },
                                "timestamp": { "type": "STRING" },
                                "ph": { "type": "NUMBER" },
                                "do": { "type": "NUMBER" },
                                "turbidity": { "type": "NUMBER" }
                            }
                        }
                    },
                    "weatherForecast": {
                        type: "ARRAY",
                        items: {
                            type: "OBJECT",
                            properties: {
                                "wellId": { "type": "STRING" },
                                "timestamp": { "type": "STRING" },
                                "precipitation": { "type": "NUMBER" },
                                "temperature": { "type": "NUMBER" }
                            }
                        }
                    },
                    "waterUsage": {
                        type: "ARRAY",
                        items: {
                            type: "OBJECT",
                            properties: {
                                "wellId": { "type": "STRING" },
                                "timestamp": { "type": "STRING" },
                                "pumping": { "type": "NUMBER" },
                                "consumption": { "type": "NUMBER" }
                            }
                        }
                    }
                },
                required: ["groundwaterData", "waterQualityData", "weatherForecast", "waterUsage"]
            }
        };
        const payload = {
            contents: [{ role: "user", parts: [{ text: prompt }] }],
            generationConfig: generationConfig
        };

        try {
            const apiKey = ""; // Canvas will provide API key
            const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=${apiKey}`;
            const response = await fetch(`${FETCH_API_URL}/api/v1/ai_fetch/raw_text`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'X-Client-Key': FETCH_CLIENT_KEY
                },
                body: JSON.stringify({
                    // Tên key này phải khớp với data.get('promptForFunction') trong Python
                    promptForFunction: prompt, 
                
                    // Tên key này phải khớp với data.get('generationConfig') trong Python
                    generationConfig: generationConfig
                )
            });
            const rawText = await response.text();
            let result;
            result = JSON.parse(rawText);

            if (result.candidates && result.candidates.length > 0 && result.candidates[0].content && result.candidates[0].content.parts && result.candidates[0].content.parts.length > 0) {
                const textResponse = result.candidates[0].content.parts[0].text;
                let parsedData;
                try {
                    parsedData = JSON.parse(textResponse);
                } catch (jsonError) {
                    throw new Error(`Failed to parse AI JSON response for sustainability insights: ${jsonError.message}. Raw: ${textResponse.substring(0, 200)}...`);
                }

                // Function to merge new data with existing data, updating by wellId and timestamp
                const mergeData = (existingData, newData) => {
                    const merged = [...existingData];
                    newData.forEach(newItem => {
                        const index = merged.findIndex(item => item.wellId === newItem.wellId && item.timestamp === newItem.timestamp);
                        if (index > -1) {
                            merged[index] = { ...merged[index], ...newItem }; // Update existing
                        } else {
                            merged.push(newItem); // Add new
                        }
                    });
                    return merged.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
                };

                // Filter out data for the selected well first, then merge generated data
                const currentGroundwater = groundwaterData.filter(d => d.wellId !== selectedWellId);
                const currentWaterQuality = waterQualityData.filter(d => d.wellId !== selectedWellId);
                const currentWeather = weatherForecast.filter(d => d.wellId !== selectedWellId);
                const currentWaterUsage = waterUsage.filter(d => d.wellId !== selectedWellId);

                // Then merge the new data for the selected well
                setGroundwaterData(mergeData(currentGroundwater, parsedData.groundwaterData || []));
                setWaterQualityData(mergeData(currentWaterQuality, parsedData.waterQualityData || []));
                setWeatherForecast(mergeData(currentWeather, parsedData.weatherForecast || []));
                setWaterUsage(mergeData(currentWaterUsage, parsedData.waterUsage || []));

                showMessage('Thành công', 'Dữ liệu khả tín đã được tạo và thêm vào.', 'success');
            } else {
                showMessage('Lỗi', 'Không nhận được dữ liệu khả tín hợp lệ từ AI.', 'error');
                console.error("AI response missing candidates or content:", result);
            }
        } catch (error) {
            showMessage('Lỗi', `Lỗi khi tạo dữ liệu khả tín: ${error.message}`, 'error');
            console.error("Error generating plausible data:", error);
        } finally {
            setIsGeneratingPlausibleData(false);
        }
    }, [selectedWellId, groundwaterData, waterQualityData, weatherForecast, waterUsage, showMessage, setGroundwaterData, setWaterQualityData, setWeatherForecast, setWaterUsage, setWellLocations]);

    // Consolidated function for initial schema explanation and follow-up questions
    const handleGenerateDataSchemaExplanation = useCallback(async () => {
        setIsGeneratingDataSchemaExplanation(true);
        let currentPrompt;

        if (followUpSchemaQuestion.trim()) {
            // User has entered a follow-up question
            if (!aiDataSchemaExplanation) {
                showMessage('Thông báo', 'Vui lòng tạo giải thích lược đồ dữ liệu ban đầu trước khi đặt câu hỏi tiếp theo.', 'info');
                setIsGeneratingDataSchemaExplanation(false);
                return;
            }
            currentPrompt = `Đây là giải thích hiện tại về lược đồ dữ liệu của chúng ta:\n\n${aiDataSchemaExplanation}\n\nNgười dùng có một câu hỏi tiếp theo dựa trên điều này:\n\n"${followUpSchemaQuestion}"\n\nVui lòng cung cấp một giải thích cập nhật và chi tiết hơn, trả lời câu hỏi của người dùng và giữ lại tất cả các thông tin đã được cung cấp trước đó trong định dạng Markdown. Tập trung vào việc mở rộng và làm rõ các điểm liên quan đến câu hỏi.`;
        } else {
            // No follow-up question, generate initial explanation
            currentPrompt = `Giải thích một cách toàn diện và dễ hiểu về lược đồ dữ liệu và các liên kết ngữ nghĩa giữa các loại dữ liệu sau trong bối cảnh quản lý tài nguyên nước và dự đoán AI. Bao gồm:
            - wellLocations: { id, name, lat, lon }
            - groundwaterData: { wellId, timestamp, gwl, ec }
            - waterQualityData: { wellId, timestamp, ph, do, turbidity }
            - weatherForecast: { wellId, timestamp, precipitation, temperature }
            - waterUsage: { wellId, timestamp, pumping, consumption }

            Hãy tập trung vào cách các trường dữ liệu được liên kết (ví dụ: wellId và timestamp), ý nghĩa của từng trường và cách chúng cùng nhau tạo thành một Single Source of Truth (SSOT) cho việc dự đoán AI. Cung cấp một giải thích bằng Markdown.`;
        }

        try {
            const apiKey = ""; // Canvas will provide API key
            const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=${apiKey}`;
            const response = await fetch(`${FETCH_API_URL}/api/v1/ai_fetch/raw_text`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'X-Client-Key': FETCH_CLIENT_KEY
                },
                body: JSON.stringify({
                    // Tên key này phải khớp với data.get('promptForFunction') trong Python
                    promptForFunction: currentPrompt
                })
            });
            const rawText = await response.text();
            let result;
            result = JSON.parse(rawText);
            if (result.candidates && result.candidates.length > 0 && result.candidates[0].content && result.candidates[0].content.parts && result.candidates[0].content.parts.length > 0) {
                setAiDataSchemaExplanation(result.candidates[0].content.parts[0].text);
                setFollowUpSchemaQuestion(''); // Clear the input after asking
                showMessage('Thành công', 'Giải thích lược đồ dữ liệu đã được tạo.', 'success');
            } else {
                showMessage('Lỗi', 'Không nhận được giải thích lược đồ dữ liệu hợp lệ từ AI.', 'error');
                console.error("AI response missing candidates or content:", result);
            }
        } catch (error) {
            showMessage('Lỗi', `Lỗi khi tạo giải thích lược đồ dữ liệu: ${error.message}`, 'error');
            console.error("Error generating data schema explanation:", error);
        } finally {
            setIsGeneratingDataSchemaExplanation(false);
        }
    }, [followUpSchemaQuestion, aiDataSchemaExplanation, showMessage, setAiDataSchemaExplanation, setFollowUpSchemaQuestion]);


    const handleGenerateSustainabilityInsights = useCallback(async (
        filteredGroundwaterData,
        filteredWaterQualityData,
        filteredWaterUsage,
        isGwlCompliant, // This is now true/false/null
        isEcCompliant,  // This is now true/false/null
        isPhCompliant,  // This is now true/false/null
        sevenDayGroundwaterPrediction, // This will now be derived from allWellForecasts
        sevenDayGroundwaterPredictionDates // This will now be derived from allWellForecasts
    ) => {
        if (!selectedWellId) {
            showMessage('Cảnh báo', 'Vui lòng chọn một giếng để tạo thông tin bền vững.', 'warning');
            return;
        }
        if (filteredGroundwaterData.length === 0 && filteredWaterQualityData.length === 0 && filteredWaterUsage.length === 0) {
            showMessage('Thông báo', 'Không có đủ dữ liệu để tạo thông tin bền vững. Vui lòng nhập dữ liệu trước.', 'info');
            return;
        }

        setIsGeneratingSustainabilityInsights(true);

        // Placeholder for sustainability thresholds (these should match the values in SustainabilityComplianceTab)
        const MIN_GWL_THRESHOLD = 10; // meters below ground surface
        const MAX_EC_THRESHOLD = 1000; // µS/cm
        const MIN_PH_THRESHOLD = 6.5;
        const MAX_PH_THRESHOLD = 8.5;

        const averageGwl = filteredGroundwaterData.length > 0 ? (filteredGroundwaterData.reduce((sum, d) => sum + d.gwl, 0) / filteredGroundwaterData.length).toFixed(2) : 'N/A';
        const latestEc = filteredGroundwaterData.length > 0 ? filteredGroundwaterData[filteredGroundwaterData.length - 1]?.ec.toFixed(0) : 'N/A';
        const latestPh = filteredWaterQualityData.length > 0 ? filteredWaterQualityData[filteredWaterQualityData.length - 1]?.ph.toFixed(1) : 'N/A';
        
        // Convert boolean/null compliance status to Vietnamese string for AI
        const getComplianceStatusString = (status) => {
            if (status === null) return 'Không đủ dữ liệu';
            if (status === true) return 'ĐẠT';
            return 'KHÔNG ĐẠT';
        };

        const complianceStatus = {
            gwl: getComplianceStatusString(isGwlCompliant),
            ec: getComplianceStatusString(isEcCompliant),
            ph: getComplianceStatusString(isPhCompliant)
        };

        let forecastSection = '';
        if (sevenDayGroundwaterPrediction.length > 0 && sevenDayGroundwaterPredictionDates.length === sevenDayGroundwaterPrediction.length) {
            forecastSection = `
**Dự báo Mực nước ngầm ${PREDICTING_PERIOD} ngày tới:** ${JSON.stringify(sevenDayGroundwaterPrediction)}
**Ngày dự báo:** ${JSON.stringify(sevenDayGroundwaterPredictionDates)}
`;
        }


        const prompt = `Bạn là một chuyên gia về bền vững và môi trường, đặc biệt là trong quản lý tài nguyên nước.
        Hãy phân tích dữ liệu và trạng thái tuân thủ sau cho giếng ${selectedWellId} và tạo ra "Chi tiết Tuân thủ", "Đề xuất Tối ưu hóa" và một "Đề xuất Dashboard" ngắn gọn.
        
        **Dữ liệu hiện có:**
        - Dữ liệu mực nước ngầm: ${JSON.stringify(filteredGroundwaterData)}
        - Dữ liệu chất lượng nước: ${JSON.stringify(filteredWaterQualityData)}
        - Dữ liệu sử dụng nước: ${JSON.stringify(filteredWaterUsage)}
        ${forecastSection}
        
        **Ngưỡng tuân thủ:**
        - Mực nước ngầm (GWL): Tối thiểu ${MIN_GWL_THRESHOLD} m bgs
        - Độ dẫn điện (EC): Tối đa ${MAX_EC_THRESHOLD} µS/cm
        - Độ pH: ${MIN_PH_THRESHOLD} - ${MAX_PH_THRESHOLD}
        
        **Trạng thái tuân thủ hiện tại:**
        - GWL: ${complianceStatus.gwl}
        - EC: ${complianceStatus.ec}
        - pH: ${complianceStatus.ph}

        **Thông tin bổ sung:**
        - GWL trung bình gần nhất: ${averageGwl} m bgs
        - EC gần nhất: ${latestEc} µS/cm
        - pH gần nhất: ${latestPh}

        **Nhiệm vụ:**
        1.  **"Chi tiết Tuân thủ"**: Viết một phần giải thích chi tiết bằng Markdown, phân tích tình hình tuân thủ hiện tại. Nêu rõ các điểm đã ĐẠT, KHÔNG ĐẠT hoặc KHÔNG ĐỦ DỮ LIỆU. Nếu có vi phạm, hãy phân tích xu hướng hoặc các giá trị cụ thể gây ra vấn đề. Nếu có đủ dữ liệu, hãy phân tích mối quan hệ giữa các yếu tố (ví dụ: lượng bơm và GWL).
        2.  **"Đề xuất Tối ưu hóa"**: Viết một phần bằng Markdown, đưa ra các đề xuất cụ thể, khả thi để cải thiện các chỉ số bền vững và khắc phục các vi phạm tuân thủ (nếu có). Các đề xuất nên dựa trên dữ liệu, DỰ BÁO ${PREDICTING_PERIOD} NGÀY TỚI và lý do rõ ràng. Ví dụ: "Giảm lượng bơm thêm X% trong tháng tới", "Xem xét các phương pháp xử lý để giảm độ dẫn điện", "Tiến hành phân tích sâu hơn về nguồn gây ra pH bất thường". Nếu không đủ dữ liệu, hãy đề xuất thu thập thêm dữ liệu.
        3.  **"Đề xuất Dashboard"**: Tạo một câu ngắn gọn (tối đa 2 dòng), trực tiếp, mang tính hành động để duy trì các chỉ số bền vững trong ${PREDICTING_PERIOD} ngày tới dựa trên dự báo hiện có. Đây là đề xuất được hiển thị trên Dashboard Tổng quan. Nếu không có đủ dữ liệu để đưa ra đề xuất cụ thể, hãy đề xuất thu thập thêm dữ liệu hoặc kiểm tra dữ liệu. Ví dụ: "Giảm lượng bơm 10% trong tuần tới để duy trì mực nước ngầm trên 12m bgs." hoặc "Tăng cường theo dõi chất lượng nước do dự báo EC sẽ tăng nhẹ."

        Định dạng phản hồi của bạn phải là JSON với các trường sau:
        \`\`\`json
        {
            "details": "/* Nội dung Markdown cho Chi tiết Tuân thủ */",
            "recommendations": "/* Nội dung Markdown cho Đề xuất Tối ưu hóa */",
            "dashboardRecommendation": "/* Nội dung Markdown ngắn gọn cho Đề xuất Dashboard */"
        }
        \`\`\`
        `;

        const generationConfig = {
            responseMimeType: "application/json",
            responseSchema: {
                type: "OBJECT",
                properties: {
                    "details": { "type": "STRING" },
                    "recommendations": { "type": "STRING" },
                    "dashboardRecommendation": { "type": "STRING" } // New field for dashboard
                },
                required: ["details", "recommendations", "dashboardRecommendation"] // Now required
            }
        };
        const payload = {
            contents: [{ role: "user", parts: [{ text: prompt }] }],
            generationConfig: generationConfig
        };

        try {
            const apiKey = ""; // Canvas will provide API key
            const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=${apiKey}`;
            const response = await fetch(`${FETCH_API_URL}/api/v1/ai_fetch/raw_text`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'X-Client-Key': FETCH_CLIENT_KEY
                },
                body: JSON.stringify({
                    // Tên key này phải khớp với data.get('promptForFunction') trong Python
                    promptForFunction: prompt, 
                
                    // Tên key này phải khớp với data.get('generationConfig') trong Python
                    generationConfig: generationConfig
                })
            });
            const rawText = await response.text();
            let result;
            result = JSON.parse(rawText);

            if (result.candidates && result.candidates.length > 0 && result.candidates[0].content && result.candidates[0].content.parts && result.candidates[0].content.parts.length > 0) {
                const textResponse = result.candidates[0].content.parts[0].text;
                let parsedResponse;
                try {
                    parsedResponse = JSON.parse(textResponse);
                } catch (jsonError) {
                    throw new Error(`Failed to parse AI JSON response for sustainability insights: ${jsonError.message}. Raw: ${textResponse.substring(0, 200)}...`);
                }
                setSustainabilityInsights({
                    details: parsedResponse.details || '',
                    recommendations: parsedResponse.recommendations || ''
                });
                // Update dashboard recommendation with the concise text
                setAiDashboardRecommendation(parsedResponse.dashboardRecommendation || '');
                showMessage('Thành công', 'AI đã tạo chi tiết tuân thủ và đề xuất tối ưu hóa.', 'success');
            } else {
                showMessage('Lỗi', 'Không nhận được thông tin bền vững hợp lệ từ AI.', 'error');
                console.error("AI response missing candidates or content for sustainability insights:", result);
            }
        } catch (error) {
            showMessage('Lỗi', `Lỗi khi tạo thông tin bền vững: ${error.message}`, 'error');
            console.error("Error generating sustainability insights:", error);
        } finally {
            setIsGeneratingSustainabilityInsights(false);
        }
    }, [selectedWellId, filteredGroundwaterData, filteredWaterQualityData, filteredWaterUsage, showMessage, setSustainabilityInsights, setAiDashboardRecommendation, setIsGeneratingSustainabilityInsights, sevenDayGroundwaterPrediction, sevenDayGroundwaterPredictionDates]);


    const openImportModal = useCallback((type) => {
        setImportDataType(type);
        setJsonInput('');
        setImportErrorMessage('');
        setSelectedFileName('');
        fileInputKey.current = Date.now(); // Reset file input
        setIsImportModalOpen(true);
    }, []);

    const closeImportModal = useCallback(() => {
        setIsImportModalOpen(false);
        setImportDataType(null);
        setJsonInput('');
        setImportErrorMessage('');
        setSelectedFileName('');
    }, []);

    const handleImportJson = useCallback(async () => {
        let parsedData;
        try {
            parsedData = JSON.parse(jsonInput);
            if (!Array.isArray(parsedData)) {
                setImportErrorMessage('Dữ liệu JSON phải là một mảng.');
                return;
            }
            if (parsedData.length === 0) {
                setImportErrorMessage('Dữ liệu JSON không được rỗng.');
                return;
            }

                        // Ensure wellId and timestamp exist and are valid for all records
            const isValidData = parsedData.every(item => item.wellId && item.timestamp && new Date(item.timestamp).toString() !== 'Invalid Date');
            if (!isValidData) {
                setImportErrorMessage('Mỗi bản ghi phải có wellId và timestamp hợp lệ.');
                return;
            }

        } catch (error) {
            setImportErrorMessage(`Lỗi phân tích cú pháp JSON: ${error.message}`);
            return;
        }

        // Function to merge imported data with existing data, updating by wellId and timestamp
        const mergeImportedData = (existingData, newData) => {
            const merged = [...existingData];
            newData.forEach(newItem => {
                const index = merged.findIndex(item => item.wellId === newItem.wellId && item.timestamp === newItem.timestamp);
                if (index > -1) {
                    merged[index] = { ...merged[index], ...newItem }; // Update existing
                } else {
                    merged.push(newItem); // Add new
                }
            });
            return merged.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        };

        let updatedData = [];
        if (importDataType === 'groundwater') {
            updatedData = mergeImportedData(groundwaterData, parsedData);
            setGroundwaterData(updatedData);
        } else if (importDataType === 'waterQuality') {
            updatedData = mergeImportedData(waterQualityData, parsedData);
            setWaterQualityData(updatedData);
        } else if (importDataType === 'weather') {
            updatedData = mergeImportedData(weatherForecast, parsedData);
            setWeatherForecast(updatedData);
        } else if (importDataType === 'usage') {
            updatedData = mergeImportedData(waterUsage, parsedData);
            setWaterUsage(updatedData);
        }

        // Determine which well IDs from the imported data are new
        const newWellIdsInImportedData = new Set(parsedData.map(d => d.wellId));
        let wellsNewlyAddedCount = 0;
        let wellsAlreadyPresentCount = 0;
        let wellsToSelectAfterImport = null; // To hold the ID of a well from the imported data, if it's new or the current one

        const existingWellIds = new Set(wellLocations.map(w => w.id));

        // Create a temporary array for new well locations, to be merged once
        const tempNewWellLocations = [];

        newWellIdsInImportedData.forEach(id => {
            if (!existingWellIds.has(id)) {
                // If it's a completely new well ID, add a placeholder
                tempNewWellLocations.push({ id: id, name: `Giếng ${id}`, lat: 10.76 + Math.random() * 0.1 - 0.05, lon: 106.70 + Math.random() * 0.1 - 0.05 });
                wellsNewlyAddedCount++;
            } else {
                wellsAlreadyPresentCount++;
            }
        });

        // Update wellLocations state once with new wells
        if (tempNewWellLocations.length > 0) {
            setWellLocations(prev => [...prev, ...tempNewWellLocations]);
        }

        // Determine the message based on import result
        let successMessage = `Đã nhập dữ liệu ${importDataType}. ${parsedData.length} bản ghi đã được xử lý.`;
        let messageType = 'success';

        if (wellsNewlyAddedCount > 0) {
            successMessage += ` ${wellsNewlyAddedCount} giếng mới đã được thêm vào hệ thống (ID: ${Array.from(newWellIdsInImportedData).filter(id => !existingWellIds.has(id)).join(', ')}).`;
        }

        const importedDataWellIds = new Set(parsedData.map(d => d.wellId));
        if (selectedWellId && !importedDataWellIds.has(selectedWellId) && importedDataWellIds.size > 0) {
            // Data imported, but current well not in imported data
            successMessage += ` Lưu ý: Dữ liệu nhập không bao gồm giếng hiện đang chọn (ID: ${selectedWellId}). Vui lòng chọn giếng khác để xem dữ liệu đã nhập.`;
            messageType = 'warning';
            // Suggest selecting the first imported well, if any, to reflect changes
            if (newWellIdsInImportedData.size > 0) {
                wellsToSelectAfterImport = Array.from(newWellIdsInImportedData)[0];
            }
        } else if (importedDataWellIds.has(selectedWellId)) {
            // Data imported for the currently selected well
            successMessage += ` Dữ liệu cho giếng hiện tại (ID: ${selectedWellId}) đã được cập nhật.`;
            wellsToSelectAfterImport = selectedWellId; // Keep current well selected
        } else if (wellLocations.length === 0 && newWellIdsInImportedData.size > 0) {
             // No wells initially, but imported data has wells
             wellsToSelectAfterImport = Array.from(newWellIdsInImportedData)[0];
             successMessage += ` Một giếng mới (ID: ${wellsToSelectAfterImport}) đã được chọn để xem dữ liệu.`;
        }


        showMessage(messageType === 'success' ? 'Thành công' : 'Thông báo', successMessage, messageType);
        closeImportModal();

        // Automatically select a well from imported data if it's new, or stick to current
        if (wellsToSelectAfterImport && selectedWellId !== wellsToSelectAfterImport) {
            setSelectedWellId(wellsToSelectAfterImport);
        } else if (!selectedWellId && wellLocations.length > 0) {
            // If nothing was selected and now there are wells, select the first one
            setSelectedWellId(wellLocations[0].id);
        }


    }, [jsonInput, importDataType, groundwaterData, waterQualityData, weatherForecast, waterUsage, wellLocations, selectedWellId, showMessage, closeImportModal, setGroundwaterData, setWaterQualityData, setWeatherForecast, setWaterUsage, setWellLocations, setSelectedWellId]);


    const handleRefreshDashboard = useCallback(() => {
        // This function will re-trigger memoized selectors and potentially data fetching if it were from an external API
        // For current local/Firestore-saved data, it just forces a re-render of dependent components
        showMessage('Thông báo', 'Dashboard đã được làm mới.', 'info');
    }, [showMessage]);

    return (
        <div className="min-h-screen bg-gray-100 font-sans antialiased text-gray-900">
            <header className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white p-6 shadow-lg">
                <div className="container mx-auto flex justify-between items-center">
                    <h1 className="text-3xl font-bold">Hệ thống Quan trắc & Dự đoán Mực nước ngầm</h1>
                    <div className="flex items-center space-x-4">
                        <div className="flex flex-col gap-4">
                            <span className="text-sm">User ID: {userId || 'Đang tải...'}</span>
                            <span className="text-sm">App ID: {appId || 'Đang tải...'}</span>
                        </div>
                        <div className="relative inline-block text-left">
                            <select
                                className="block w-full py-2 px-3 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm text-gray-800"
                                value={dataStorageMode}
                                onChange={(e) => setDataStorageMode(e.target.value)}
                            >
                                <option value="local">Chế độ Tệp cục bộ</option>
                                <option value="firestore">Chế độ Đám mây (Firestore)</option>
                            </select>
                        </div>
                        <button
                            className="px-4 py-2 bg-blue-500 text-white font-bold rounded-lg shadow-md hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            onClick={handleSaveSession}
                            disabled={dataStorageMode === 'firestore' && (!db || !userId)}
                        >
                            Lưu Phiên làm việc
                        </button>
                        <button
                            className="px-4 py-2 bg-purple-500 text-white font-bold rounded-lg shadow-md hover:bg-purple-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            onClick={handleLoadSession}
                            disabled={dataStorageMode === 'firestore' && (!db || !userId)}
                        >
                            Tải Phiên làm việc
                        </button>
                    </div>
                </div>
            </header>

            <nav className="bg-white shadow-md">
                <div className="container mx-auto p-4">
                    <div className="flex space-x-4">
                        <button
                            className={`px-4 py-2 rounded-lg font-semibold transition-colors ${currentTab === 'dashboard' ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'}`}
                            onClick={() => setCurrentTab('dashboard')}
                        >
                            Dashboard
                        </button>
                        <button
                            className={`px-4 py-2 rounded-lg font-semibold transition-colors ${currentTab === 'data-management' ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'}`}
                            onClick={() => setCurrentTab('data-management')}
                        >
                            Quản lý Dữ liệu
                        </button>
                        <button
                            className={`px-4 py-2 rounded-lg font-semibold transition-colors ${currentTab === 'prediction-optimization' ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'}`}
                            onClick={() => setCurrentTab('prediction-optimization')}
                        >
                            Dự báo & Tối ưu AI
                        </button>
                        <button
                            className={`px-4 py-2 rounded-lg font-semibold transition-colors ${currentTab === 'sustainability-compliance' ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'}`}
                            onClick={() => setCurrentTab('sustainability-compliance')}
                        >
                            Bền vững & Tuân thủ
                        </button>
                        <button
                            className={`px-4 py-2 rounded-lg font-semibold transition-colors ${currentTab === 'ai-learning-insights' ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'}`}
                            onClick={() => setCurrentTab('ai-learning-insights')}
                        >
                            Học tập & Tri thức AI
                        </button>
                        <button
                            className={`px-4 py-2 rounded-lg font-semibold transition-colors ${currentTab === 'gis' ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'}`}
                            onClick={() => setCurrentTab('gis')}
                        >
                            GIS & Bản đồ
                        </button>
                        <button
                            className={`px-4 py-2 rounded-lg font-semibold transition-colors ${currentTab === 'statistical-validation' ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'}`}
                            onClick={() => setCurrentTab('statistical-validation')}
                        >
                            Kiểm định Thống kê
                        </button>
                    </div>
                </div>
            </nav>

            <main className="container mx-auto my-8 p-4 bg-white rounded-xl shadow-lg">
                {currentTab === 'dashboard' && (
                    <DashboardTab
                        filteredGroundwaterData={filteredGroundwaterData}
                        sevenDayGroundwaterPrediction={sevenDayGroundwaterPrediction}
                        // sevenDayGroundwaterPredictionDates is not used in DashboardTab directly, derived from allWellForecasts
                        predictionErrors={predictionErrors}
                        selectedPredictionModel={selectedPredictionModel}
                        handleRefreshDashboard={handleRefreshDashboard}
                        aiDashboardRecommendation={aiDashboardRecommendation}
                    />
                )}
                {currentTab === 'data-management' && (
                    <DataManagementTab
                        dataStorageMode={dataStorageMode}
                        db={db}
                        userId={userId}
                        showMessage={showMessage}
                        showConfirm={showConfirm}
                        closeConfirmModal={closeConfirmModal}
                        wellLocations={wellLocations}
                        setWellLocations={setWellLocations}
                        selectedWellId={selectedWellId}
                        setSelectedWellId={setSelectedWellId}
                        filteredGroundwaterData={filteredGroundwaterData}
                        filteredWaterQualityData={filteredWaterQualityData}
                        filteredWeatherForecast={filteredWeatherForecast}
                        filteredWaterUsage={filteredWaterUsage}
                        handleSuggestPlausibleData={handleSuggestPlausibleData}
                        isGeneratingPlausibleData={isGeneratingPlausibleData}
                        handleGenerateDataSchemaExplanation={handleGenerateDataSchemaExplanation} // Consolidated function
                        aiDataSchemaExplanation={aiDataSchemaExplanation}
                        groundwaterData={groundwaterData}
                        waterQualityData={waterQualityData}
                        weatherForecast={weatherForecast}
                        waterUsage={waterUsage}
                        openImportModal={openImportModal}
                        setGroundwaterData={setGroundwaterData}
                        setWaterQualityData={setWaterQualityData}
                        setWeatherForecast={setWeatherForecast}
                        setWaterUsage={setWaterUsage}
                        followUpSchemaQuestion={followUpSchemaQuestion}
                        setFollowUpSchemaQuestion={setFollowUpSchemaQuestion}
                        isGeneratingDataSchemaExplanation={isGeneratingDataSchemaExplanation} // Pass loading state
                        setAllWellForecasts={setAllWellForecasts}
                    />
                )}
                {currentTab === 'prediction-optimization' && (
                    <PredictionOptimizationTab
                        filteredGroundwaterData={filteredGroundwaterData}
                        predictionErrors={predictionErrors}
                        selectedPredictionModel={selectedPredictionModel}
                        setSelectedPredictionModel={setSelectedPredictionModel}
                        selectedPerformanceMetric={selectedPerformanceMetric} // Pass new prop
                        setSelectedPerformanceMetric={setSelectedPerformanceMetric} // Pass new prop
                        arimaParams={arimaParams}
                        setArimaParams={setArimaParams}
                        gpKernelType={gpKernelType}
                        setGpKernelType={setGpKernelType}
                        promptMode={promptMode}
                        setPromptMode={setPromptMode}
                        userHint={userHint}
                        setUserHint={setUserHint}
                        isAISuggestingHint={isAISuggestingHint}
                        setIsAISuggestingHint={setIsAISuggestingHint}
                        aiPredictionFunctionBody={aiPredictionFunctionBody}
                        setAiPredictionFunctionBody={setAiPredictionFunctionBody}
                        isGeneratingAiFunction={isGeneratingAiFunction}
                        aiFunctionError={aiFunctionError}
                        isAiFunctionChecked={isAiFunctionChecked}
                        handleGenerateAiFunction={handleGenerateAiFunction}
                        handleCheckPrediction={handleCheckPrediction}
                        handleRevertToLastValidAiFunction={handleRevertToLastValidAiFunction}
                        aiIterationCount={aiIterationCount}
                        aiTheoryHistory={aiTheoryHistory}
                        aiTheory={aiTheory}
                        setAiTheory={setAiTheory}
                        aiNaturalLanguageExplanation={aiNaturalLanguageExplanation}
                        setAiNaturalLanguageExplanation={setAiNaturalLanguageExplanation}
                        isAnalyzingCurrentMismatches={isAnalyzingCurrentMismatches}
                        showMessage={showMessage}
                        selectedWellId={selectedWellId}
                        filteredWaterQualityData={filteredWaterQualityData}
                        filteredWeatherForecast={filteredWeatherForecast}
                        filteredWaterUsage={filteredWaterUsage}
                        allWellForecasts={allWellForecasts} // Pass all forecasts
                        setAllWellForecasts={setAllWellForecasts} // Pass setter for all forecasts
                        aiModelSpecificData={aiModelSpecificData} // Pass aiModelSpecificData
                        setAiModelSpecificData={setAiModelSpecificData} // Pass aiModelSpecificData setter
                        setAiFunctionError = {setAiFunctionError}
                        setIsAiFunctionChecked = {setIsAiFunctionChecked}
                        db={db} // Pass db for Firestore operations
                        userId={userId} // Pass userId for Firestore operations
                        appId={appId} // Pass appId for Firestore operations
                        futureCiBounds={futureCiBounds}
                        bootstrapStartStep={bootstrapStartStep}
                        handleBootstrapStartStepChange={handleBootstrapStartStepChange}
                    />
                )}
                {currentTab === 'sustainability-compliance' && (
                    <SustainabilityComplianceTab
                        filteredGroundwaterData={filteredGroundwaterData}
                        filteredWaterQualityData={filteredWaterQualityData}
                        filteredWaterUsage={filteredWaterUsage}
                        showMessage={showMessage}
                        selectedWellId={selectedWellId}
                        onGenerateSustainabilityInsights={handleGenerateSustainabilityInsights} // Pass the function
                        sustainabilityInsights={sustainabilityInsights}
                        isGeneratingSustainabilityInsights={isGeneratingSustainabilityInsights}
                        sevenDayGroundwaterPrediction={sevenDayGroundwaterPrediction} // Derived
                        sevenDayGroundwaterPredictionDates={sevenDayGroundwaterPredictionDates} // Derived
                    />
                )}
                {currentTab === 'ai-learning-insights' && (
                    <AILearningInsightsTab
                        aiTheoryHistory={aiTheoryHistory}
                        aiTheory={aiTheory}
                        aiNaturalLanguageExplanation={aiNaturalLanguageExplanation}
                        aiPredictionFunctionBody={aiPredictionFunctionBody}
                        selectedPredictionModel={selectedPredictionModel}
                        arimaParams={arimaParams}
                        gpKernelType={gpKernelType}
                        predictionErrors={predictionErrors}
                        selectedPerformanceMetric={selectedPerformanceMetric}
                    />
                )}
                {currentTab === 'gis' && (
                    <GISTab
                        wellLocations={wellLocations}
                        groundwaterData={groundwaterData}
                        waterQualityData={waterQualityData}
                        weatherForecast={weatherForecast}
                        waterUsage={waterUsage}
                        selectedWellId={selectedWellId}
                        setSelectedWellId={setSelectedWellId}
                        showMessage={showMessage}
                    />
                )}
                {currentTab === 'statistical-validation' && (
                    <StatisticalValidationTab
                        selectedPredictionModel={selectedPredictionModel}
                        historicalPredictionResults={historicalPredictionResults}
                        residuals={residuals}
                        meanResidual={meanResidual}
                        stdDevResiduals={stdDevResiduals}
                        skewnessResiduals={skewnessResiduals}
                        kurtosisResiduals={kurtosisResiduals}
                        acfResidualsData={acfResidualsData}
                        qqPlotData={qqPlotData}
                        histogramBinsData={histogramBinsData}
                        rawGroundwaterAcfData={rawGroundwaterAcfData}
                        aiStatisticalAnalysis={aiStatisticalAnalysis}
                        isCalculatingStatistics={isCalculatingStatistics}
                        filteredGroundwaterData={filteredGroundwaterData}
                    />
                )}
            </main>

            {/* Global Modals */}
            <MessageModal
                isOpen={isMessageModalOpen}
                onClose={closeMessageModal}
                title={messageModalContent.title}
                message={messageModalContent.message}
                type={messageModalContent.type}
            />
            <ConfirmModal
                isOpen={isConfirmModalOpen}
                onClose={confirmModalContent.onCancel}
                onConfirm={confirmModalContent.onConfirm}
                title={confirmModalContent.title}
                message={confirmModalContent.message}
                type={confirmModalContent.type}
            />
            <JsonInputModal
                isOpen={isImportModalOpen}
                onClose={closeImportModal}
                onSubmit={handleImportJson}
                title={`Nhập Dữ liệu ${importDataType === 'groundwater' ? 'Nước ngầm' : importDataType === 'waterQuality' ? 'Chất lượng Nước' : importDataType === 'weather' ? 'Thời tiết' : importDataType === 'usage' ? 'Sử dụng Nước' : ''} (JSON)`}
                jsonInput={jsonInput}
                onJsonInputChange={(e) => setJsonInput(e.target.value)}
                selectedFileName={selectedFileName}
                onFileChange={handleFileChange}
                errorMessage={importErrorMessage}
                fileInputKey={fileInputKey.current}
            />
        </div>
    );
}

export default App;
