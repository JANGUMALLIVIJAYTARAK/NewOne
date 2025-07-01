// client/src/services/api.js
import axios from 'axios';

// Dynamically determine API Base URL
const getApiBaseUrl = () => {
    const backendPort = process.env.REACT_APP_BACKEND_PORT || 5003;
    const hostname = window.location.hostname;
    // Use the same protocol as the frontend is currently using
    const protocol = window.location.protocol; 
    const backendHost = (hostname === 'localhost' || hostname === '127.0.0.1') ? 'localhost' : hostname;
    return `${protocol}//${backendHost}:${backendPort}/api`;
};

const API_BASE_URL = getApiBaseUrl();

// ✅ THIS LINE WAS MISSING. It creates the axios instance that all other functions use.
const apiClient = axios.create({ 
    baseURL: API_BASE_URL 
});

// --- Interceptors ---
// This interceptor attaches the user ID to every outgoing request
apiClient.interceptors.request.use(
    (config) => {
        const userId = localStorage.getItem('userId');
        if (userId) {
            config.headers['x-user-id'] = userId;
        } else if (!config.url.includes('/auth/')) {
            console.warn("API Interceptor: userId not found for non-auth request to", config.url);
        }
        
        // Ensure Content-Type is set correctly, but not for file uploads
        if (!(config.data instanceof FormData)) {
            config.headers['Content-Type'] = 'application/json';
        }
        return config;
    },
    (error) => Promise.reject(error)
);

// This interceptor handles 401 Unauthorized errors by logging the user out
apiClient.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response && error.response.status === 401) {
            console.warn("API Interceptor: 401 Unauthorized. Clearing auth & redirecting.");
            localStorage.clear();
            if (!window.location.pathname.includes('/login')) {
                window.location.href = '/login?sessionExpired=true';
            }
        }
        return Promise.reject(error);
    }
);


// --- AUTHENTICATION ---
export const signupUser = (userData) => apiClient.post('/auth/signup', userData);
export const signinUser = (userData) => apiClient.post('/auth/signin', userData);
export const requestAdminKeyAccess = () => apiClient.post('/auth/request-access');


// --- USER SETTINGS ---
export const getUserSettings = () => apiClient.get('/settings');
export const saveUserSettings = (settingsData) => apiClient.post('/settings', settingsData);


// --- ADMIN PANEL ---
export const getAdminAccessRequests = () => apiClient.get('/admin/requests');
export const processAdminRequest = (userId, isApproved) => apiClient.post('/admin/approve', { userId, isApproved });
export const getAcceptedUsers = () => apiClient.get('/admin/accepted');


// --- CHAT & HISTORY ---
export const sendMessage = (messageData) => apiClient.post('/chat/message', messageData);
export const saveChatHistory = (historyData) => apiClient.post('/chat/history', historyData);
export const getChatSessions = () => apiClient.get('/chat/sessions');
export const getSessionDetails = (sessionId) => apiClient.get(`/chat/session/${sessionId}`);
export const deleteChatSession = (sessionId) => apiClient.delete(`/chat/session/${sessionId}`);


// --- FILE UPLOAD & MANAGEMENT ---
export const uploadFile = (formData) => apiClient.post('/upload', formData);
export const getUserFiles = () => apiClient.get('/files');
export const renameUserFile = (serverFilename, newOriginalName) => apiClient.patch(`/files/${serverFilename}`, { newOriginalName });
export const deleteUserFile = (serverFilename) => apiClient.delete(`/files/${serverFilename}`);


// --- DOCUMENT ANALYSIS ---
export const analyzeDocument = (analysisData) => apiClient.post('/analysis/document', analysisData);


// --- CONTENT GENERATION ---
// ✅ CHANGE THE FUNCTION SIGNATURE AND THE POST BODY
export const generateReport = (topic, apiKeys) => {
    return apiClient.post('/generation/report', { topic, apiKeys }, {
        responseType: 'blob',
    });
};


// --- DEFAULT EXPORT ---
// Export the configured instance for use in other parts of the app
export default apiClient;