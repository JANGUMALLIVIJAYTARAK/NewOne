// client/src/components/ChatPage.js
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import { v4 as uuidv4 } from 'uuid';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { jsPDF } from 'jspdf';
                 
// --- Services & Configuration ---
// ✅ 1. IMPORT getUserSettings at the top of the file
import { sendMessage, saveChatHistory, getUserFiles, generateReport, getUserSettings } from '../services/api';
import { LLM_OPTIONS } from '../config/constants';
import { useTheme } from '../context/ThemeContext';

// --- Child Components ---
import SystemPromptWidget, { getPromptTextById } from './SystemPromptWidget';
import HistoryModal from './HistoryModal';
import FileUploadWidget from './FileUploadWidget';
import FileManagerWidget from './FileManagerWidget';
import AnalysisResultModal from './AnalysisResultModal';
import VoiceInputButton from './VoiceInputButton';

// --- Icons ---
import { FiFileText, FiMessageSquare, FiDatabase, FiSettings, FiLogOut, FiSun, FiMoon, FiSend, FiPlus, FiArchive, FiShield, FiDownload } from 'react-icons/fi'; // Add FiFileText

// --- Styles ---
import './ChatPage.css';

// --- UI Sub-Components (for organization) ---
const ActivityBar = ({ activeView, setActiveView }) => (
    <div className="activity-bar">
        <button className={`activity-button ${activeView === 'ASSISTANT' ? 'active' : ''}`} onClick={() => setActiveView('ASSISTANT')} title="Assistant Settings">
            <FiSettings size={24} />
        </button>
        <button className={`activity-button ${activeView === 'DATA' ? 'active' : ''}`} onClick={() => setActiveView('DATA')} title="Data Sources">
            <FiDatabase size={24} />
        </button>
    </div>
);

const AssistantSettingsPanel = (props) => (
    <div className="sidebar-panel">
        <h3 className="sidebar-header">Assistant Settings</h3>
        <SystemPromptWidget
            selectedPromptId={props.currentSystemPromptId}
            promptText={props.editableSystemPromptText}
            onSelectChange={props.handlePromptSelectChange}
            onTextChange={props.handlePromptTextChange}
        />
        <div className="llm-settings-widget">
            <h4>AI Settings</h4>
            <div className="setting-item">
                <label htmlFor="llm-provider-select">Provider:</label>
                <select id="llm-provider-select" value={props.llmProvider} onChange={props.handleLlmProviderChange} disabled={props.isProcessing}>
                    {Object.keys(LLM_OPTIONS).map(key => (
                        <option key={key} value={key}>{LLM_OPTIONS[key].name}</option>
                    ))}
                </select>
            </div>
            {LLM_OPTIONS[props.llmProvider]?.models.length > 0 && (
                <div className="setting-item">
                    <label htmlFor="llm-model-select">Model:</label>
                    <select id="llm-model-select" value={props.llmModelName} onChange={props.handleLlmModelChange} disabled={props.isProcessing}>
                        {LLM_OPTIONS[props.llmProvider].models.map(model => <option key={model} value={model}>{model}</option>)}
                        <option value="">Provider Default</option>
                    </select>
                </div>
            )}
            <div className="setting-item rag-toggle-container" title="Enable Multi-Query for RAG">
                <label htmlFor="multi-query-toggle">Multi-Query (RAG)</label>
                <input type="checkbox" id="multi-query-toggle" checked={props.enableMultiQuery} onChange={props.handleMultiQueryToggle} disabled={props.isProcessing || !props.isRagEnabled} />
            </div>
        </div>
    </div>
);

const DataSourcePanel = (props) => (
    <div className="sidebar-panel">
        <h3 className="sidebar-header">Data Sources</h3>
        <FileUploadWidget onUploadSuccess={props.triggerFileRefresh} />
        <FileManagerWidget refreshTrigger={props.refreshTrigger} onAnalysisComplete={props.onAnalysisComplete} setHasFiles={props.setHasFiles} />
    </div>
);

const Sidebar = ({ activeView, ...props }) => (
    <div className="sidebar-area">
        {activeView === 'ASSISTANT' && <AssistantSettingsPanel {...props} />}
        {activeView === 'DATA' && <DataSourcePanel {...props} />}
    </div>
);

const ThemeToggleButton = () => {
    const { theme, toggleTheme } = useTheme();
    return (
        <button onClick={toggleTheme} className="header-button theme-toggle-button" title={`Switch to ${theme === 'light' ? 'Dark' : 'Light'} Mode`}>
            {theme === 'light' ? <FiMoon size={20} /> : <FiSun size={20} />}
        </button>
    );
};


// ===================================================================================
//  Main ChatPage Component
// ===================================================================================

const ChatPage = ({ setIsAuthenticated }) => {
    // State for report modal
    const [isReportModalOpen, setIsReportModalOpen] = useState(false);
    const [reportTopic, setReportTopic] = useState('');
    const [isGeneratingReport, setIsGeneratingReport] = useState(false);

    // --- State Management ---
    const [activeView, setActiveView] = useState('ASSISTANT');
    const [messages, setMessages] = useState([]);
    const [inputText, setInputText] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [sessionId, setSessionId] = useState('');
    const [username, setUsername] = useState('');
    const [userRole, setUserRole] = useState(null);
    const [currentSystemPromptId, setCurrentSystemPromptId] = useState('friendly');
    const [editableSystemPromptText, setEditableSystemPromptText] = useState(() => getPromptTextById('friendly'));
    const [isHistoryModalOpen, setIsHistoryModalOpen] = useState(false);
    const [fileRefreshTrigger, setFileRefreshTrigger] = useState(0);
    const [hasFiles, setHasFiles] = useState(false);
    const [isRagEnabled, setIsRagEnabled] = useState(false);
    const [llmProvider, setLlmProvider] = useState('gemini');
    const [llmModelName, setLlmModelName] = useState(LLM_OPTIONS['gemini']?.models[0] || '');
    const [enableMultiQuery, setEnableMultiQuery] = useState(true);
    const [isAnalysisModalOpen, setIsAnalysisModalOpen] = useState(false);
    const [analysisModalData, setAnalysisModalData] = useState(null);
    const [activeFile, setActiveFile] = useState(localStorage.getItem('activeFile') || null);

    // --- Refs & Hooks ---
    const messagesEndRef = useRef(null);
    const navigate = useNavigate();
    const { transcript, listening, resetTranscript, browserSupportsSpeechRecognition } = useSpeechRecognition();

    useEffect(() => {
        if (listening) {
            setInputText(transcript);
        }
    }, [transcript, listening]);

    const performLogoutCleanup = useCallback(() => {
        localStorage.clear();
        setIsAuthenticated(false);
        navigate('/login', { replace: true });
    }, [setIsAuthenticated, navigate]);

    // ✅ 2. REPLACE THE ENTIRE useEffect HOOK for fetching user info with this new version
    useEffect(() => {
        const initializeApp = async () => {
            try {
                // 1. Check for basic authentication details
                const storedSessionId = localStorage.getItem('sessionId') || uuidv4();
                if (!localStorage.getItem('sessionId')) {
                    localStorage.setItem('sessionId', storedSessionId);
                }
                setSessionId(storedSessionId);
    
                const userRole = localStorage.getItem('userRole');
                const username = localStorage.getItem('username');
    
                if (!userRole || !username) {
                    // If essential user info isn't present, they are not authenticated.
                    performLogoutCleanup();
                    return;
                }
                setUserRole(userRole);
                setUsername(username);
    
                // 2. NEW ROBUST LOGIC: Always ensure API keys are loaded into storage.
                // This fixes the issue for users who were approved by an admin while they were logged out.
                const storedKeys = localStorage.getItem('userApiKeys');
                if (!storedKeys || storedKeys === '{}') {
                    console.log("API keys not found in browser storage. Attempting to fetch from server...");
                    
                    const settingsResponse = await getUserSettings();
                    const settings = settingsResponse.data;
    
                    if (settings && (settings.geminiApiKey || settings.grokApiKey)) {
                        const keysToStore = {
                            gemini: settings.geminiApiKey,
                            groq: settings.grokApiKey,
                            ollama_host: settings.ollamaHost
                        };
                        localStorage.setItem('userApiKeys', JSON.stringify(keysToStore));
                        console.log("Successfully fetched and stored API keys in localStorage.");
                    } else {
                        // This is normal for users who need to request access.
                        console.warn("User is authenticated, but no API keys are set in their account settings.");
                    }
                } else {
                    console.log("API keys found in browser storage. No fetch needed.");
                }
    
            } catch (error) {
                console.error("Error during app initialization:", error);
                setError("Could not validate user settings. Some features may be unavailable.");
            }
        };
    
        initializeApp();
    }, [performLogoutCleanup]); // Dependency is simplified as it runs once on load

    const handlePromptSelectChange = useCallback((newId) => {
        setCurrentSystemPromptId(newId);
        setEditableSystemPromptText(getPromptTextById(newId));
    }, []);
    
    const saveAndReset = useCallback(async (isLoggingOut = false, onCompleteCallback = null) => {
        const messagesToSave = messages.filter(m => m.role && m.parts);
        if (messagesToSave.length > 0) {
            setIsLoading(true);
            setError('');
            try {
                await saveChatHistory({ sessionId: localStorage.getItem('sessionId'), messages: messagesToSave });
            } catch (err) {
                setError(`Session Error: ${err.response?.data?.message || 'Failed to save session.'}`);
            }
        }
        
        const newSessionId = uuidv4();
        localStorage.setItem('sessionId', newSessionId);
        setSessionId(newSessionId);
        setMessages([]);
        if (!isLoggingOut) handlePromptSelectChange('friendly');
        setIsLoading(false);
        if (onCompleteCallback) onCompleteCallback();

    }, [messages, handlePromptSelectChange]);
    
    const handleLogout = useCallback(() => saveAndReset(true, performLogoutCleanup), [saveAndReset, performLogoutCleanup]);

    const handleSendMessage = useCallback(async (e) => {
        if (e) e.preventDefault();
        const textToSend = inputText.trim();
        if (!textToSend || isLoading) return;
        SpeechRecognition.stopListening();
        setIsLoading(true);
        setError('');
        const newUserMessage = { role: 'user', parts: [{ text: textToSend }], timestamp: new Date().toISOString() };
        
        const updatedMessages = [...messages, newUserMessage];
        setMessages(updatedMessages);

        setInputText('');
        resetTranscript();
        
        const messageData = {
            message: textToSend,
            history: updatedMessages.map(m => ({ role: m.role, parts: m.parts })),
            sessionId: localStorage.getItem('sessionId'),
            systemPrompt: editableSystemPromptText,
            isRagEnabled, llmProvider, llmModelName: llmModelName || null, enableMultiQuery,
            activeFile: activeFile || null
        };

        try {
            const response = await sendMessage(messageData);
            if (!response.data?.reply?.parts?.[0]) { throw new Error("Received an invalid response from the AI."); }
            setMessages(prev => [...prev, response.data.reply]);
        } catch (err) {
            const errorMessage = err.response?.data?.message || 'Failed to get response.';
            setError(`Chat Error: ${errorMessage}`);
            setMessages(prev => [...prev, { role: 'model', parts: [{ text: `Error: ${errorMessage}` }], isError: true, timestamp: new Date().toISOString() }]);
        } finally {
            setIsLoading(false);
        }
    }, [inputText, isLoading, messages, editableSystemPromptText, isRagEnabled, llmProvider, llmModelName, enableMultiQuery, resetTranscript, activeFile]);
    
    const triggerFileRefresh = useCallback(() => {
        // This function is called by FileUploadWidget on a successful upload.
        setFileRefreshTrigger(p => p + 1); // Refreshes the file list in FileManagerWidget.
        setIsRagEnabled(true); // Automatically enable the RAG toggle.
        setHasFiles(true); // Assume we have files now, allowing RAG to be enabled.
        // Auto-activate the most recently uploaded file
        getUserFiles().then(response => {
            const files = response.data || [];
            if (files.length > 0) {
                // Sort by lastModified or just pick the last one
                const latestFile = files.reduce((a, b) => (a.lastModified > b.lastModified ? a : b));
                setActiveFile(latestFile.relativePath);
                localStorage.setItem('activeFile', latestFile.relativePath);
            }
        });
    }, []);
    // ==================================================================
    //  END OF MODIFICATION
    // ==================================================================

    const handleNewChat = useCallback(() => { if (!isLoading) { resetTranscript(); saveAndReset(false); } }, [isLoading, saveAndReset, resetTranscript]);
    const handleEnterKey = useCallback((e) => { if (e.key === 'Enter' && !e.shiftKey && !isLoading) { e.preventDefault(); handleSendMessage(e); } }, [handleSendMessage, isLoading]);
    const handlePromptTextChange = useCallback((newText) => { setEditableSystemPromptText(newText); }, []);
    const handleLlmProviderChange = (e) => { const newProvider = e.target.value; setLlmProvider(newProvider); setLlmModelName(LLM_OPTIONS[newProvider]?.models[0] || ''); };
    const handleLlmModelChange = (e) => { setLlmModelName(e.target.value); };
    const handleRagToggle = (e) => setIsRagEnabled(e.target.checked);
    const handleMultiQueryToggle = (e) => setEnableMultiQuery(e.target.checked);
    const handleHistory = useCallback(() => setIsHistoryModalOpen(true), []);
    const closeHistoryModal = useCallback(() => setIsHistoryModalOpen(false), []);
   
    const handleSessionSelectForContinuation = useCallback((sessionData) => {
        if (sessionData && sessionData.sessionId && sessionData.messages) {
            localStorage.setItem('sessionId', sessionData.sessionId);
            setSessionId(sessionData.sessionId);
            setMessages(sessionData.messages);
            setError('');
            closeHistoryModal();
        }
    }, [closeHistoryModal]);

    const onAnalysisComplete = useCallback((data) => { setAnalysisModalData(data); setIsAnalysisModalOpen(true); }, []);
    const closeAnalysisModal = useCallback(() => { setAnalysisModalData(null); setIsAnalysisModalOpen(false); }, []);
    const handleToggleListen = () => { if (listening) { SpeechRecognition.stopListening(); } else { resetTranscript(); SpeechRecognition.startListening({ continuous: true }); } };
    
    const handleDownloadChat = useCallback(() => {
        if (messages.length === 0) return;
        const doc = new jsPDF();
        let y = 10;
        doc.setFontSize(12);
        messages.forEach((msg) => {
            const sender = msg.role === 'user' ? username || 'User' : 'Assistant';
            const text = msg.parts.map(part => part.text).join(' ');
            const lines = doc.splitTextToSize(`${sender}: ${text}`, 180);
            if (y + (lines.length * 10) > 280) {
                doc.addPage();
                y = 10;
            }
            doc.text(lines, 10, y);
            y += lines.length * 10;
        });
        doc.save('chat_history.pdf');
    }, [messages, username]);
    
    const handleGenerateReport = useCallback(async () => {
        const topic = reportTopic.trim();
        if (!topic) {
            setError("Report topic cannot be empty.");
            return;
        }

        setError(''); 
        setIsGeneratingReport(true);
        
        // ======================= THE FINAL FIX =======================
        // Replace the entire try...catch...finally block with this new version.
        try {
            const storedKeys = JSON.parse(localStorage.getItem('userApiKeys') || '{}');

            if (!storedKeys.gemini && !storedKeys.groq) {
                setError("Report Error: No API key for Gemini or Groq found in browser storage. Please check your settings.");
                setIsGeneratingReport(false);
                return;
            }

            const responseBlob = await generateReport(topic, storedKeys);

            // Create a new Blob object from the response data with the correct MIME type
            const pdfBlob = new Blob([responseBlob], { type: 'application/pdf' });

            // Create a URL for the blob
            const downloadUrl = window.URL.createObjectURL(pdfBlob);

            // Create a temporary anchor element and set its properties
            const link = document.createElement('a');
            link.href = downloadUrl;
            const safeFilename = topic.replace(/[^a-zA-Z0-9_]/g, '_').substring(0, 50) + '_report.pdf';
            link.setAttribute('download', safeFilename); // Set the filename for the download

            // Append the link to the body, click it, and then remove it
            document.body.appendChild(link);
            link.click();
            link.remove();

            // Optional: Clean up the object URL after a short delay
            setTimeout(() => window.URL.revokeObjectURL(downloadUrl), 100);

            // Reset the UI state
            setIsReportModalOpen(false);
            setReportTopic('');

        } catch (err) {
            // This improved error handling can parse errors even from blob responses
            let errorMessage = 'An unknown error occurred. Please check the server logs.';
            if (err.response && err.response.data) {
                if (err.response.data instanceof Blob && err.response.data.type === "application/json") {
                    try {
                        const errorJson = await err.response.data.text();
                        const errorObj = JSON.parse(errorJson);
                        errorMessage = errorObj.message || errorMessage;
                    } catch (parseErr) {
                        // Fallback if parsing the error blob fails
                        errorMessage = "Failed to parse error response from server.";
                    }
                } else if (err.response.data.message) {
                    errorMessage = err.response.data.message;
                }
            }
            setError(`Report Error: ${errorMessage}`);
        } finally {
            setIsGeneratingReport(false);
        }
        // =============================================================
    }, [reportTopic]); // The dependency is correct

    const handleFileSelect = useCallback((filePath) => {
        setActiveFile(filePath);
        localStorage.setItem('activeFile', filePath);
    }, []);

    const sidebarProps = {
        currentSystemPromptId, editableSystemPromptText,
        handlePromptSelectChange, handlePromptTextChange,
        llmProvider, handleLlmProviderChange,
        isProcessing: isLoading, llmModelName, handleLlmModelChange,
        enableMultiQuery, handleMultiQueryToggle, isRagEnabled,
        triggerFileRefresh, refreshTrigger: fileRefreshTrigger, onAnalysisComplete,
        setHasFiles, onFileSelect: handleFileSelect
    };

    return (
        <div className="main-layout">
            <ActivityBar activeView={activeView} setActiveView={setActiveView} />
            <Sidebar activeView={activeView} {...sidebarProps} activeFile={activeFile} />
            <div className="chat-view">
                <header className="chat-header">
                    <h1>FusedChat</h1>
                    <div className="header-controls">
                        <span className="username-display">Hi, {username}</span>
                        <ThemeToggleButton />
                        <button onClick={() => setIsReportModalOpen(true)} className="header-button" title="Generate Report" disabled={isLoading || isGeneratingReport}>
                            <FiFileText size={20} />
                        </button>
                        <button onClick={handleHistory} className="header-button" title="Chat History" disabled={isLoading}><FiArchive size={20} /></button>
                        <button onClick={handleDownloadChat} className="header-button" title="Download Chat" disabled={messages.length === 0}><FiDownload size={20} /></button>
                        <button onClick={handleNewChat} className="header-button" title="New Chat" disabled={isLoading}><FiPlus size={20} /></button>
                        <button onClick={() => navigate('/settings')} className="header-button" title="Settings" disabled={isLoading}><FiSettings size={20} /></button>
                        {userRole === 'admin' && (
                            <button onClick={() => navigate('/admin')} className="header-button admin-button" title="Admin Panel">
                                <FiShield size={20} />
                            </button>
                        )}
                        <button onClick={handleLogout} className="header-button" title="Logout" disabled={isLoading}><FiLogOut size={20} /></button>
                    </div>
                </header>
                <main className="messages-area" ref={messagesEndRef}>
                    {messages.length === 0 && !isLoading && (
                         <div className="welcome-screen">
                            <FiMessageSquare size={48} className="welcome-icon" />
                            <h2>Start a conversation</h2>
                            <p>Ask a question, upload a document, or select a model to begin.</p>
                         </div>
                    )}
                    {messages.map((msg, index) => (
                        <div key={`${sessionId}-${index}`} className={`message ${msg.role.toLowerCase()}${msg.isError ? '-error-message' : ''}`}>
                            <div className="message-content-wrapper">
                                <p className="message-sender-name">{msg.role === 'user' ? username : 'Assistant'}</p>
                                <div className="message-text"><ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.parts[0].text}</ReactMarkdown></div>
                                
                                {msg.thinking && <details className="message-thinking-trace"><summary>Thinking Process</summary><pre>{msg.thinking}</pre></details>}
                                
                                {msg.role === 'model' && msg.provider && (
                                    <div className="message-metadata">
                                        <span>Provider: {msg.provider} | Model: {msg.model || 'Default'}</span>
                                    </div>
                                )}

                                {msg.references?.length > 0 && <div className="message-references"><strong>References:</strong><ul>{msg.references.map((ref, i) => <li key={i} title={ref.preview_snippet}>{ref.documentName} (Score: {ref.score?.toFixed(2)})</li>)}</ul></div>}
                            </div>
                        </div>
                    ))}
                </main>
                <div className="indicator-container">
                    {isLoading && <div className="loading-indicator"><span>Thinking...</span></div>}
                    {!isLoading && error && <div className="error-indicator">{error}</div>}
                </div>
                {inputText.match(/pdf|topics|headings|subheadings/i) && !activeFile && (
                    <div className="fm-error" style={{margin:'10px',textAlign:'center'}}>Please activate a file in the file manager to ask questions about a PDF.</div>
                )}
                <footer className="input-area">
                    <textarea value={inputText} onChange={(e) => setInputText(e.target.value)} onKeyDown={handleEnterKey} placeholder="Type or say something..." rows="1" disabled={isLoading} />
                    <VoiceInputButton isListening={listening} onToggleListen={handleToggleListen} isSupported={browserSupportsSpeechRecognition} />
                    <div className="rag-toggle-container" title={!hasFiles ? "Upload files to enable RAG" : "Toggle RAG"}>
                        <label htmlFor="rag-toggle">RAG</label>
                        <input type="checkbox" id="rag-toggle" checked={isRagEnabled} onChange={handleRagToggle} disabled={!hasFiles || isLoading} />
                    </div>
                    <button onClick={handleSendMessage} disabled={isLoading || !inputText.trim()} title="Send Message" className="send-button">
                        <FiSend size={20} />
                    </button>
                </footer>
            </div>
            <HistoryModal isOpen={isHistoryModalOpen} onClose={closeHistoryModal} onSessionSelect={handleSessionSelectForContinuation} />
            {analysisModalData && <AnalysisResultModal isOpen={isAnalysisModalOpen} onClose={closeAnalysisModal} analysisData={analysisModalData} />}

            {isReportModalOpen && (
                <div className="report-modal-overlay">
                    <div className="report-modal-content">
                        <h3>Generate New Report</h3>
                        <p>Enter a topic, and the AI will perform a web search to generate a structured PDF report.</p>
                        <input
                            type="text"
                            value={reportTopic}
                            onChange={(e) => setReportTopic(e.target.value)}
                            placeholder="e.g., The Future of Quantum Computing"
                            disabled={isGeneratingReport}
                            autoFocus
                            onKeyDown={(e) => e.key === 'Enter' && handleGenerateReport()}
                        />
                        <div className="report-modal-actions">
                            <button onClick={() => setIsReportModalOpen(false)} disabled={isGeneratingReport} className="secondary-button">
                                Cancel
                            </button>
                            <button onClick={handleGenerateReport} disabled={isGeneratingReport || !reportTopic.trim()} className="primary-button">
                                {isGeneratingReport ? 'Generating...' : 'Generate PDF'}
                            </button>
                        </div>
                        {isGeneratingReport && <div className="loading-indicator" style={{ marginTop: '10px' }}><span>Gathering sources and writing report... This may take a few minutes.</span></div>}
                        {error && <div className="error-indicator" style={{ marginTop: '10px' }}>{error}</div>}
                    </div>
                </div>
            )}
        </div>
    );
};

export default ChatPage;