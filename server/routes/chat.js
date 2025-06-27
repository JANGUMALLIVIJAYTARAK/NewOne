/**
 * @fileoverview Express router for chat-related functionalities.
 */

const express = require('express');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');
const { tempAuth } = require('../middleware/authMiddleware');
const ChatHistory = require('../models/ChatHistory');
const User = require('../models/User');
const { decrypt } = require('../services/encryptionService');

const router = express.Router();

// --- Configuration & Constants ---
const PYTHON_AI_SERVICE_URL = process.env.PYTHON_AI_CORE_SERVICE_URL;
if (!PYTHON_AI_SERVICE_URL) {
    console.error("FATAL ERROR: PYTHON_AI_CORE_SERVICE_URL is not set.");
}
const KNOWLEDGE_CHECK_IDENTIFIER = "You are a Socratic quizmaster";

// --- Helper Functions ---
const getApiAuthDetails = async (userId, selectedLlmProvider) => {
    // This function is correct.
    if (!selectedLlmProvider) throw new Error("LLM Provider is required.");
    const user = await User.findById(userId).select('+geminiApiKey +grokApiKey +apiKeyAccessRequest +ollamaHost');
    if (!user) throw new Error("User account not found.");
    const apiKeys = { gemini: null, grok: null };
    if (user.apiKeyAccessRequest?.status === 'approved') {
        apiKeys.gemini = process.env.ADMIN_GEMINI_API_KEY;
        apiKeys.grok = process.env.ADMIN_GROQ_API_KEY;
    } else {
        if (user.geminiApiKey) apiKeys.gemini = decrypt(user.geminiApiKey);
        if (user.grokApiKey) apiKeys.grok = decrypt(user.grokApiKey);
    }
    if (selectedLlmProvider.startsWith('gemini') && !apiKeys.gemini) throw new Error("A required Gemini API key was not available.");
    if (selectedLlmProvider.startsWith('groq') && !apiKeys.grok) throw new Error("A required Groq API key was not available.");
    const ollamaHost = user.ollamaHost || null;
    return { apiKeys, ollamaHost };
};

// --- Routes ---
router.post('/message', tempAuth, async (req, res) => {
    // ==================================================================
    //  DEFINITIVE FIX: Add logging to see the incoming request data
    // ==================================================================
    console.log("--- Received /message request from frontend with body: ---");
    console.log(req.body);
    console.log("----------------------------------------------------------");
    // ==================================================================

    const {
        message, history, sessionId, systemPrompt, isRagEnabled,
        llmProvider, llmModelName, enableMultiQuery, activeFile
    } = req.body;
    const userId = req.user._id.toString();

    if (!message || !sessionId || !llmProvider) {
        return res.status(400).json({ message: 'Bad request: message, sessionId, and llmProvider are required.' });
    }

    if (!PYTHON_AI_SERVICE_URL) {
        return res.status(503).json({ message: "AI Service is temporarily unavailable." });
    }

    try {
        const { apiKeys, ollamaHost } = await getApiAuthDetails(userId, llmProvider);
        const isKnowledgeCheck = systemPrompt?.includes(KNOWLEDGE_CHECK_IDENTIFIER) && history?.length === 0;
        const performRagRequest = !isKnowledgeCheck && !!isRagEnabled;

        const pythonPayload = {
            user_id: userId,
            query: message.trim(),
            chat_history: history || [],
            llm_provider: llmProvider,
            llm_model_name: llmModelName || null,
            system_prompt: systemPrompt,
            perform_rag: performRagRequest,
            enable_multi_query: enableMultiQuery ?? true,
            api_keys: apiKeys,
            ollama_host: ollamaHost,
            active_file: activeFile || null
        };

        const pythonResponse = await axios.post(`${PYTHON_AI_SERVICE_URL}/generate_chat_response`, pythonPayload, { timeout: 120000 });

        if (pythonResponse.data?.status !== 'success') {
            throw new Error(pythonResponse.data?.error || "Failed to get valid response from AI service.");
        }

        const modelResponseMessage = {
            role: 'model',
            parts: [{ text: pythonResponse.data.llm_response || "[No response text from AI]" }],
            timestamp: new Date(),
            references: pythonResponse.data.references || [],
            thinking: pythonResponse.data.thinking_content || null,
            provider: pythonResponse.data.provider_used,
            model: pythonResponse.data.model_used,
            context_source: pythonResponse.data.context_source
        };

        res.status(200).json({ reply: modelResponseMessage });

    } catch (error) {
        console.error(`!!! Error in /message route for session ${sessionId}:`, error.message);
        const status = error.response?.status || 500;
        const message = error.response?.data?.error || error.message || "An unexpected server error occurred.";
        res.status(status).json({ message });
    }
});

// The /history, /sessions, and other routes are correct and remain unchanged.
// ...
router.post('/history', tempAuth, async (req, res) => {
    // ... (code is unchanged) ...
    const { sessionId, messages } = req.body;
    const userId = req.user._id;

    if (!sessionId) return res.status(400).json({ message: 'Session ID required.' });
    if (!Array.isArray(messages)) return res.status(400).json({ message: 'Invalid messages format.' });

    try {
        const validMessages = messages
            .filter(m => m && m.role && m.parts?.[0]?.text && m.timestamp)
            .map(m => ({
                role: m.role,
                parts: m.parts,
                timestamp: m.timestamp,
                references: m.role === 'model' ? (m.references || []) : undefined,
                thinking: m.role === 'model' ? (m.thinking || null) : undefined,
            }));

        const newSessionId = uuidv4();
        if (validMessages.length === 0) {
            return res.status(200).json({ message: 'No history to save.', savedSessionId: null, newSessionId });
        }

        const savedHistory = await ChatHistory.findOneAndUpdate(
            { sessionId: sessionId, userId: userId },
            { $set: { userId, sessionId, messages: validMessages, updatedAt: Date.now() } },
            { new: true, upsert: true, setDefaultsOnInsert: true }
        );

        res.status(200).json({ message: 'Chat history saved.', savedSessionId: savedHistory.sessionId, newSessionId });
    } catch (error) {
        console.error(`Error saving chat history for session ${sessionId}:`, error);
        res.status(500).json({ message: 'Failed to save chat history.' });
    }
});

router.get('/sessions', tempAuth, async (req, res) => {
    // ... (code is unchanged) ...
    const userId = req.user._id;
    try {
        const sessions = await ChatHistory.find({ userId })
            .sort({ updatedAt: -1 })
            .select('sessionId createdAt updatedAt messages')
            .lean();

        const sessionSummaries = sessions.map(session => {
            const firstUserMessage = session.messages?.find(m => m.role === 'user');
            let preview = firstUserMessage?.parts?.[0]?.text.substring(0, 75) || 'Chat Session';
            if (preview.length === 75) preview += '...';

            return {
                sessionId: session.sessionId,
                createdAt: session.createdAt,
                updatedAt: session.updatedAt,
                messageCount: session.messages?.length || 0,
                preview: preview,
            };
        });
        res.status(200).json(sessionSummaries);
    } catch (error) {
        console.error(`Error fetching sessions for user ${userId}:`, error);
        res.status(500).json({ message: 'Failed to retrieve sessions.' });
    }
});

router.get('/session/:sessionId', tempAuth, async (req, res) => {
    // ... (code is unchanged) ...
    const { sessionId } = req.params;
    const userId = req.user._id;

    if (!sessionId) return res.status(400).json({ message: 'Session ID is required.' });

    try {
        const session = await ChatHistory.findOne({ sessionId, userId }).lean();
        if (!session) {
            return res.status(404).json({ message: 'Chat session not found or access denied.' });
        }
        res.status(200).json(session);
    } catch (error) {
        console.error(`Error fetching session ${sessionId} for user ${userId}:`, error);
        res.status(500).json({ message: 'Failed to retrieve session.' });
    }
});

router.delete('/session/:sessionId', tempAuth, async (req, res) => {
    // ... (code is unchanged) ...
    const { sessionId } = req.params;
    const userId = req.user._id;

    if (!sessionId) return res.status(400).json({ message: 'Session ID is required to delete.' });

    try {
        console.log(`>>> DELETE /api/chat/session/${sessionId} requested by User ${userId}`);
        const result = await ChatHistory.findOneAndDelete({ sessionId, userId });

        if (!result) {
            console.warn(`   Session not found or user mismatch for session ${sessionId} and user ${userId}.`);
            return res.status(404).json({ message: 'Session not found or you do not have permission to delete it.' });
        }

        console.log(`<<< Session ${sessionId} successfully deleted for user ${userId}.`);
        res.status(200).json({ message: 'Session deleted successfully.' });
    } catch (error) {
        console.error(`!!! Error deleting session ${sessionId} for user ${userId}:`, error);
        res.status(500).json({ message: 'Failed to delete session due to a server error.' });
    }
});


module.exports = router;