// server/routes/generation.js
const express = require('express');
const axios = require('axios');
const puppeteer = require('puppeteer');
const fs = require('fs').promises;

// ✅ 1. Revert to the simpler 'require' syntax. This now works with marked@4.3.0
const { marked } = require('marked');

const { tempAuth } = require('../middleware/authMiddleware');

const router = express.Router();

// getReportMarkdown function remains unchanged
async function getReportMarkdown(topic, apiKeys) {
    const pythonServiceUrl = process.env.PYTHON_AI_CORE_SERVICE_URL;
    if (!pythonServiceUrl) {
        throw new Error("Python AI Core Service URL is not configured in the environment.");
    }
    const reportUrl = `${pythonServiceUrl}/generate_report`;
    console.log(`Requesting Markdown report for topic: "${topic}" from ${reportUrl}`);
    const response = await axios.post(reportUrl, {
        topic: topic,
        api_keys: apiKeys
    }, { timeout: 300000 });
    if (response.data && response.data.status === 'success') {
        return response.data.report_markdown;
    } else {
        throw new Error(response.data.message || "Failed to get valid report data from AI service.");
    }
}


router.post('/report', tempAuth, async (req, res) => {
    const { topic, apiKeys } = req.body;
    if (!topic || typeof topic !== 'string' || topic.trim().length === 0) {
        return res.status(400).json({ message: "A valid 'topic' is required." });
    }
    console.log(`Received request to generate report for topic: "${topic}"`);

    try {
        const markdownContent = await getReportMarkdown(topic, apiKeys);
        if (!markdownContent) {
            return res.status(500).json({ message: "AI service failed to generate report content." });
        }
        
        // Save the raw Markdown to a file (You can remove this later)
        await fs.writeFile('debug_report.md', markdownContent);
        console.log("--- DEBUG: Saved markdown to debug_report.md ---");

        // ✅ 2. The marked.parse call is now synchronous again.
        const htmlContent = marked.parse(markdownContent);
        
        const styledHtml = `
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; padding: 30px; max-width: 800px; margin: auto; }
                    h1, h2, h3 { border-bottom: 1px solid #eee; padding-bottom: 5px; }
                    code { background-color: #f6f8fa; padding: 3px 5px; border-radius: 4px; font-family: monospace; }
                    pre { background-color: #f6f8fa; padding: 15px; border-radius: 6px; overflow-x: auto; }
                </style>
            </head>
            <body>
                ${htmlContent}
            </body>
            </html>
        `;

        await fs.writeFile('debug_report.html', styledHtml);
        console.log("--- DEBUG: Saved HTML to debug_report.html ---");

        console.log("Launching headless browser to generate PDF...");
        const browser = await puppeteer.launch({ 
            headless: true,
            args: ['--no-sandbox', '--disable-setuid-sandbox'] 
        });
        const page = await browser.newPage();
        await page.setContent(styledHtml, { waitUntil: 'networkidle0' });
        
        const pdfBuffer = await page.pdf({
            format: 'A4',
            printBackground: true,
            margin: { top: '1in', right: '1in', bottom: '1in', left: '1in' } 
        });
        
        await browser.close();
        if (!pdfBuffer || pdfBuffer.length === 0) {
            throw new Error("PDF generation resulted in an empty file.");
        }
        console.log(`PDF generated successfully. Size: ${pdfBuffer.length} bytes.`);

        const safeFilename = topic.replace(/[^a-zA-Z0-9_]/g, '_').substring(0, 50) + '_report.pdf';
        res.setHeader('Content-Type', 'application/pdf');
        res.setHeader('Content-Disposition', `attachment; filename="${safeFilename}"`);
        res.send(pdfBuffer);
        console.log(`Successfully sent "${safeFilename}" to the client.`);

    } catch (error) {
        const errorMsg = error.response?.data?.error || error.message || "An unknown error occurred.";
        console.error(`[Error] in /generate/report route for topic "${topic}": ${errorMsg}`, error);
        res.status(502).json({ message: `Failed to generate report: ${errorMsg}` });
    }
});

module.exports = router;