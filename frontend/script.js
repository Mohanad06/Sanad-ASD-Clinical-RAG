const API_BASE = "http://localhost:8000/api";

// Local state
let conversations = [];
let activeConversationId = null;
let selectedFile = null;

document.addEventListener("DOMContentLoaded", async () => {
    // 1. Fetch system metrics dynamically
    await loadMetrics();

    // 2. Load conversations from localStorage
    loadSessionsFromStorage();

    // 3. Render sidebar history
    renderSidebar();

    // 4. Initialize active chat
    if (activeConversationId) {
        renderActiveChat();
    } else {
        renderWelcomeScreen();
    }

    // 5. Setup Action Listeners
    document.getElementById("new-chat-btn").addEventListener("click", startNewChat);
    document.getElementById("submit-btn").addEventListener("click", submitQuery);
    
    document.getElementById("query").addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submitQuery();
        }
    });

    // File upload listeners
    const attachBtn = document.getElementById("attach-btn");
    const fileInput = document.getElementById("file-input");

    attachBtn.addEventListener("click", () => {
        const activeConv = conversations.find(c => c.id === activeConversationId);
        if (activeConv && activeConv.documentId) {
            alert("This conversation already has an attached document. Start a new chat to attach a different document.");
            return;
        }
        fileInput.click();
    });

    fileInput.addEventListener("change", (e) => {
        const activeConv = conversations.find(c => c.id === activeConversationId);
        if (activeConv && activeConv.documentId) {
            alert("This conversation already has an attached document. Start a new chat to attach a different document.");
            fileInput.value = "";
            return;
        }

        const file = e.target.files[0];
        if (!file) return;

        // Validate extension
        const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
        if (ext !== '.pdf' && ext !== '.docx') {
            alert(`Unsupported file format '${ext}'. Only .pdf and .docx are supported.`);
            fileInput.value = "";
            return;
        }

        // Validate size (10 MB)
        const maxSize = 10 * 1024 * 1024;
        if (file.size > maxSize) {
            alert(`File size exceeds limit of 10MB (actual: ${(file.size / (1024*1024)).toFixed(2)}MB).`);
            fileInput.value = "";
            return;
        }

        selectedFile = file;
        renderAttachmentPreview();
    });

    // Mobile Hamburger Menu Listeners
    const mobileMenuBtn = document.getElementById("mobile-menu-toggle");
    const sidebar = document.getElementById("sidebar");
    const overlay = document.getElementById("sidebar-overlay");

    mobileMenuBtn.addEventListener("click", () => {
        sidebar.classList.add("open");
        overlay.classList.remove("hidden");
    });

    overlay.addEventListener("click", () => {
        sidebar.classList.remove("open");
        overlay.classList.add("hidden");
    });

    // Desktop Collapse/Expand Sidebar Listeners
    const sidebarToggleBtn = document.getElementById("sidebar-toggle");
    const logoBtn = document.getElementById("sidebar-logo-button");
    if (sidebarToggleBtn) {
        // Helper function to update toggle UI
        const updateToggleUI = (isCollapsed) => {
            if (isCollapsed) {
                sidebar.classList.add("collapsed");
                sidebarToggleBtn.setAttribute("aria-label", "Expand sidebar");
                sidebarToggleBtn.setAttribute("title", "Expand sidebar");
                sidebarToggleBtn.innerHTML = `<i data-lucide="panel-left-open"></i>`;
                
                if (logoBtn) {
                    logoBtn.setAttribute("tabindex", "0");
                    logoBtn.setAttribute("aria-label", "Open sidebar");
                    logoBtn.setAttribute("title", "Open sidebar");
                }
            } else {
                sidebar.classList.remove("collapsed");
                sidebarToggleBtn.setAttribute("aria-label", "Collapse sidebar");
                sidebarToggleBtn.setAttribute("title", "Collapse sidebar");
                sidebarToggleBtn.innerHTML = `<i data-lucide="panel-left-close"></i>`;
                
                if (logoBtn) {
                    logoBtn.setAttribute("tabindex", "-1");
                    logoBtn.setAttribute("aria-label", "Clinical ASD RAG logo");
                    logoBtn.removeAttribute("title");
                }
            }
            createIconsSafe();
        };

        // Load persisted state
        const isCollapsed = localStorage.getItem("clinicalSidebarCollapsed") === "true";
        updateToggleUI(isCollapsed);

        sidebarToggleBtn.addEventListener("click", () => {
            const currentlyCollapsed = sidebar.classList.contains("collapsed");
            const nextCollapsed = !currentlyCollapsed;
            localStorage.setItem("clinicalSidebarCollapsed", nextCollapsed ? "true" : "false");
            updateToggleUI(nextCollapsed);
        });

        if (logoBtn) {
            logoBtn.addEventListener("click", () => {
                if (sidebar.classList.contains("collapsed")) {
                    localStorage.setItem("clinicalSidebarCollapsed", "false");
                    updateToggleUI(false);
                }
            });
        }
    }

    // Initial Lucide icons create
    createIconsSafe();
});

// Create Lucide icons safely without throwing exceptions
function createIconsSafe() {
    if (window.lucide) {
        window.lucide.createIcons();
    }
}

// Fetch evaluation summary metrics from API
async function loadMetrics() {
    try {
        const res = await fetch(`${API_BASE}/evaluation/summary`);
        if (res.ok) {
            const data = await res.json();
            const metrics = data.metrics;
            
            const p5El = document.getElementById("metric-p5");
            const safetyEl = document.getElementById("metric-safety");
            const citationEl = document.getElementById("metric-citation");
            
            if (p5El) p5El.textContent = `${(metrics.precision_at_5 * 100).toFixed(0)}%`;
            if (safetyEl) safetyEl.textContent = `${(metrics.safety_pass_rate * 100).toFixed(0)}%`;
            if (citationEl) citationEl.textContent = `${(metrics.citation_validity * 100).toFixed(0)}%`;
        }
    } catch (err) {
        console.error("Failed to load evaluation metrics:", err);
    }
}

// Load sessions from local storage
function loadSessionsFromStorage() {
    const stored = localStorage.getItem("asd_rag_conversations");
    const active = localStorage.getItem("asd_rag_active_conv");
    
    if (stored) {
        try {
            conversations = JSON.parse(stored);
        } catch (e) {
            conversations = [];
        }
    }
    
    if (active) {
        activeConversationId = active;
        // Verify active conversation exists
        if (!conversations.some(c => c.id === activeConversationId)) {
            activeConversationId = null;
        }
    }
}

// Save sessions to local storage
function saveSessionsToStorage() {
    localStorage.setItem("asd_rag_conversations", JSON.stringify(conversations));
    localStorage.setItem("asd_rag_active_conv", activeConversationId || "");
}

// Start a fresh conversation session
function startNewChat() {
    selectedFile = null;
    const fileInput = document.getElementById("file-input");
    if (fileInput) fileInput.value = "";
    renderAttachmentPreview();

    activeConversationId = "conv_" + Date.now();
    const newConv = {
        id: activeConversationId,
        title: "New Chat",
        createdTime: Date.now(),
        messages: [],
        documentId: null,
        documentName: null
    };
    conversations.unshift(newConv);
    saveSessionsToStorage();

    // Render updates
    renderSidebar();
    renderActiveChat();
    
    // Close sidebar drawer on mobile if open
    document.getElementById("sidebar").classList.remove("open");
    document.getElementById("sidebar-overlay").classList.add("hidden");

    document.getElementById("query").focus();
}

// Switch active conversation
function selectConversation(id) {
    selectedFile = null;
    const fileInput = document.getElementById("file-input");
    if (fileInput) fileInput.value = "";
    renderAttachmentPreview();

    activeConversationId = id;
    saveSessionsToStorage();
    renderSidebar();
    renderActiveChat();

    // Close mobile side panel
    document.getElementById("sidebar").classList.remove("open");
    document.getElementById("sidebar-overlay").classList.add("hidden");
}

// Render the sidebar history list chronologically
function renderSidebar() {
    const todayList = document.getElementById("history-today");
    const prevList = document.getElementById("history-previous");

    todayList.innerHTML = "";
    prevList.innerHTML = "";

    const todayStart = new Date().setHours(0, 0, 0, 0);

    conversations.forEach(conv => {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.className = `history-item ${conv.id === activeConversationId ? 'active' : ''}`;
        a.href = "#";
        a.innerHTML = `<i data-lucide="message-square" class="icon-sm"></i> ${conv.title}`;
        
        a.addEventListener("click", (e) => {
            e.preventDefault();
            selectConversation(conv.id);
        });

        li.appendChild(a);

        if (conv.createdTime >= todayStart) {
            todayList.appendChild(li);
        } else {
            prevList.appendChild(li);
        }
    });

    createIconsSafe();
}

// Render welcome screen when no active chat
function renderWelcomeScreen() {
    const container = document.getElementById("chat-messages");
    container.innerHTML = `
        <div class="welcome-container">
            <div class="welcome-logo">
                <i data-lucide="shield-check" style="width:36px; height:36px;"></i>
            </div>
            <h2 class="welcome-title">Clinical ASD Decision Support</h2>
            <p class="welcome-subtitle">
                Access safe, citation-bound guidelines for Autism Spectrum Disorder (ASD). Start a new chat session to ask questions about diagnostic criteria, therapies, and clinical assessments.
            </p>
        </div>
    `;
    createIconsSafe();
}

// Render active conversation history
function renderActiveChat() {
    const container = document.getElementById("chat-messages");
    container.innerHTML = "";

    const activeConv = conversations.find(c => c.id === activeConversationId);
    if (!activeConv || activeConv.messages.length === 0) {
        // Render simple start state
        container.innerHTML = `
            <div class="welcome-container">
                <h2 class="welcome-title">New Conversation</h2>
                <p class="welcome-subtitle">Ask any question to begin retrieving validated clinical evidence.</p>
            </div>
        `;
        return;
    }

    activeConv.messages.forEach(msg => {
        if (msg.role === "user") {
            const row = document.createElement("div");
            row.className = "message-user";
            let attachmentHtml = "";
            if (msg.attachedFile) {
                attachmentHtml = `
                    <div class="message-user-attachment">
                        <i data-lucide="file-text" class="icon-xs"></i>
                        <span>${escapeHtml(msg.attachedFile)}</span>
                    </div>
                `;
            }
            row.innerHTML = `<div class="message-user-bubble">${escapeHtml(msg.content)}${attachmentHtml}</div>`;
            container.appendChild(row);
        } else {
            const row = document.createElement("div");
            row.className = "message-assistant";
            row.appendChild(createAssistantCard(msg.responseObj));
            container.appendChild(row);
        }
    });

    createIconsSafe();
    // Scroll container to bottom
    const chatContainer = document.querySelector(".chat-container");
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Create assistant card based on status code
function createAssistantCard(data) {
    const card = document.createElement("div");
    card.className = "message-assistant-card";

    const status = (data.status || "answered").toLowerCase();
    const confidence = (data.confidence || "low").toUpperCase();
    const recommendation = data.recommendation || "No recommendation provided.";
    const evidenceItems = data.supporting_evidence || [];
    const missingItems = data.missing_information || [];
    const safetyNote = data.safety_note || "Educational information only; not a diagnosis or medical advice.";

    // Class for state badge styling
    let badgeClass = "badge-answered";
    if (status === "clarified") badgeClass = "badge-clarified";
    if (status === "insufficient_evidence") badgeClass = "badge-insufficient";
    if (status === "refused") badgeClass = "badge-refused";
    if (status === "redirected") badgeClass = "badge-redirected";
    if (status === "error") badgeClass = "badge-error";

    // Card Header
    let badgeText = status.toUpperCase().replace("_", " ");
    if (status === "answered") {
        badgeText = `${confidence} CONFIDENCE`;
    }

    let headerHtml = `
        <div class="card-header">
            <div class="assistant-avatar-section">
                <div class="assistant-avatar">
                    <i data-lucide="activity" class="icon-sm"></i>
                </div>
                <h3 class="assistant-title">${status === 'redirected' ? 'Emergency Alert' : 'Clinical Support Response'}</h3>
            </div>
            <span class="state-badge ${badgeClass}">${badgeText}</span>
        </div>
    `;

    // Recommendations & Content rendering
    let contentHtml = `<div class="card-content">`;
    if (status === "redirected") {
        contentHtml += `<p class="font-bold" style="color:var(--color-red-error);">${escapeHtml(recommendation)}</p>`;
    } else {
        // Format simple lists if lines look like list items
        const paragraphs = recommendation.split("\n\n");
        paragraphs.forEach(para => {
            if (para.trim().startsWith("- ") || para.trim().startsWith("* ")) {
                contentHtml += "<ul>";
                para.split("\n").forEach(line => {
                    contentHtml += `<li>${escapeHtml(line.replace(/^[-*]\s+/, ""))}</li>`;
                });
                contentHtml += "</ul>";
            } else {
                contentHtml += `<p>${escapeHtml(para)}</p>`;
            }
        });
    }
    contentHtml += `</div>`;

    // Evidence Sources Grid (answered/clarified only)
    let sourcesHtml = "";
    if (evidenceItems.length > 0 && (status === "answered" || status === "clarified")) {
        sourcesHtml = `
            <div class="card-sources">
                <h4 class="sources-title">
                    <i data-lucide="book-open" class="icon-xs"></i>
                    Evidence Sources
                </h4>
                <div class="sources-grid">
        `;
        evidenceItems.forEach(item => {
            sourcesHtml += `
                <div class="source-item">
                    <div class="source-meta">
                        <span class="source-tag">${escapeHtml(item.citation || "Citation")}</span>
                        <i data-lucide="external-link" class="icon-xs" style="color: var(--color-on-surface-variant);"></i>
                    </div>
                    <p class="source-text">${escapeHtml(item.claim)}</p>
                </div>
            `;
        });
        sourcesHtml += `</div></div>`;
    }

    // Safety Banner note
    const safetyHtml = `
        <div class="safety-note-box">
            <i data-lucide="shield" class="icon-xs"></i>
            <span><strong>Safety Note:</strong> ${escapeHtml(safetyNote)}</span>
        </div>
    `;

    card.innerHTML = headerHtml + contentHtml + sourcesHtml + safetyHtml;
    return card;
}

// Handles submitting queries
async function submitQuery() {
    const textarea = document.getElementById("query");
    if (textarea.disabled) return; // Prevent duplicate submissions / rapid clicks
    
    const queryText = textarea.value.trim();
    if (!queryText && !selectedFile) return;

    // Check/create active conversation if missing
    if (!activeConversationId) {
        startNewChat();
    }

    const activeConv = conversations.find(c => c.id === activeConversationId);
    if (!activeConv) return;

    // If first message in conversation, update title from first question
    if (activeConv.messages.length === 0) {
        activeConv.title = queryText ? (queryText.length > 25 ? queryText.substring(0, 22) + "..." : queryText) : "Document Query";
    }

    // Disable controls
    const submitBtn = document.getElementById("submit-btn");
    const sendIcon = document.getElementById("send-icon");
    const spinner = document.getElementById("spinner");
    const attachBtn = document.getElementById("attach-btn");
    
    textarea.disabled = true;
    submitBtn.disabled = true;
    attachBtn.disabled = true;
    sendIcon.classList.add("hidden");
    spinner.classList.remove("hidden");

    let documentIdToQuery = activeConv.documentId || null;
    let fileUploadedThisTurn = null;

    try {
        // Guard against duplicate document uploads if selectedFile is set but activeConv.documentId is already set
        if (selectedFile && activeConv.documentId) {
            alert("This conversation already has an attached document. Start a new chat to attach a different document.");
            selectedFile = null;
            const fileInput = document.getElementById("file-input");
            if (fileInput) fileInput.value = "";
            renderAttachmentPreview();
            
            // Re-enable input state
            textarea.disabled = false;
            submitBtn.disabled = false;
            attachBtn.disabled = false;
            sendIcon.classList.remove("hidden");
            spinner.classList.add("hidden");
            return;
        }

        // Upload file synchronously if selected and not yet uploaded for this conversation
        if (selectedFile && !activeConv.documentId) {
            const originalPlaceholder = textarea.placeholder;
            textarea.placeholder = "Uploading document...";
            
            const formData = new FormData();
            formData.append("file", selectedFile);
            
            const uploadRes = await fetch(`${API_BASE}/documents/upload`, {
                method: "POST",
                body: formData
            });

            if (!uploadRes.ok) {
                const errData = await uploadRes.json();
                throw new Error(errData.detail || "Upload and indexing failed.");
            }

            const uploadData = await uploadRes.json();
            documentIdToQuery = uploadData.document_id;
            activeConv.documentId = documentIdToQuery;
            activeConv.documentName = selectedFile.name;
            fileUploadedThisTurn = selectedFile.name;
            
            textarea.placeholder = "Indexing document...";
        }

        // Now append User Message to UI & local history
        // If file was uploaded this turn, we associate it with the message bubble
        const userMsg = { 
            role: "user", 
            content: queryText || `Querying document: ${activeConv.documentName}` 
        };
        if (fileUploadedThisTurn) {
            userMsg.attachedFile = fileUploadedThisTurn;
        } else if (selectedFile && activeConv.documentId) {
            // Already uploaded in this chat, but user tried to attach it again? No, we render it
            userMsg.attachedFile = activeConv.documentName;
        } else if (activeConv.documentName && queryText) {
            // Include document tag context silently or explicitly if you want, but user request doesn't demand it
        }
        
        activeConv.messages.push(userMsg);
        textarea.value = "";
        
        // Reset selectedFile now that it has been successfully sent/uploaded
        selectedFile = null;
        const fileInput = document.getElementById("file-input");
        if (fileInput) fileInput.value = "";
        renderAttachmentPreview();
        
        renderActiveChat();

        // Build payload using isolated active history only (max last 6 messages)
        const historyPayload = activeConv.messages.slice(0, -1).map(msg => ({
            role: msg.role,
            content: msg.content
        }));

        const queryPayload = {
            question: userMsg.content,
            conversation: historyPayload
        };
        if (documentIdToQuery) {
            queryPayload.document_id = documentIdToQuery;
        }

        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(queryPayload)
        });

        if (!response.ok) {
            throw new Error(`Server returned error status: ${response.statusText}`);
        }

        const data = await response.json();
        
        // Append response
        activeConv.messages.push({
            role: "assistant",
            content: data.recommendation || "",
            responseObj: data
        });
        
    } catch (e) {
        console.error(e);
        activeConv.messages.push({
            role: "assistant",
            content: `Failed to process request: ${e.message}`,
            responseObj: {
                status: "error",
                recommendation: `Unable to process this document. Please check the file and try again. (Details: ${e.message})`,
                confidence: "low",
                safety_note: "System failure."
            }
        });
    } finally {
        saveSessionsToStorage();
        renderSidebar();
        renderActiveChat();

        // Re-enable input state
        textarea.disabled = false;
        submitBtn.disabled = false;
        attachBtn.disabled = false;
        sendIcon.classList.remove("hidden");
        spinner.classList.add("hidden");
        textarea.placeholder = "Ask a question about Autism Spectrum Disorder...";
        textarea.focus();
    }
}

// Utility to escape HTML variables preventing injection
function escapeHtml(str) {
    if (!str) return "";
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function renderAttachmentPreview() {
    const previewContainer = document.getElementById("attachment-preview");
    if (!selectedFile) {
        previewContainer.innerHTML = "";
        previewContainer.classList.add("hidden");
        return;
    }
    
    // Truncate filename if too long
    let displayName = selectedFile.name;
    if (displayName.length > 25) {
        const ext = displayName.substring(displayName.lastIndexOf('.'));
        displayName = displayName.substring(0, 20) + "..." + ext;
    }
    
    previewContainer.innerHTML = `
        <div class="attachment-chip">
            <i data-lucide="file" class="icon-sm"></i>
            <span>${escapeHtml(displayName)}</span>
            <button class="chip-remove-btn" id="remove-attachment-btn" aria-label="Remove attachment" title="Remove attachment">&times;</button>
        </div>
    `;
    previewContainer.classList.remove("hidden");
    
    document.getElementById("remove-attachment-btn").addEventListener("click", () => {
        selectedFile = null;
        const fileInput = document.getElementById("file-input");
        if (fileInput) fileInput.value = "";
        renderAttachmentPreview();
    });
    
    createIconsSafe();
}
