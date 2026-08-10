// NCERT Biology Teacher -- chat frontend logic.
// WHAT: streams answers from POST /api/chat/stream (newline-delimited JSON)
// and renders them token-by-token as they arrive, instead of waiting for
// the whole response -- this is the actual fix for "too much latency": the
// first words appear in ~1-2s instead of a silent multi-second wait.
// WHY plain fetch()/ReadableStream, no framework: keeps the whole app
// deployable with zero build step.
// Conversation history is tracked HERE, client-side, as a plain array, and
// sent explicitly with every request -- see server.py's docstring for why
// (Flask sessions don't play well with streaming responses).

const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const clearBtn = document.getElementById("clear-btn");
const chapterSelect = document.getElementById("chapter-select");

let conversationHistory = []; // [[question, answer], ...]

function scrollToBottom() {
  chatLog.scrollTop = chatLog.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function addUserMessage(question) {
  const msg = document.createElement("div");
  msg.className = "msg msg-user";
  msg.innerHTML = `<div class="avatar">🧑‍🎓</div><div class="bubble">${escapeHtml(question)}</div>`;
  chatLog.appendChild(msg);
  scrollToBottom();
}

/** Creates an empty assistant message bubble (with the live-typing cursor
 * via the "streaming" class) that sendQuestion() fills in as text arrives. */
function addStreamingAssistantMessage() {
  const msg = document.createElement("div");
  msg.className = "msg msg-assistant streaming";
  msg.innerHTML = `
    <div class="avatar">🧬</div>
    <div class="msg-content">
      <div class="bubble"></div>
    </div>`;
  chatLog.appendChild(msg);
  scrollToBottom();
  return msg;
}

function finishAssistantMessage(msgEl, sources, imageUrl) {
  msgEl.classList.remove("streaming");
  const content = msgEl.querySelector(".msg-content");

  if (sources && sources.length) {
    const tags = sources.map((s) => `<span class="source-tag">Ch.${s.chapter} p.${s.page}</span>`).join("");
    const sourcesDiv = document.createElement("div");
    sourcesDiv.className = "sources";
    sourcesDiv.innerHTML = tags;
    content.appendChild(sourcesDiv);
  }

  if (imageUrl) {
    const wrap = document.createElement("div");
    wrap.innerHTML = `<img class="answer-image" src="${imageUrl}" alt="Source page diagram">
                       <div class="image-caption">Chapter ${sources[0].chapter}, page ${sources[0].page}</div>`;
    content.appendChild(wrap);
  }
  scrollToBottom();
}

async function sendQuestion(question) {
  addUserMessage(question);
  const assistantMsg = addStreamingAssistantMessage();
  const bubble = assistantMsg.querySelector(".bubble");

  chatInput.disabled = true;
  chatForm.querySelector("button").disabled = true;

  const chapter = chapterSelect.value ? Number(chapterSelect.value) : null;
  let fullAnswer = "";
  let sources = [];
  let imageUrl = null;

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, chapter, history: conversationHistory }),
    });

    if (!response.ok || !response.body) {
      bubble.textContent = "Something went wrong. Please try again.";
    } else {
      // Read the response as a stream of newline-delimited JSON objects --
      // each {"type":"text", "content":"..."} appends to the bubble live;
      // the final {"type":"meta", ...} carries sources/image.
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let newlineIndex;
        while ((newlineIndex = buffer.indexOf("\n")) !== -1) {
          const line = buffer.slice(0, newlineIndex);
          buffer = buffer.slice(newlineIndex + 1);
          if (!line.trim()) continue;

          const event = JSON.parse(line);
          if (event.type === "text") {
            fullAnswer += event.content;
            bubble.textContent = fullAnswer; // live-updating as chunks arrive
            scrollToBottom();
          } else if (event.type === "meta") {
            sources = event.sources || [];
            imageUrl = event.image_url || null;
          } else if (event.type === "error") {
            fullAnswer = event.message || "Something went wrong.";
            bubble.textContent = fullAnswer;
          }
        }
      }
    }

    finishAssistantMessage(assistantMsg, sources, imageUrl);
    conversationHistory.push([question, fullAnswer]);
  } catch (err) {
    bubble.textContent = "Sorry, I couldn't reach the server. Please try again.";
    finishAssistantMessage(assistantMsg, [], null);
  } finally {
    chatInput.disabled = false;
    chatForm.querySelector("button").disabled = false;
    chatInput.focus();
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;
  chatInput.value = "";
  sendQuestion(question);
});

document.querySelectorAll(".chip").forEach((btn) => {
  btn.addEventListener("click", () => sendQuestion(btn.dataset.question));
});

clearBtn.addEventListener("click", () => {
  conversationHistory = [];
  chatLog.innerHTML = "";
  const msg = document.createElement("div");
  msg.className = "msg msg-assistant";
  msg.innerHTML = `<div class="avatar">🧬</div><div class="bubble">Conversation cleared. What would you like to ask?</div>`;
  chatLog.appendChild(msg);
});
