"use strict";

const $ = (id) => document.getElementById(id);
const storageKey = "lumen.conversations.v1";
let chats = [];
try {
  const saved = JSON.parse(localStorage.getItem(storageKey) || "[]");
  if (Array.isArray(saved)) chats = saved.filter((chat) =>
    typeof chat.id === "string" && typeof chat.title === "string" &&
    Array.isArray(chat.messages) && chat.messages.length % 2 === 0 &&
    chat.messages.every((m, i) => m && m.role === (i % 2 ? "assistant" : "user") &&
      typeof m.text === "string" && m.text.length > 0 && m.text.length <= 20000)
  ).slice(0, 20);
} catch { /* Private browsing or unavailable storage: keep this session in memory. */ }

let current = { id: crypto.randomUUID(), title: "New conversation", messages: [] };
let busy = false;

function save() {
  chats = [current, ...chats.filter((chat) => chat.id !== current.id)].slice(0, 20);
  try { localStorage.setItem(storageKey, JSON.stringify(chats)); }
  catch { $("composer-hint").textContent = "This conversation is available until you close the page."; }
}

function messageElement(role, text, pending = false) {
  const article = document.createElement("article");
  article.className = `message ${role}${pending ? " pending" : ""}`;
  const label = document.createElement("div");
  label.className = "message-label";
  if (role === "assistant") {
    const star = document.createElement("span");
    star.className = "answer-star";
    star.textContent = "✳";
    star.setAttribute("aria-hidden", "true");
    label.append(star);
  }
  label.append(document.createTextNode(role === "user" ? "You" : "Static"));
  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = text;
  article.append(label, body);
  if (role === "assistant" && !pending) {
    const copy = document.createElement("button");
    copy.className = "copy-button";
    copy.textContent = "Copy answer";
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(text); copy.textContent = "Copied ✓"; }
      catch { copy.textContent = "Select the answer to copy"; }
      setTimeout(() => { copy.textContent = "Copy answer"; }, 2000);
    });
    article.append(copy);
  }
  return article;
}

function render() {
  $("welcome").hidden = current.messages.length > 0 || busy;
  $("messages").replaceChildren(...current.messages.map((m) => messageElement(m.role, m.text)));
  $("room-label").textContent = current.messages.length ? "Following the thread" : "A fresh perspective";
  $("chat-list").replaceChildren();
  $("chat-count").textContent = String(chats.length).padStart(2, "0");
  if (!chats.length) {
    const hint = document.createElement("p");
    hint.className = "empty-history";
    hint.textContent = "Your conversations will find a home here.";
    $("chat-list").append(hint);
  }
  for (const chat of chats) {
    const button = document.createElement("button");
    button.className = `chat-item${chat.id === current.id ? " active" : ""}`;
    button.textContent = chat.title;
    button.title = chat.title;
    button.disabled = busy;
    if (chat.id === current.id) button.setAttribute("aria-current", "true");
    button.addEventListener("click", () => { if (!busy) { current = chat; resetComposer(); render(); scrollDown(); } });
    $("chat-list").append(button);
  }
  $("new-chat").disabled = busy;
}

function scrollDown() { $("scroll-area").scrollTop = $("scroll-area").scrollHeight; }
function updateSend() { $("send").disabled = busy || !$("query").value.trim(); }
function resetComposer() { $("query").value = ""; $("error").hidden = true; updateSend(); $("query").focus(); }

$("new-chat").addEventListener("click", () => {
  if (busy) return;
  current = { id: crypto.randomUUID(), title: "New conversation", messages: [] };
  resetComposer();
  render();
  $("scroll-area").scrollTop = 0;
});
$("query").addEventListener("input", updateSend);
$("query").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!busy && $("query").value.trim()) $("chat-form").requestSubmit();
  }
});
document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    $("query").value = button.dataset.query;
    updateSend();
    $("query").focus();
  });
});

$("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = $("query").value.trim();
  if (busy || !query) return;
  busy = true;
  $("error").hidden = true;
  $("query").value = "";
  $("query").disabled = true;
  $("composer-hint").textContent = "A little thinking in progress…";
  updateSend();
  render();
  $("messages").append(messageElement("user", query), messageElement("assistant", "Finding context and preparing your answer…", true));
  scrollDown();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 180000);
  try {
    const response = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, history: current.messages.slice(-12) }),
      signal: controller.signal,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "The answer could not be prepared. Please try again.");
    if (typeof result.answer !== "string" || !result.answer.trim()) throw new Error("No answer was returned. Please try again.");
    if (!current.messages.length) current.title = query.slice(0, 70);
    current.messages.push({ role: "user", text: query }, { role: "assistant", text: result.answer });
    $("composer-hint").textContent = "Keep pulling the thread. Ask a follow-up.";
    save();
  } catch (error) {
    $("error").textContent = error.name === "AbortError"
      ? "This is taking longer than expected. Your question is back in the message box so you can try again."
      : error instanceof TypeError ? "Could not connect. Check that your local chat service is running, then try again." : error.message;
    $("error").hidden = false;
    $("query").value = query;
    $("composer-hint").textContent = "Your question is ready to try again.";
  } finally {
    clearTimeout(timer);
    busy = false;
    $("query").disabled = false;
    render();
    updateSend();
    scrollDown();
    $("query").focus();
  }
});

render();
updateSend();



