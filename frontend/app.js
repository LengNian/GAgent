(function () {
  "use strict";
  var API_BASE = (window.NMS_AGENT_CONFIG && window.NMS_AGENT_CONFIG.apiBase) || "";
  var STORAGE_KEY = "nms_agent_thread_id";
  var MAX_LENGTH = 4000;
  
  var state = { 
    threadId: null,
    messages: [],
    busy: false, 
    creating: false, 
    failedContent: "" 
  };

  var el = {};

  ["newThread", "clearThread", "statusDot", "sessionStatus", "conversationTitle", "messages", "error", "errorText", "retry", "dismissError", "chatForm", "input", "send", "hint", "counter", "railStatus", "threadId", "connectionDot", "connectionStatus", "connectionNote"].forEach(function (id) { el[id] = document.getElementById(id); });

  function apiUrl(path) { return API_BASE.replace(/\/$/, "") + path; }
  
  
  function setStored(id) { try { id ? localStorage.setItem(STORAGE_KEY, id) : localStorage.removeItem(STORAGE_KEY); } catch (error) {} }
  
  
  function setThread(id) {
    state.threadId = id || null;
    setStored(state.threadId);
    var url = new URL(window.location.href);
    state.threadId ? url.searchParams.set("thread_id", state.threadId) : url.searchParams.delete("thread_id");
    window.history.replaceState({}, "", url);
    updateUi();
  }


  function setConnection(text, note, tone) {
    el.connectionStatus.textContent = text;
    el.connectionNote.textContent = note;
    el.connectionDot.className = tone || "";
  }


  function setSession(text, tone) {
    el.sessionStatus.textContent = text;
    el.railStatus.textContent = text;
    el.statusDot.className = tone || "";
  }


  function updateUi() {
    var active = Boolean(state.threadId);
    var canSend = active && !state.busy && !state.creating;
    el.input.disabled = !canSend;
    el.send.disabled = !canSend || !el.input.value.trim();
    el.newThread.disabled = state.creating;
    el.clearThread.classList.toggle("hidden", !active);
    el.threadId.textContent = state.threadId || "—";
    el.hint.textContent = !active ? "创建会话后可发送" : state.busy ? "正在生成回复" : "Enter 换行，发送按钮提交";
    el.counter.textContent = el.input.value.length + " / " + MAX_LENGTH;
  }


  function clearError() {
    el.error.classList.add("hidden");
    el.errorText.textContent = "";
    el.retry.classList.add("hidden");
    state.failedContent = "";
  }


  function showError(text, retryContent) {
    el.errorText.textContent = text;
    el.error.classList.remove("hidden");
    el.retry.classList.toggle("hidden", !retryContent);
    state.failedContent = retryContent || "";
    setSession("需要处理", "error");
    setConnection("连接异常", text, "error");
  }


  function showEmpty() {
    el.messages.innerHTML = "";
    var box = document.createElement("div");
    box.className = "empty";
    var mark = document.createElement("div");
    mark.className = "empty-mark";
    mark.textContent = state.threadId ? "•" : "+";
    var title = document.createElement("h2");
    title.textContent = state.threadId ? "会话已创建" : "会话尚未开始";
    var copy = document.createElement("p");
    copy.textContent = state.threadId ? "输入问题开始本次对话。" : "创建一个新会话后开始提问。";
    box.appendChild(mark);
    box.appendChild(title);
    box.appendChild(copy);
    if (!state.threadId) {
      var button = document.createElement("button");
      button.className = "button";
      button.type = "button";
      button.textContent = "+  新对话";
      button.addEventListener("click", createThread);
      box.appendChild(button);
    }
    el.messages.appendChild(box);
  }


  function messageNode(message) {
    var row = document.createElement("div");
    row.className = "message-row " + message.role;
    var bubble = document.createElement("div");
    bubble.className = "message-bubble" + (message.error ? " error" : "");
    var meta = document.createElement("div");
    meta.className = "message-meta";
    meta.textContent = message.role === "user" ? "YOU" : "AGENT";
    bubble.appendChild(meta);
    if (message.streaming && !message.content) {
      var typing = document.createElement("span");
      typing.className = "typing";
      typing.innerHTML = "生成中 <i></i><i></i><i></i>";
      bubble.appendChild(typing);
    } else {
      var text = document.createElement("p");
      text.className = "message-text";
      text.textContent = message.content || "";
      bubble.appendChild(text);
    }
    row.appendChild(bubble);
    return row;
  }


  function render(forceBottom) {
    var nearBottom = forceBottom || el.messages.scrollHeight - el.messages.scrollTop - el.messages.clientHeight < 120;
    el.messages.innerHTML = "";
    if (!state.messages.length) { showEmpty(); return; }
    var fragment = document.createDocumentFragment();
    state.messages.forEach(function (message) { fragment.appendChild(messageNode(message)); });
    el.messages.appendChild(fragment);
    if (nearBottom) el.messages.scrollTop = el.messages.scrollHeight;
  }


  function responseError(response) {
    return response.json().then(function (data) {
      return data.detail || data.message || data.error || "请求失败（" + response.status + "）";
    }).catch(function () { return "请求失败（" + response.status + "）"; });
  }


  function updateHeader() {
    el.conversationTitle.textContent = state.threadId ? "当前运维会话" : "从一个问题开始";
    setSession(state.threadId ? "已连接" : "无活动会话", state.threadId ? "active" : "");
    if (!state.threadId) setConnection("等待连接", "创建会话后连接服务。", "");
  }


  function createThread() {
    if (state.creating) return;
    state.creating = true;
    clearError();
    updateUi();
    setConnection("正在连接", "创建新会话。", "");
    fetch(apiUrl("/api/threads"), { method: "POST", headers: { "Content-Type": "application/json" } })
      .then(function (response) {
        if (!response.ok) return responseError(response).then(function (message) { throw new Error(message); });
        return response.json();
      })
      .then(function (data) {
        if (!data.thread_id) throw new Error("服务未返回有效的 thread_id。");
        state.messages = [];
        setThread(data.thread_id);
        updateHeader();
        render(false);
        setConnection("已连接", "会话已准备就绪。", "active");
        el.input.focus();
      })
      .catch(function (error) { showError(error.message || "创建会话失败，请稍后重试。"); })
      .finally(function () { state.creating = false; updateUi(); });
  }


  function loadThread(id) {
    state.busy = true;
    setThread(id);
    updateHeader();
    updateUi();
    setConnection("正在加载", "恢复当前会话。", "");
    fetch(apiUrl("/api/threads/" + encodeURIComponent(id) + "/messages"))
      .then(function (response) {
        if (response.status === 404) {
          setThread(null);
          state.messages = [];
          updateHeader();
          render(false);
          return Promise.reject(new Error("当前会话不可用，请创建新对话。"));
        }
        if (!response.ok) return responseError(response).then(function (message) { throw new Error(message); });
        return response.json();
      })
      .then(function (data) {
        state.messages = Array.isArray(data) ? data : (data.messages || []);
        render(false);
        setConnection("已连接", "会话已恢复。", "active");
      })
      .catch(function (error) { showError(error.message || "无法恢复当前会话。"); })
      .finally(function () { state.busy = false; updateUi(); });
  }


  function setAssistant(content, streaming, failed) {
    var message = state.messages[state.messages.length - 1];
    if (!message || message.role !== "assistant") {
      message = { role: "assistant", content: "" };
      state.messages.push(message);
    }
    message.content = content;
    message.streaming = Boolean(streaming);
    message.error = Boolean(failed);
    render(true);
  }


  function parseEvent(block) {
    var name = "message";
    var dataLines = [];
    block.split(/\r?\n/).forEach(function (line) {
      if (line.indexOf("event:") === 0) name = line.slice(6).trim();
      if (line.indexOf("data:") === 0) dataLines.push(line.slice(5).trim());
    });
    if (!dataLines.length) return null;
    try { return { name: name, data: JSON.parse(dataLines.join("\n")) }; }
    catch (error) { throw new Error("服务返回了无法解析的流式数据。"); }
  }


  function applyEvent(event, content) {
    if (!event) return content;
    if (event.name === "delta") return content + (event.data.text || "");
    if (event.name === "done") return (event.data.message && event.data.message.content) || content;
    if (event.name === "error") throw new Error(event.data.message || "Agent 执行失败。");
    return content;
  }


  function readStream(response) {
    if (!response.body) return Promise.reject(new Error("当前浏览器不支持流式响应。"));
    var reader = response.body.getReader();
    var decoder = new TextDecoder();
    var buffer = "";
    var content = "";
    function next() {
      return reader.read().then(function (chunk) {
        if (chunk.done) {
          buffer += decoder.decode();
          if (buffer.trim()) content = applyEvent(parseEvent(buffer), content);
          setAssistant(content, false, false);
          return;
        }
        buffer += decoder.decode(chunk.value, { stream: true });
        var blocks = buffer.split(/\r?\n\r?\n/);
        buffer = blocks.pop();
        blocks.forEach(function (block) {
          var event = parseEvent(block);
          content = applyEvent(event, content);
          if (event && (event.name === "delta" || event.name === "done")) setAssistant(content, event.name === "delta", false);
        });
        return next();
      });
    }
    return next();
  }


  function sendMessage(value) {
    var content = value.trim();
    if (!content || state.busy || !state.threadId) return;
    if (content.length > MAX_LENGTH) { showError("问题不能超过 " + MAX_LENGTH + " 个字符。"); return; }
    clearError();
    state.busy = true;
    state.messages.push({ role: "user", content: content });
    state.messages.push({ role: "assistant", content: "", streaming: true });
    el.input.value = "";
    resizeInput();
    render(true);
    updateUi();
    setSession("生成中", "active");
    setConnection("生成中", "正在接收 Agent 回复。", "active");
    fetch(apiUrl("/api/threads/" + encodeURIComponent(state.threadId) + "/chat"), {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ content: content })
    })
      .then(function (response) {
        if (!response.ok) return responseError(response).then(function (message) { throw new Error(message); });
        return readStream(response);
      })
      .then(function () {
        setSession("已连接", "active");
        setConnection("已连接", "回复已完成。", "active");
      })
      .catch(function (error) {
        var assistant = state.messages[state.messages.length - 1];
        if (assistant && assistant.role === "assistant") {
          assistant.streaming = false;
          assistant.content = error.message || "本次回答失败，请重试。";
          assistant.error = true;
        }
        render(true);
        state.failedContent = content;
        showError(error.message || "本次回答失败，请重试。", content);
      })
      .finally(function () { state.busy = false; updateUi(); el.input.focus(); });
  }


  function resizeInput() {
    el.input.style.height = "auto";
    el.input.style.height = Math.min(el.input.scrollHeight, 180) + "px";
    updateUi();
  }


  function clearThread() {
    setThread(null);
    state.messages = [];
    state.busy = false;
    clearError();
    updateHeader();
    render(false);
    updateUi();
  }

  el.newThread.addEventListener("click", createThread);
  el.clearThread.addEventListener("click", clearThread);
  el.dismissError.addEventListener("click", clearError);
  el.retry.addEventListener("click", function () {
    var content = state.failedContent;
    clearError();
    el.input.value = content;
    resizeInput();
    el.chatForm.requestSubmit();
  });
  el.input.addEventListener("input", resizeInput);
  el.input.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      el.chatForm.requestSubmit();
    }
  });
  el.chatForm.addEventListener("submit", function (event) {
    event.preventDefault();
    sendMessage(el.input.value);
  });

  var params = new URLSearchParams(window.location.search);
  var savedId = null;
  try { savedId = localStorage.getItem(STORAGE_KEY); } catch (error) {}
  var initialId = params.get("thread_id") || savedId;
  updateHeader();
  updateUi();
  if (initialId) loadThread(initialId);
  else render(false);
})();
