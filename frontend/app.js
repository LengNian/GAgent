(function () {
  "use strict";
  var API_BASE = (window.NMS_AGENT_CONFIG && window.NMS_AGENT_CONFIG.apiBase) || "";
  var STORAGE_KEY = "nms_agent_thread_id";
  var MAX_LENGTH = 4000;
  
  var threadState = {
    threadId: null,
    messages: [],
    busy: false, 
    creating: false, 
    failedContent: "" 
  };

  var elements = {};

  ["newThread", "clearThread", "statusDot", "sessionStatus", "conversationTitle", "messages", "error", "errorText", "retry", "dismissError", "chatForm", "input", "send", "hint", "counter", "railStatus", "threadId", "connectionDot", "connectionStatus", "connectionNote"].forEach(function (id) { elements[id] = document.getElementById(id); });

  // [逻辑规划] 去掉配置地址末尾的斜杠，再拼接接口路径，兼容同源和独立前端部署。
  function apiUrl(path) { return API_BASE.replace(/\/$/, "") + path; }
  
  
  function setStoredThreadId(threadId) {
    // [逻辑规划] 有会话时保存 ID，没有会话时删除 ID；浏览器存储失败只记录调试信息，不能阻塞对话。
    try {
      threadId ? localStorage.setItem(STORAGE_KEY, threadId) : localStorage.removeItem(STORAGE_KEY);
    } catch (error) {
      console.debug("Unable to update the stored thread ID.", error);
    }
  }
  
  
  function setCurrentThread(threadId) {
    // [逻辑规划] 同步内存、localStorage 和 URL 三处会话状态，使刷新页面能够恢复当前会话。
    threadState.threadId = threadId || null;
    setStoredThreadId(threadState.threadId);
    var url = new URL(window.location.href);
    threadState.threadId ? url.searchParams.set("thread_id", threadState.threadId) : url.searchParams.delete("thread_id");
    window.history.replaceState({}, "", url);
    updateUi();
  }


  function setConnection(text, note, tone) {
    // [逻辑规划] 同时更新顶部连接状态、说明文字和状态点样式，保持两个视图一致。
    elements.connectionStatus.textContent = text;
    elements.connectionNote.textContent = note;
    elements.connectionDot.className = tone || "";
  }


  function setSession(text, tone) {
    // [逻辑规划] 将会话状态同步到顶部状态和侧栏状态，避免页面不同区域显示不一致。
    elements.sessionStatus.textContent = text;
    elements.railStatus.textContent = text;
    elements.statusDot.className = tone || "";
  }


  function updateUi() {
    // [逻辑规划] 根据是否有会话、是否创建中、是否生成中，统一计算输入框和按钮的可用状态。
    var active = Boolean(threadState.threadId);
    var canSend = active && !threadState.busy && !threadState.creating;
    elements.input.disabled = !canSend;
    elements.send.disabled = !canSend || !elements.input.value.trim();
    elements.newThread.disabled = threadState.creating;
    elements.clearThread.classList.toggle("hidden", !active);
    elements.threadId.textContent = threadState.threadId || "—";
    elements.hint.textContent = !active ? "创建会话后可发送" : threadState.busy ? "正在生成回复" : "Enter 换行，发送按钮提交";
    elements.counter.textContent = elements.input.value.length + " / " + MAX_LENGTH;
  }


  function clearError() {
    // [逻辑规划] 隐藏错误区域、清除错误文本，并删除上一次失败请求的重试内容。
    elements.error.classList.add("hidden");
    elements.errorText.textContent = "";
    elements.retry.classList.add("hidden");
    threadState.failedContent = "";
  }


  function showError(text, retryContent) {
    // [逻辑规划] 展示用户可理解的错误；只有存在原始输入时才显示重试入口。
    elements.errorText.textContent = text;
    elements.error.classList.remove("hidden");
    elements.retry.classList.toggle("hidden", !retryContent);
    threadState.failedContent = retryContent || "";
    setSession("需要处理", "error");
    setConnection("连接异常", text, "error");
  }


  function showEmpty() {
    // [逻辑规划] 根据是否已创建会话渲染不同空状态；未创建时额外提供新对话入口。
    elements.messages.innerHTML = "";
    var box = document.createElement("div");
    box.className = "empty";
    var mark = document.createElement("div");
    mark.className = "empty-mark";
    mark.textContent = threadState.threadId ? "•" : "+";
    var title = document.createElement("h2");
    title.textContent = threadState.threadId ? "会话已创建" : "会话尚未开始";
    var copy = document.createElement("p");
    copy.textContent = threadState.threadId ? "输入问题开始本次对话。" : "创建一个新会话后开始提问。";
    box.appendChild(mark);
    box.appendChild(title);
    box.appendChild(copy);
    if (!threadState.threadId) {
      var button = document.createElement("button");
      button.className = "button";
      button.type = "button";
      button.textContent = "+  新对话";
      button.addEventListener("click", createThread);
      box.appendChild(button);
    }
    elements.messages.appendChild(box);
  }


  function messageNode(message) {
    // [逻辑规划] 按消息角色创建气泡；生成中的助手消息显示动画，其余内容使用 textContent 渲染。
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
    // [逻辑规划] 记录用户是否接近底部，重建消息 DOM 后仅在必要时自动滚动到底部。
    var nearBottom = forceBottom || elements.messages.scrollHeight - elements.messages.scrollTop - elements.messages.clientHeight < 120;
    elements.messages.innerHTML = "";
    if (!threadState.messages.length) { showEmpty(); return; }
    var fragment = document.createDocumentFragment();
    threadState.messages.forEach(function (message) { fragment.appendChild(messageNode(message)); });
    elements.messages.appendChild(fragment);
    if (nearBottom) elements.messages.scrollTop = elements.messages.scrollHeight;
  }


  function responseError(response) {
    // [逻辑规划] 优先解析服务端结构化错误字段；响应体无法解析时使用 HTTP 状态码生成提示。
    return response.json().then(function (data) {
      return data.detail || data.message || data.error || "请求失败（" + response.status + "）";
    }).catch(function () { return "请求失败（" + response.status + "）"; });
  }


  function updateHeader() {
    // [逻辑规划] 会话创建、恢复或清除后重新计算标题、顶部状态和连接提示。
    elements.conversationTitle.textContent = threadState.threadId ? "当前运维会话" : "从一个问题开始";
    setSession(threadState.threadId ? "已连接" : "无活动会话", threadState.threadId ? "active" : "");
    if (!threadState.threadId) setConnection("等待连接", "创建会话后连接服务。", "");
  }


  function createThread() {
    // [逻辑规划] 先锁定创建状态防止重复点击，再请求服务端 UUID；成功后清空旧消息并激活输入框。
    if (threadState.creating) return;
    threadState.creating = true;
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
        threadState.messages = [];
        setCurrentThread(data.thread_id);
        updateHeader();
        render(false);
        setConnection("已连接", "会话已准备就绪。", "active");
        elements.input.focus();
      })
      .catch(function (error) { showError(error.message || "创建会话失败，请稍后重试。"); })
      .finally(function () { threadState.creating = false; updateUi(); });
  }


  function loadThread(threadId) {
    // [逻辑规划] 请求指定会话的历史消息；成功后恢复消息，404 时清除本地会话并提示重新创建。
    threadState.busy = true;
    setCurrentThread(threadId);
    updateHeader();
    updateUi();
    setConnection("正在加载", "恢复当前会话。", "");
    fetch(apiUrl("/api/threads/" + encodeURIComponent(threadId) + "/messages"))
      .then(function (response) {
        if (response.status === 404) {
          setCurrentThread(null);
          threadState.messages = [];
          updateHeader();
          render(false);
          return Promise.reject(new Error("当前会话不可用，请创建新对话。"));
        }
        if (!response.ok) return responseError(response).then(function (message) { throw new Error(message); });
        return response.json();
      })
      .then(function (data) {
        threadState.messages = Array.isArray(data) ? data : (data.messages || []);
        render(false);
        setConnection("已连接", "会话已恢复。", "active");
      })
      .catch(function (error) { showError(error.message || "无法恢复当前会话。"); })
      .finally(function () { threadState.busy = false; updateUi(); });
  }


  function setAssistant(content, streaming, failed) {
    // [逻辑规划] 复用最后一条助手消息接收增量文本；没有助手消息时创建一条，保证消息顺序不变。
    var message = threadState.messages[threadState.messages.length - 1];
    if (!message || message.role !== "assistant") {
      message = { role: "assistant", content: "" };
      threadState.messages.push(message);
    }
    message.content = content;
    message.streaming = Boolean(streaming);
    message.error = Boolean(failed);
    render(true);
  }


  function parseEvent(block) {
    // [逻辑规划] 读取 event 和 data 行，合并多行 data 后解析 JSON；格式错误时抛出可展示异常。
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
    // [逻辑规划] delta 追加文本，done 使用最终文本，error 转换为异常，未知事件保持当前内容。
    if (!event) return content;
    if (event.name === "delta") return content + (event.data.text || "");
    if (event.name === "done") return (event.data.message && event.data.message.content) || content;
    if (event.name === "error") throw new Error(event.data.message || "Agent 执行失败。");
    return content;
  }


  function readStream(response) {
    // [逻辑规划] 持续读取网络块并缓存不完整事件，逐个解析 SSE；每个增量事件都更新同一条助手消息。
    if (!response.body) return Promise.reject(new Error("当前浏览器不支持流式响应。"));
    var reader = response.body.getReader();
    var decoder = new TextDecoder();
    var buffer = "";
    var content = "";
    function next() {
      // [逻辑规划] 读取一块数据；未结束则递归继续，结束则处理剩余缓冲并提交最终助手消息。
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
    // [逻辑规划] 先校验输入和会话状态，再乐观展示用户消息；失败时标记临时助手消息并保留重试内容。
    var content = value.trim();
    if (!content || threadState.busy || !threadState.threadId) return;
    if (content.length > MAX_LENGTH) { showError("问题不能超过 " + MAX_LENGTH + " 个字符。"); return; }
    clearError();
    threadState.busy = true;
    threadState.messages.push({ role: "user", content: content });
    threadState.messages.push({ role: "assistant", content: "", streaming: true });
    elements.input.value = "";
    resizeInput();
    render(true);
    updateUi();
    setSession("生成中", "active");
    setConnection("生成中", "正在接收 Agent 回复。", "active");
    fetch(apiUrl("/api/threads/" + encodeURIComponent(threadState.threadId) + "/chat"), {
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
        var assistant = threadState.messages[threadState.messages.length - 1];
        if (assistant && assistant.role === "assistant") {
          assistant.streaming = false;
          assistant.content = error.message || "本次回答失败，请重试。";
          assistant.error = true;
        }
        render(true);
        threadState.failedContent = content;
        showError(error.message || "本次回答失败，请重试。", content);
      })
      .finally(function () { threadState.busy = false; updateUi(); elements.input.focus(); });
  }


  function resizeInput() {
    // [逻辑规划] 先重置高度再按内容设置最大高度，最后重新计算发送按钮状态。
    elements.input.style.height = "auto";
    elements.input.style.height = Math.min(elements.input.scrollHeight, 180) + "px";
    updateUi();
  }


  function clearThread() {
    // [逻辑规划] 清除当前会话、消息、错误和运行状态，并同步移除 URL 与 localStorage 中的会话 ID。
    setCurrentThread(null);
    threadState.messages = [];
    threadState.busy = false;
    clearError();
    updateHeader();
    render(false);
    updateUi();
  }

  // [事件绑定] 将页面操作映射到会话创建、清除、重试、输入和发送逻辑。
  elements.newThread.addEventListener("click", createThread);
  elements.clearThread.addEventListener("click", clearThread);
  elements.dismissError.addEventListener("click", clearError);
  elements.retry.addEventListener("click", function () {
    var content = threadState.failedContent;
    clearError();
    elements.input.value = content;
    resizeInput();
    elements.chatForm.requestSubmit();
  });
  elements.input.addEventListener("input", resizeInput);
  elements.input.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      elements.chatForm.requestSubmit();
    }
  });
  elements.chatForm.addEventListener("submit", function (event) {
    event.preventDefault();
    sendMessage(elements.input.value);
  });

  var params = new URLSearchParams(window.location.search);
  var savedId = null;
  try {
    savedId = localStorage.getItem(STORAGE_KEY);
  } catch (error) {
    console.debug("Unable to read the stored thread ID.", error);
  }
  var initialId = params.get("thread_id") || savedId;
  updateHeader();
  updateUi();
  if (initialId) loadThread(initialId);
  else render(false);
})();
