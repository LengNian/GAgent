(function () {
  "use strict";
  var API_BASE = (window.NMS_AGENT_CONFIG && window.NMS_AGENT_CONFIG.apiBase) || "";
  var STORAGE_KEY = "nms_agent_thread_id";
  var HISTORY_STORAGE_KEY = "nms_agent_thread_history";
  var SIDEBAR_STORAGE_KEY = "nms_agent_sidebar_collapsed";
  var MAX_LENGTH = 4000;
  
  var threadState = {
    threadId: null,
    messages: [],
    busy: false, 
    creating: false, 
    failedContent: "",
    executionSteps: [],
    history: [],
    sidebarCollapsed: false
  };

  var elements = {};

  ["appShell", "sidebar", "toggleSidebar", "openSidebar", "threadHistory", "historyEmpty", "newThread", "clearThread", "statusDot", "sessionStatus", "agentLogin", "conversationTitle", "messages", "error", "errorText", "retry", "dismissError", "chatForm", "input", "send", "hint", "counter", "railStatus", "threadId", "connectionDot", "connectionStatus", "connectionNote"].forEach(function (id) { elements[id] = document.getElementById(id); });

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


  function loadStoredHistory() {
    // [逻辑规划] 读取浏览器中的会话摘要；数据库接入后只替换这里的数据来源，不改变侧栏渲染协议。
    try {
      var stored = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) || "[]");
      if (!Array.isArray(stored)) return [];
      return stored.filter(function (item) {
        return item && typeof item.id === "string" && typeof item.title === "string";
      }).slice(0, 20);
    } catch (error) {
      console.debug("Unable to read the stored thread history.", error);
      return [];
    }
  }


  function saveHistory() {
    // [逻辑规划] 本地保存仅用于数据库接入前的演示和刷新恢复，写入失败不影响当前对话。
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(threadState.history.slice(0, 20)));
    } catch (error) {
      console.debug("Unable to update the stored thread history.", error);
    }
  }


  function renderHistory() {
    // [逻辑规划] 用稳定的按钮元素呈现会话摘要，当前会话只通过 active 状态标识，不复制消息内容。
    elements.threadHistory.innerHTML = "";
    elements.historyEmpty.classList.toggle("hidden", threadState.history.length > 0);
    threadState.history.forEach(function (item) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "thread-history-item" + (item.id === threadState.threadId ? " active" : "");
      button.title = item.title;
      var dot = document.createElement("i");
      dot.setAttribute("aria-hidden", "true");
      var title = document.createElement("span");
      title.textContent = item.title;
      button.appendChild(dot);
      button.appendChild(title);
      button.addEventListener("click", function () {
        if (item.id !== threadState.threadId) loadThread(item.id);
        setSidebarOpen(false);
      });
      elements.threadHistory.appendChild(button);
    });
  }


  function rememberThread(threadId, title) {
    // [逻辑规划] 新会话置顶；已有会话仅在仍是默认标题时更新，避免后续问题覆盖原始标题。
    if (!threadId) return;
    var existing = threadState.history.find(function (item) { return item.id === threadId; });
    if (existing) {
      if (title && existing.title === "新对话") existing.title = title;
      threadState.history = [existing].concat(threadState.history.filter(function (item) { return item.id !== threadId; }));
    } else {
      threadState.history.unshift({ id: threadId, title: title || "新对话" });
    }
    threadState.history = threadState.history.slice(0, 20);
    saveHistory();
    renderHistory();
  }


  function removeThreadFromHistory(threadId) {
    threadState.history = threadState.history.filter(function (item) { return item.id !== threadId; });
    saveHistory();
    renderHistory();
  }


  function setSidebarCollapsed(collapsed) {
    // [逻辑规划] 桌面端保存折叠偏好；移动端使用独立的打开状态，不改变桌面布局宽度。
    threadState.sidebarCollapsed = Boolean(collapsed);
    elements.appShell.classList.toggle("sidebar-collapsed", threadState.sidebarCollapsed);
    elements.toggleSidebar.querySelector("span").textContent = threadState.sidebarCollapsed ? "›" : "‹";
    elements.toggleSidebar.setAttribute("aria-expanded", String(!threadState.sidebarCollapsed));
    elements.toggleSidebar.setAttribute("title", threadState.sidebarCollapsed ? "展开侧栏" : "收起侧栏");
    elements.toggleSidebar.setAttribute("aria-label", threadState.sidebarCollapsed ? "展开侧栏" : "收起侧栏");
    try {
      localStorage.setItem(SIDEBAR_STORAGE_KEY, threadState.sidebarCollapsed ? "1" : "0");
    } catch (error) {
      console.debug("Unable to store the sidebar preference.", error);
    }
  }


  function setSidebarOpen(open) {
    // [逻辑规划] 移动端以抽屉状态打开或关闭侧栏，桌面端不改变布局折叠状态。
    elements.appShell.classList.toggle("sidebar-open", Boolean(open));
  }
  
  
  function setCurrentThread(threadId) {
    // [逻辑规划] 同步内存、localStorage 和 URL 三处会话状态，使刷新页面能够恢复当前会话。
    threadState.threadId = threadId || null;
    setStoredThreadId(threadState.threadId);
    var url = new URL(window.location.href);
    threadState.threadId ? url.searchParams.set("thread_id", threadState.threadId) : url.searchParams.delete("thread_id");
    window.history.replaceState({}, "", url);
    renderHistory();
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
    // [逻辑规划] 没有活动会话或正在创建会话时禁止编辑；模型生成期间允许编辑草稿，但禁止发送。
    var active = Boolean(threadState.threadId);
    var canEdit = active && !threadState.creating;
    var canSend = canEdit && !threadState.busy;
    elements.input.disabled = !canEdit;
    elements.send.disabled = !canSend || !elements.input.value.trim();
    elements.newThread.disabled = threadState.creating;
    elements.clearThread.classList.toggle("hidden", !active);
    elements.threadId.textContent = threadState.threadId || "—";
    elements.hint.textContent = !active ? "创建会话后可发送" : threadState.busy ? "正在生成回复，可编辑下一条问题" : "Enter 换行，发送按钮提交";
    elements.counter.textContent = elements.input.value.length + " / " + MAX_LENGTH;
  }


  function clearError() {
    // [逻辑规划] 隐藏错误区域、清除错误文本，并删除上一次失败请求的重试内容。
    elements.error.classList.add("hidden");
    elements.errorText.textContent = "";
    elements.retry.classList.add("hidden");
    threadState.failedContent = "";
  }


  function clearExecution() {
    // [逻辑规划] 清除上一次请求的结构化执行记录，避免不同问题的工具步骤混在一起。
    threadState.executionSteps = [];
  }


  function appendExecutionStep(eventName, data) {
    // [逻辑规划] 只保存服务端提供的状态摘要、工具名和脱敏参数，统一由当前助手气泡渲染。
    // 1. 根据事件类型生成用户可读文本；未知字段不影响已有进度。
    // 2. 去除连续重复事件，避免后端初始状态和前端即时状态重复显示。
    // 3. 触发消息重绘，使执行进度始终位于对应的助手消息内部。
    var step = { type: eventName, message: data.message || "" };
    if (eventName === "tool_start") {
      step.message = data.message || ("已选择工具：" + (data.tool_name || "工具") + "。" );
      if (data.arguments && Object.keys(data.arguments).length) {
        step.message += " 参数：" + JSON.stringify(data.arguments);
      }
    }
    if (!step.message) return;
    var lastStep = threadState.executionSteps[threadState.executionSteps.length - 1];
    if (lastStep && lastStep.type === eventName && lastStep.message === step.message) return;
    threadState.executionSteps.push(step);
    render(true);
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
    // [逻辑规划] 按消息角色创建气泡；当前助手消息将结构化执行进度与最终回答放在同一气泡内。
    var row = document.createElement("div");
    row.className = "message-row " + message.role;
    var bubble = document.createElement("div");
    bubble.className = "message-bubble" + (message.error ? " error" : "");
    var meta = document.createElement("div");
    meta.className = "message-meta";
    meta.textContent = message.role === "user" ? "YOU" : "AGENT";
    bubble.appendChild(meta);

    var isCurrentAssistant = message.role === "assistant" && message === threadState.messages[threadState.messages.length - 1];
    if (isCurrentAssistant && threadState.executionSteps.length) {
      var thinking = document.createElement("details");
      thinking.className = "thinking";
      thinking.open = true;
      var summary = document.createElement("summary");
      summary.textContent = "思考过程";
      thinking.appendChild(summary);
      var steps = document.createElement("div");
      steps.className = "thinking-steps";
      threadState.executionSteps.forEach(function (step) {
        var item = document.createElement("p");
        item.className = "thinking-step " + step.type;
        item.textContent = step.message;
        steps.appendChild(item);
      });
      thinking.appendChild(steps);
      bubble.appendChild(thinking);
    }

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


  var mockLoginPayload = {
    code: 0,
    data: {
      access_token: "string",
      expires_in: 0,
      token_type: "Bearer",
      user_info: {
        display_name: "string",
        granted_permissions: ["string"],
        mfa_enabled: true,
        roles: ["string"],
        tenant_id: "string",
        user_id: "string",
        username: "string"
      }
    },
    message: "string"
  };


  function agentLogin() {
    // [逻辑规划] 发送模拟登录响应；只展示登录状态，不输出 access_token，避免敏感信息泄露。
    elements.agentLogin.disabled = true;
    setConnection("正在登录", "正在提交模拟用户信息。", "");
    fetch(apiUrl("/api/agent/login"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(mockLoginPayload)
    })
      .then(function (response) {
        return response.json().then(function (data) {
          if (!response.ok) throw new Error(data.detail || "Agent 登录失败");
          return data;
        });
      })
      .then(function (result) {
        var userInfo = result.data && result.data.user_info;
        var displayName = userInfo && userInfo.display_name ? userInfo.display_name : "用户";
        setSession("已登录", "active");
        setConnection("已登录", displayName + " 已获得 Agent 访问结果。", "active");
      })
      .catch(function (error) {
        showError(error.message || "Agent 登录失败，请稍后重试。");
      })
      .finally(function () {
        elements.agentLogin.disabled = false;
      });
  }


  function createThread() {
    // [逻辑规划] 先锁定创建状态防止重复点击，再请求服务端 UUID；成功后清空旧消息并激活输入框。
    if (threadState.creating) return;
    threadState.creating = true;
    clearError();
    clearExecution();
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
        rememberThread(data.thread_id, "新对话");
        updateHeader();
        render(false);
        setConnection("已连接", "会话已准备就绪。", "active");
        setSidebarOpen(false);
        elements.input.focus();
      })
      .catch(function (error) { showError(error.message || "创建会话失败，请稍后重试。"); })
      .finally(function () { threadState.creating = false; updateUi(); });
  }


  function loadThread(threadId) {
    // [逻辑规划] 请求指定会话的历史消息；当前内存会话在刷新后不可恢复，404 或 405 时清除旧 ID 并回到无活动会话。
    threadState.busy = true;
    setCurrentThread(threadId);
    updateHeader();
    updateUi();
    setConnection("正在加载", "恢复当前会话。", "");
    fetch(apiUrl("/api/threads/" + encodeURIComponent(threadId) + "/messages"))
      .then(function (response) {
        if (response.status === 404 || response.status === 405) {
          removeThreadFromHistory(threadId);
          setCurrentThread(null);
          threadState.messages = [];
          clearError();
          clearExecution();
          updateHeader();
          render(false);
          return null;
        }
        if (!response.ok) return responseError(response).then(function (message) { throw new Error(message); });
        return response.json();
      })
      .then(function (data) {
        if (data === null) return;
        threadState.messages = Array.isArray(data) ? data : (data.messages || []);
        var firstUserMessage = threadState.messages.find(function (message) { return message.role === "user" && message.content; });
        rememberThread(threadId, firstUserMessage ? firstUserMessage.content.slice(0, 32) : "新对话");
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
    // [逻辑规划] 进度事件只更新独立状态区；delta 追加最终文本，done 使用服务端最终文本，error 转换为异常。
    if (!event) return content;
    if (event.name === "progress" || event.name === "agent_progress" || event.name === "tool_start" || event.name === "tool_end") {
      appendExecutionStep(event.name, event.data || {});
      return content;
    }
    if (event.name === "delta") return content + (event.data.text || "");
    if (event.name === "done") return (event.data.message && event.data.message.content) || content;
    if (event.name === "error") {
      var errorMessage = event.data.message || "Agent 执行失败。";
      if (event.data.trace_id) errorMessage += "（trace_id: " + event.data.trace_id + "）";
      throw new Error(errorMessage);
    }
    return content;
  }


  function readStream(response) {
    // [逻辑规划] 持续读取网络块并缓存不完整事件，逐个解析 SSE；每个增量事件都更新同一条助手消息。
    if (!response.body) return Promise.reject(new Error("当前浏览器不支持流式响应。"));
    var reader = response.body.getReader();
    var decoder = new TextDecoder();
    var buffer = "";
    var content = "";
    var sawTerminalEvent = false;
    function next() {
      // [逻辑规划] 读取一块数据；未结束则递归继续，结束则处理剩余缓冲并提交最终助手消息。
      return reader.read().then(function (chunk) {
        if (chunk.done) {
          buffer += decoder.decode();
          if (buffer.trim()) {
            var finalEvent = parseEvent(buffer);
            if (finalEvent && (finalEvent.name === "done" || finalEvent.name === "error")) sawTerminalEvent = true;
            content = applyEvent(finalEvent, content);
          }
          if (!sawTerminalEvent) throw new Error("流式响应提前结束，请重试。");
          setAssistant(content, false, false);
          return;
        }
        buffer += decoder.decode(chunk.value, { stream: true });
        var blocks = buffer.split(/\r?\n\r?\n/);
        buffer = blocks.pop();
        blocks.forEach(function (block) {
          var event = parseEvent(block);
          if (event && (event.name === "done" || event.name === "error")) sawTerminalEvent = true;
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
    clearExecution();
    threadState.busy = true;
    threadState.messages.push({ role: "user", content: content });
    threadState.messages.push({ role: "assistant", content: "", streaming: true });
    rememberThread(threadState.threadId, content.slice(0, 32));
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
    clearExecution();
    updateHeader();
    render(false);
    updateUi();
  }

  // [事件绑定] 将页面操作映射到侧栏、会话创建、清除、重试、输入和发送逻辑。
  elements.newThread.addEventListener("click", createThread);
  elements.agentLogin.addEventListener("click", agentLogin);
  elements.toggleSidebar.addEventListener("click", function () {
    if (window.matchMedia("(max-width: 820px)").matches) {
      setSidebarOpen(false);
      return;
    }
    setSidebarCollapsed(!threadState.sidebarCollapsed);
  });
  elements.openSidebar.addEventListener("click", function () { setSidebarOpen(true); });
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
      if (threadState.busy) return;
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
  var savedSidebarState = null;
  try {
    savedId = localStorage.getItem(STORAGE_KEY);
    savedSidebarState = localStorage.getItem(SIDEBAR_STORAGE_KEY);
  } catch (error) {
    console.debug("Unable to read the stored frontend state.", error);
  }
  threadState.history = loadStoredHistory();
  threadState.sidebarCollapsed = savedSidebarState === "1";
  setSidebarCollapsed(threadState.sidebarCollapsed);
  renderHistory();
  var initialId = params.get("thread_id") || savedId;
  updateHeader();
  updateUi();
  if (initialId) loadThread(initialId);
  else render(false);
})();
