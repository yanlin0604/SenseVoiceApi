---
name: realtime-field-extraction
description: 医生语音问诊时实时提取患者信息并填充入院记录表单字段
date: 2026-06-23
status: approved
---

# 实时字段提取与填充设计方案

## 一、需求概述

医生在使用病历助手插件进行语音问诊时，系统需要实时分析医患对话内容，自动提取患者的基本信息（如姓名、年龄、主诉等），并将提取结果实时填充到入院记录表单的对应字段中。

**核心目标：**
- 医生边问诊边自动填表，减少手动录入工作量
- 实时响应，类似 Proactor AI 的交互体验
- 不覆盖从 HIS 系统预填充的历史数据
- 支持多轮对话的上下文理解

## 二、架构设计

### 2.1 整体架构

```
┌─────────────────┐
│   医生 + 患者    │
│   (语音对话)     │
└────────┬────────┘
         │ 实时音频流
         ▼
┌─────────────────────────┐
│  Python ASR Service     │
│  - VAD 断句             │
│  - 语音识别 (2pass)     │
│  - 说话人分离           │
└────────┬────────────────┘
         │ WebSocket: { text, speaker, is_final }
         ▼
┌─────────────────────────┐
│      前端浏览器          │
│  - 显示识别文本         │
│  - 转发文本给 Java      │
│  - 接收字段更新         │
│  - 实时填充表单         │
└────┬──────────┬─────────┘
     │          │
     │ WS1      │ WS2
     │ (识别)    │ (字段)
     │          ▼
     │    ┌─────────────────────────┐
     │    │   Java 后台服务          │
     │    │  - 接收对话文本          │
     │    │  - 维护对话上下文        │
     │    │  - 查询字段模板          │
     │    │  - 调用大模型提取字段     │
     │    │  - 推送字段更新          │
     │    └──────────┬──────────────┘
     │               │
     │               ▼
     │         ┌───────────┐
     │         │  MySQL DB │
     │         │  - 字段模板│
     │         │  - 历史病历│
     └────────►└───────────┘
           Webhook (可选，用于持久化识别文本)
```

**Why:** 采用前端双 WebSocket 协调的架构，保持 Python 专注语音识别，Java 专注业务逻辑，职责清晰且易于扩展。

**How to apply:** Python ASR 服务无需修改，Java 后台新增字段提取模块，前端建立两个 WebSocket 连接实现实时协调。

### 2.2 数据流设计

**流程 1: Python → 前端（识别文本推送）**
```json
{
  "session_id": "xxx",
  "text": "患者叫张三，今年45岁",
  "speaker": "用户2",
  "is_final": true,
  "mode": "refined"
}
```

**流程 2: 前端 → Java（转发识别文本）**
```json
{
  "session_id": "xxx",
  "doc_code": "DOC001",
  "patient_id": "123456",
  "text": "患者叫张三，今年45岁",
  "speaker": "用户2",
  "timestamp": 1719123456789
}
```

**流程 3: Java → 前端（推送字段更新）**
```json
{
  "session_id": "xxx",
  "action": "update_fields",
  "fields": {
    "patient_name": "张三",
    "age": "45",
    "age_unit": "岁"
  },
  "confidence": 0.95
}
```

**Why:** WebSocket 双向通信保证实时性，JSON 格式便于前后端解析和扩展。

**How to apply:** 前端接收到 `is_final=true` 的消息才转发给 Java（避免流式中间结果干扰），Java 推送字段时附带置信度供前端判断是否需要高亮提示。

## 三、前端实现

### 3.1 双 WebSocket 连接管理

```javascript
// 连接 Python ASR 服务
const wsAsr = new WebSocket('ws://asr-server:8000/ws/asr?mode=1');

// 连接 Java 后台服务
const wsJava = new WebSocket('ws://java-server:8080/ws/field-extraction');

// 会话初始化
wsJava.onopen = () => {
  wsJava.send(JSON.stringify({
    action: 'init_session',
    session_id: sessionId,
    doc_code: 'DOC001',  // 入院记录
    patient_id: currentPatientId,
    pre_filled_fields: {  // 从 HIS 预填充的字段
      id_card: '110101199001011234',
      phone: '13800138000'
    }
  }));
};
```

**Why:** 会话初始化时告知 Java 当前上下文（文书类型、患者ID、已有字段），避免重复提取。

**How to apply:** 医生打开入院记录表单时立即建立连接并初始化会话，关闭表单时发送 `close_session` 消息释放资源。

### 3.2 接收识别文本并转发

```javascript
wsAsr.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  // 显示识别文本（实时更新对话记录区域）
  appendTranscript(data.text, data.speaker, data.is_final);
  
  // 只转发最终结果（is_final=true）
  if (data.is_final && data.text.trim()) {
    wsJava.send(JSON.stringify({
      session_id: data.session_id,
      text: data.text,
      speaker: data.speaker,
      timestamp: Date.now()
    }));
  }
};
```

**Why:** 仅转发最终精修结果（`is_final=true`），避免流式中间结果的不稳定性干扰字段提取。

**How to apply:** 前端需要同时维护识别文本显示区（展示所有结果）和转发逻辑（仅转发最终结果）。

### 3.3 接收字段更新并填充表单

```javascript
wsJava.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.action === 'update_fields') {
    Object.keys(data.fields).forEach(fieldKey => {
      const inputElement = document.querySelector(`[name="${fieldKey}"]`);
      
      if (inputElement && !inputElement.value) {
        // 只填充空字段，不覆盖已有值
        inputElement.value = data.fields[fieldKey];
        
        // 标记为 AI 自动填充（可选高亮样式）
        inputElement.classList.add('ai-filled');
        inputElement.setAttribute('data-confidence', data.confidence);
      }
    });
  }
};
```

**Why:** 只填充空字段避免覆盖从 HIS 预填充的历史数据，AI 标记让医生知道哪些是自动填充的。

**How to apply:** 可以通过 CSS 为 `.ai-filled` 添加淡黄色背景或左侧蓝色边框，医生确认后可点击移除标记。

## 四、Java 后台实现

### 4.1 会话管理（基于 LangGraph4j）

**LangGraph4j 状态定义：**

```java
@Data
public class FieldExtractionState {
    // 会话基本信息
    private String sessionId;
    private String docCode;           // 文书编码，如 "DOC001"
    private Long patientId;           // 患者ID
    private String patientIdHis;      // HIS系统患者号
    
    // 对话历史
    private List<DialogTurn> dialogHistory = new ArrayList<>();
    
    // 已提取的字段
    private Map<String, Object> extractedFields = new ConcurrentHashMap<>();
    
    // 前端预填充的字段（从 HIS 获取）
    private Map<String, Object> preFilledFields = new ConcurrentHashMap<>();
    
    // 字段模板列表
    private List<DocFieldTemplate> fieldTemplates;
    
    // 提取状态
    private int lastExtractedIndex = 0;  // 上次提取时的对话索引
    private Instant lastExtractedAt;     // 上次提取时间
    private Instant createdAt;
    private Instant lastActivityAt;
    
    // LangGraph 流程控制
    private String currentNode;          // 当前节点："receive" / "check" / "extract" / "push"
    private boolean shouldExtract;       // 是否触发提取
}

@Data
public class DialogTurn {
    private String speaker;      // "用户1", "用户2"
    private String text;         // 识别文本
    private Instant timestamp;   // 时间戳
}
```

**Why:** 使用 LangGraph4j 管理会话状态和流程转换，状态机清晰可控，便于调试和扩展。

**How to apply:** LangGraph4j 的 State 对象贯穿整个流程，每个节点接收 State 并返回更新后的 State，避免全局变量和线程安全问题。

### 4.2 LangGraph4j 流程设计

**状态机流程图：**

```
                    ┌──────────────┐
                    │  INIT_SESSION│
                    │  (初始化)     │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
              ┌────►│  RECEIVE_TEXT│◄────┐
              │     │  (接收文本)   │     │
              │     └──────┬───────┘     │
              │            │             │
              │            ▼             │
              │     ┌──────────────┐    │
              │     │ CHECK_TRIGGER│    │
              │     │ (判断是否提取)│    │
              │     └──────┬───────┘    │
              │            │             │
              │       ┌────┴────┐        │
              │       │         │        │
              │    NO │         │ YES    │
              └───────┘         ▼        │
                         ┌──────────────┐│
                         │ EXTRACT_FIELD││
                         │ (提取字段)    ││
                         └──────┬───────┘│
                                │        │
                                ▼        │
                         ┌──────────────┐│
                         │  PUSH_UPDATE ││
                         │  (推送更新)   ││
                         └──────┬───────┘│
                                │        │
                                └────────┘
```

**LangGraph4j 实现代码：**

```java
@Service
public class FieldExtractionGraph {
    
    private final StateGraph<FieldExtractionState> graph;
    
    public FieldExtractionGraph() {
        this.graph = new StateGraph<>(FieldExtractionState.class)
            // 定义节点
            .addNode("receive_text", this::receiveTextNode)
            .addNode("check_trigger", this::checkTriggerNode)
            .addNode("extract_fields", this::extractFieldsNode)
            .addNode("push_update", this::pushUpdateNode)
            
            // 定义边
            .addEdge("__start__", "receive_text")
            .addConditionalEdge("check_trigger", this::shouldExtractCondition,
                Map.of(
                    "extract", "extract_fields",
                    "wait", "receive_text"
                ))
            .addEdge("extract_fields", "push_update")
            .addEdge("push_update", "receive_text")
            
            .compile();
    }
    
    // 节点1: 接收文本
    private FieldExtractionState receiveTextNode(FieldExtractionState state) {
        state.setCurrentNode("receive_text");
        state.setLastActivityAt(Instant.now());
        // 文本已经在外部添加到 dialogHistory，这里只做状态标记
        return state;
    }
    
    // 节点2: 判断是否触发提取
    private FieldExtractionState checkTriggerNode(FieldExtractionState state) {
        state.setCurrentNode("check_trigger");
        
        int newDialogCount = state.getDialogHistory().size() - state.getLastExtractedIndex();
        
        // 条件1: 累积了 3 条新对话
        if (newDialogCount >= 3) {
            state.setShouldExtract(true);
            return state;
        }
        
        // 条件2: 距离上次提取超过 30 秒
        if (state.getLastExtractedAt() != null) {
            long secondsSinceLastExtraction = Duration.between(
                state.getLastExtractedAt(), Instant.now()
            ).getSeconds();
            if (secondsSinceLastExtraction > 30 && newDialogCount > 0) {
                state.setShouldExtract(true);
                return state;
            }
        }
        
        // 条件3: 检测到关键词
        String lastText = state.getDialogHistory()
            .get(state.getDialogHistory().size() - 1)
            .getText();
        if (containsKeywords(lastText, Arrays.asList("叫", "岁", "年龄", "症状", "不舒服"))) {
            state.setShouldExtract(true);
            return state;
        }
        
        state.setShouldExtract(false);
        return state;
    }
    
    // 条件边: 判断是否提取
    private String shouldExtractCondition(FieldExtractionState state) {
        return state.isShouldExtract() ? "extract" : "wait";
    }
    
    // 节点3: 提取字段
    private FieldExtractionState extractFieldsNode(FieldExtractionState state) {
        state.setCurrentNode("extract_fields");
        
        // 调用大模型提取字段
        String prompt = buildPrompt(state);
        String aiResponse = aiModelService.chat(prompt);
        Map<String, Object> extractedFields = parseJsonResponse(aiResponse);
        
        // 过滤有效字段
        Map<String, Object> validFields = filterValidFields(state, extractedFields);
        
        // 增量更新
        Map<String, Object> newFields = new HashMap<>();
        validFields.forEach((key, value) -> {
            if (!state.getExtractedFields().containsKey(key)) {
                newFields.put(key, value);
                state.getExtractedFields().put(key, value);
            }
        });
        
        // 更新提取状态
        state.setLastExtractedIndex(state.getDialogHistory().size());
        state.setLastExtractedAt(Instant.now());
        
        // 将新字段暂存到状态中，供下一个节点推送
        state.put("newFields", newFields);
        
        return state;
    }
    
    // 节点4: 推送更新
    private FieldExtractionState pushUpdateNode(FieldExtractionState state) {
        state.setCurrentNode("push_update");
        
        Map<String, Object> newFields = (Map<String, Object>) state.get("newFields");
        
        if (newFields != null && !newFields.isEmpty()) {
            webSocketService.sendToSession(state.getSessionId(), Map.of(
                "action", "update_fields",
                "fields", newFields,
                "confidence", 0.95
            ));
        }
        
        return state;
    }
    
    // 对外接口: 处理新到达的对话文本
    public void handleIncomingText(String sessionId, DialogTurn turn) {
        FieldExtractionState state = sessionManager.getOrCreateState(sessionId);
        state.getDialogHistory().add(turn);
        
        // 执行流程: receive_text -> check_trigger -> (extract_fields -> push_update)?
        CompletableFuture.runAsync(() -> {
            try {
                graph.invoke(state);
            } catch (Exception e) {
                log.error("字段提取流程失败: session={}, error={}", sessionId, e.getMessage());
            }
        });
    }
}
```

**Why:** LangGraph4j 提供声明式的状态机定义，节点和边清晰可见，易于调试和扩展。条件边（`shouldExtractCondition`）自动路由到不同分支，避免复杂的 if-else 嵌套。

**How to apply:** 每个节点是纯函数（接收 State 返回 State），无副作用，方便单元测试。异步执行 `graph.invoke()` 避免阻塞 WebSocket 线程。

### 4.3 大模型 Prompt 设计（参考现有风格）

**说明：** 以下 Prompt 结构仅为示例，实际开发时请参考项目中现有的 Prompt 模板风格和格式。

```java
private String buildPrompt(FieldExtractionState state) {
    StringBuilder prompt = new StringBuilder();
    
    // 1. 角色与任务说明（参考现有 Prompt 的角色定义格式）
    prompt.append("你是一个医疗信息提取助手，从医患对话中提取入院记录的字段信息。\n\n");
    
    // 2. 字段模板（只包含 is_dictatable=1 的字段）
    prompt.append("【字段模板】\n");
    prompt.append("以下是需要提取的字段：\n");
    for (DocFieldTemplate field : state.getFieldTemplates()) {
        if (field.getIsDictatable() == 1) {
            prompt.append(String.format("- %s (%s)\n", 
                field.getFieldKey(), field.getFieldLabel()));
        }
    }
    prompt.append("\n");
    
    // 3. 已知信息（HIS 预填充的字段）
    if (!state.getPreFilledFields().isEmpty()) {
        prompt.append("【已知信息】\n");
        prompt.append("以下字段已从 HIS 系统获取，无需提取：\n");
        state.getPreFilledFields().forEach((key, value) -> {
            prompt.append(String.format("- %s: %s\n", key, value));
        });
        prompt.append("\n");
    }
    
    // 4. 对话历史（最近 10 轮）
    prompt.append("【对话历史】\n");
    List<DialogTurn> recentDialogs = state.getDialogHistory()
        .subList(Math.max(0, state.getDialogHistory().size() - 10), 
                 state.getDialogHistory().size());
    for (DialogTurn turn : recentDialogs) {
        prompt.append(String.format("%s: %s\n", turn.getSpeaker(), turn.getText()));
    }
    prompt.append("\n");
    
    // 5. 提取规则（参考现有 Prompt 的规则描述格式）
    prompt.append("【提取规则】\n");
    prompt.append("1. 根据对话内容判断：提问的通常是医生，回答个人信息和症状的是患者\n");
    prompt.append("2. 只提取患者相关的信息（姓名、年龄、症状等）\n");
    prompt.append("3. 只提取对话中明确提到的信息\n");
    prompt.append("4. 如果某个字段在对话中未提及，不要返回该字段\n");
    prompt.append("5. 以 JSON 格式返回，例如：{\"patient_name\": \"张三\", \"age\": \"45\", \"age_unit\": \"岁\"}\n\n");
    
    prompt.append("请提取字段：");
    
    return prompt.toString();
}
```

**Why:** 让大模型自动推断角色（避免前端标记），仅包含最近 10 轮对话控制 token 消耗，明确告知已知信息避免重复提取。

**How to apply:** 实际开发时，请查看项目中已有的大模型调用代码（如文书生成、ICD推荐等模块），复用现有的 Prompt 构建工具类和格式规范，保持风格统一。

### 4.4 大模型调用与结果处理

```java
private void extractAndPushFields(FieldExtractionSession session) {
    try {
        // 1. 构建 Prompt
        String prompt = buildPrompt(session);
        
        // 2. 调用大模型 API
        String aiResponse = aiModelService.chat(prompt);
        
        // 3. 解析 JSON 结果
        Map<String, Object> extractedFields = parseJsonResponse(aiResponse);
        
        // 4. 过滤：只保留字段模板中定义的字段
        Map<String, Object> validFields = new HashMap<>();
        Set<String> allowedKeys = session.getFieldTemplates().stream()
            .filter(f -> f.getIsDictatable() == 1)
            .map(DocFieldTemplate::getFieldKey)
            .collect(Collectors.toSet());
        
        extractedFields.forEach((key, value) -> {
            if (allowedKeys.contains(key) && value != null && !value.toString().isEmpty()) {
                validFields.put(key, value);
            }
        });
        
        // 5. 增量更新：只推送新提取的字段
        Map<String, Object> newFields = new HashMap<>();
        validFields.forEach((key, value) -> {
            if (!session.getExtractedFields().containsKey(key)) {
                newFields.put(key, value);
                session.getExtractedFields().put(key, value);
            }
        });
        
        // 6. 通过 WebSocket 推送给前端
        if (!newFields.isEmpty()) {
            webSocketService.sendToSession(session.getSessionId(), Map.of(
                "action", "update_fields",
                "fields", newFields,
                "confidence", 0.95  // 可选：从大模型响应中提取置信度
            ));
        }
        
        // 7. 更新提取状态
        session.setLastExtractedIndex(session.getDialogHistory().size());
        session.setLastExtractedAt(Instant.now());
        
    } catch (Exception e) {
        log.error("字段提取失败: session={}, error={}", session.getSessionId(), e.getMessage());
    }
}
```

**Why:** 增量推送只发送新提取的字段，避免重复推送已提取的字段造成前端闪烁。

**How to apply:** 异常处理要宽容（不能因为一次提取失败就中断会话），记录日志便于排查问题。

## 五、数据库设计

### 5.1 使用现有表

**字段模板查询：**
```sql
-- 查询入院记录的可语音输入字段
SELECT field_key, field_label, section_name, input_type
FROM tpl_doc_field
WHERE doc_code = 'DOC001'
  AND is_dictatable = 1
  AND is_active = 1
ORDER BY section_name, field_order;
```

**会话持久化（可选）：**
```sql
-- 复用 biz_recording_session 表记录会话
INSERT INTO biz_recording_session (
    session_no, session_type, status, hospital_code,
    doctor_code, visit_id, patient_id_his
) VALUES (?, 'supplement', 'recording', ?, ?, ?, ?);

-- 复用 biz_asr_transcript 表持久化识别文本
INSERT INTO biz_asr_transcript (
    recording_session_id, segment_index, start_time_sec, end_time_sec,
    speaker_id, transcript_text
) VALUES (?, ?, ?, ?, ?, ?);
```

**Why:** 无需新增表，复用现有的会话和转写记录表即可满足需求。

**How to apply:** 会话持久化是可选的，如果不需要长期保存识别文本和字段提取记录，可以只保留内存中的会话状态。

## 六、关键技术要点

### 6.1 说话人角色处理

**问题：** Python ASR 返回的说话人标识是 "用户1"、"用户2"，无法直接区分医生和患者。

**解决方案：** 不依赖角色标记，让大模型根据对话内容自动推断：
- 提问的通常是医生（如"叫什么名字？"）
- 回答个人信息和症状的是患者（如"我叫张三"）

**Why:** 避免前端或后端做复杂的角色标记逻辑，利用大模型的上下文理解能力自动判断。

**How to apply:** 在 Prompt 中明确告知大模型这个推断规则，如果提取准确率低，可以增加示例对话。

### 6.2 字段冲突处理

**场景 1：HIS 预填充 vs AI 提取**
- 策略：前端只填充空字段，不覆盖 HIS 的历史数据
- 实现：`if (inputElement && !inputElement.value)`

**场景 2：AI 多次提取同一字段**
- 策略：Java 后台只推送首次提取的结果，避免重复推送
- 实现：`if (!session.getExtractedFields().containsKey(key))`

**场景 3：医生手动修改 vs AI 提取**
- 策略：前端标记哪些字段是 AI 填充的，医生修改后移除标记
- 实现：添加 `.ai-filled` 样式，医生修改后 `removeClass('ai-filled')`

**Why:** 分层处理冲突，HIS 数据优先级最高（不覆盖），医生手动输入次之（可覆盖 AI），AI 自动填充优先级最低。

**How to apply:** 可以在前端增加"重新提取"按钮，允许医生清空某个字段后重新触发提取。

### 6.3 成本与性能优化

**触发策略优化：**
- 累积 3 条对话：平衡实时性和调用频率
- 30 秒超时：避免长时间静默后医生以为系统卡住
- 关键词检测：快速响应高价值信息（如姓名、年龄）

**成本估算：**
- 假设一次入院记录问诊 10 分钟，产生 30 轮对话
- 智能触发约调用 8-10 次大模型
- 每次 Prompt 约 1000 tokens（字段模板 + 10 轮对话）
- 使用通义千问 Qwen-Plus：约 ¥0.008/千tokens，总计 ¥0.08-0.10/次问诊

**性能优化：**
- 异步提取：不阻塞 WebSocket 接收线程
- 会话缓存：字段模板启动时加载，不每次查询数据库
- 对话历史限制：只保留最近 10 轮，控制 token 消耗

**Why:** 智能触发策略在体验和成本之间找到平衡点，每次问诊成本控制在 1 毛钱以内。

**How to apply:** 上线后监控实际调用频率和准确率，根据数据调整触发阈值。

## 七、实施步骤

### 阶段 1：Java 后台开发（核心）

1. **WebSocket 端点**：`/ws/field-extraction`
2. **LangGraph4j 流程图**：`FieldExtractionGraph`（状态机定义）
3. **状态管理器**：`FieldExtractionStateManager`（会话状态缓存）
4. **字段提取服务**：`FieldExtractionService`（辅助工具类）
5. **大模型集成**：`AIModelService`（复用现有配置和 Prompt 风格）
6. **字段模板查询**：`DocFieldTemplateMapper`

**关键依赖：**
- LangGraph4j：用于状态机管理
- 现有大模型调用框架：复用 Prompt 构建工具和 API 调用逻辑

### 阶段 2：前端开发

1. **双 WebSocket 连接管理**
2. **识别文本显示区域**
3. **字段自动填充逻辑**
4. **AI 填充标记样式**
5. **会话初始化与关闭**

### 阶段 3：联调测试

1. **功能测试**：模拟医患对话，验证字段提取准确性
2. **性能测试**：验证触发策略的实时性和成本
3. **边界测试**：长对话、网络中断、重连等场景
4. **用户体验测试**：医生实际使用反馈

### 阶段 4：上线监控

1. **监控指标**：提取准确率、调用频率、响应延迟
2. **日志收集**：记录每次提取的 Prompt 和结果，便于优化
3. **用户反馈**：收集医生对字段提取质量的评价

**Why:** 分阶段实施降低风险，Java 后台是核心（最复杂），前端相对简单，联调确保端到端流程。

**How to apply:** 可以先在测试环境上线，邀请 1-2 个科室试用，收集反馈后再全院推广。

## 八、风险与应对

### 8.1 大模型提取准确率低

**风险：** 方言、口语化表达、多轮对话上下文理解失败

**应对：**
- 在 Prompt 中增加示例对话
- 使用更强的大模型（如 GPT-4、Claude）
- 建立错误案例库，持续优化 Prompt

### 8.2 网络延迟或断线

**风险：** WebSocket 连接不稳定，导致字段更新丢失

**应对：**
- 前端增加断线重连机制
- Java 后台保留会话状态 30 分钟，支持重连后恢复
- 前端定期心跳检测连接状态

### 8.3 字段提取延迟高

**风险：** 大模型调用耗时 2-3 秒，影响实时体验

**应对：**
- 使用更快的模型（如 Qwen-Turbo）
- 关键词触发快速响应高价值字段
- 前端显示"正在分析..."提示

### 8.4 成本超预期

**风险：** 频繁调用导致成本过高

**应对：**
- 提高触发阈值（如改为累积 5 条对话）
- 使用更便宜的模型
- 增加本地规则提取（如正则匹配姓名、年龄）

**Why:** 提前识别风险并准备应对方案，避免上线后措手不及。

**How to apply:** 在测试阶段重点验证这些风险场景，根据实际情况调整策略。

## 九、总结

本设计方案采用前端双 WebSocket 协调的架构，保持 Python ASR 服务专注语音识别，Java 后台专注业务逻辑和字段提取，职责清晰且易于扩展。通过智能触发策略平衡实时性和成本，通过增量推送和冲突处理保证用户体验。

**核心优势：**
- 实时响应：类似 Proactor AI 的交互体验
- 架构清晰：Python 做语音，Java 做业务，前端协调
- 流程可控：LangGraph4j 状态机管理，节点和边清晰可见
- 风格统一：复用现有 Prompt 模板和大模型调用框架
- 成本可控：每次问诊约 ¥0.08-0.10
- 易于扩展：后续可增加更复杂的业务规则

**技术栈：**
- Python ASR 服务：无需修改
- Java 后台：Spring Boot + WebSocket + LangGraph4j + 现有大模型框架
- 前端：双 WebSocket 连接 + 字段自动填充
- 数据库：复用现有表（`tpl_doc_field`、`biz_recording_session` 等）

**下一步：** 开始 Java 后台的字段提取模块开发，实现 LangGraph4j 状态机和会话管理。
