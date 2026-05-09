# RMMS AI 服务器协议规范

**版本：** 1.0.0-draft
**状态：** 草案
**最后更新：** 2026-04-15

---

## 目录

1. [概述](#1-概述)
2. [协议版本控制](#2-协议版本控制)
3. [传输层](#3-传输层)
4. [身份认证](#4-身份认证)
5. [通用数据结构](#5-通用数据结构)
6. [REST API 端点](#6-rest-api-端点)
7. [SSE 事件流](#7-sse-事件流)
8. [能力声明](#8-能力声明)
9. [参数 Schema（UI 驱动）](#9-参数-schemaui-驱动)
10. [Pipeline 模型](#10-pipeline-模型)
11. [错误码体系](#11-错误码体系)
12. [文件传输](#12-文件传输)
13. [服务发现](#13-服务发现)
14. [幂等性与缓存](#14-幂等性与缓存)
15. [SSE 重连策略](#15-sse-重连策略)
16. [轨道类型系统](#16-轨道类型系统)
17. [未来预留](#17-未来预留)
18. [变更日志](#18-变更日志)

---

## 1. 概述

### 1.1 目的

本协议定义 **RMMS**（C++ DAW 客户端）与 **AI Server**（Python 后端）之间的通信规范。涵盖音频分轨分离、MIDI 转写、AI 辅助作曲和音符检测。

### 1.2 设计原则

- **可插拔后端**：切换 Python 环境即可支持 CUDA / NPU / XPU / MPS / CPU
- **Schema 驱动 UI**：服务端定义参数 Schema，客户端动态生成 UI
- **Pipeline 模型**：多步骤任务以单一 Pipeline 定义提交
- **三事件 SSE 体系**：`partial_result` + `progress` + `final_result`
- **前向兼容**：新字段被旧客户端忽略；破坏性变更递增主版本号

### 1.3 架构

```
RMMS (C++ / Qt6)  ←── REST + SSE ──→  AI Server (Python / FastAPI)
     QNetworkAccessManager              demucs / basic-pitch / ...
```

---

## 2. 协议版本控制

### 2.1 版本格式

语义化版本：`MAJOR.MINOR.PATCH`

- **MAJOR**：破坏性变更（字段删除、语义变更）
- **MINOR**：新功能（新端点、新字段、新能力）
- **PATCH**：缺陷修复（无协议变更）

### 2.2 版本协商

客户端在请求头中发送其支持的版本：

```
X-Protocol-Version: 1.0.0
```

服务端在响应头中返回其版本：

```
X-Protocol-Version: 1.2.0
```

若服务端主版本号 > 客户端主版本号，客户端应警告潜在不兼容。

### 2.3 前向兼容

- 基于 JSON：未知字段必须被接收方忽略
- 新枚举值可在次版本中添加
- 新 SSE 事件类型可在次版本中添加
- 字段不得在次版本中删除（应标记为废弃）

---

## 3. 传输层

### 3.1 REST API

- 基础路径：`/api/v1`
- Content-Type：`application/json` 或 `multipart/form-data`（见[第 6.3 节](#63-提交-pipeline创建任务)）
- 字符编码：UTF-8

### 3.2 SSE（Server-Sent Events）

- 端点：`GET /api/v1/tasks/{task_id}/events`
- Content-Type：`text/event-stream`
- 事件格式：

```
event: {event_type}
data: {json_payload}

```

- 心跳：服务端每 30 秒发送注释行（`: keep-alive`）
- 不支持 `Last-Event-ID`（见[第 15 节](#15-sse-重连策略)）

### 3.3 默认端口

- 默认：`8420`
- 可通过 `AI_SERVER_PORT` 环境变量或配置文件修改

---

## 4. 身份认证

### 4.1 模式

身份认证为**可选**。由服务端配置决定是否启用。

### 4.2 API Key

启用时，客户端必须包含：

```
X-API-Key: <api_key>
```

### 4.3 认证错误

| HTTP 状态码 | 错误码           | 描述                 |
| -------- | ------------- | ------------------ |
| 401      | AUTH_REQUIRED | 服务端要求 API Key 但未提供 |
| 403      | AUTH_INVALID  | 提供的 API Key 无效     |

---

## 5. 通用数据结构

### 5.1 错误对象

```json
{
  "code": "DEVICE_OOM",
  "message": "NPU out of memory",
  "details": {
    "device": "npu:0",
    "required_mb": 2048,
    "available_mb": 1024
  }
}
```

- `code`（string，必填）：机器可读错误码，见[第 11 节](#11-错误码体系)
- `message`（string，必填）：人类可读错误描述
- `details`（object，可选）：用于程序化处理的附加上下文

### 5.2 任务 ID

- 格式：UUID v4（如 `a1b2c3d4-e5f6-7890-abcd-ef1234567890`）
- 由服务端生成
- URL 安全、全局唯一、不可猜测

### 5.3 轨道对象

```json
{
  "track_type": "audio",
  "stem": "vocals",
  "label": "Vocals",
  "url": "/api/v1/files/a1b2c3d4/vocals.wav",
  "format": "wav",
  "sample_rate": 44100,
  "duration": 180.5,
  "size_bytes": 63580444
}
```

- `track_type`（string，必填）：`"audio"` | `"midi"` | `"hybrid"`
- `stem`（string，必填）：分轨标识符（如 `"vocals"`、`"drums"`）
- `label`（string，可选）：人类可读显示名称
- `url`（string，必填）：下载 URL（相对于服务端根路径）
- `format`（string，可选）：文件格式（`"wav"` | `"flac"` | `"mp3"` | `"mid"`）
- `sample_rate`（int，可选）：音频采样率（仅音频轨道）
- `duration`（float，可选）：时长（秒）
- `size_bytes`（int，可选）：文件大小（字节）

### 5.4 步骤结果对象

用于 `completed_urls` 和 `final_result.urls`，表示每个步骤的输出及元数据。

```json
{
  "step_index": 0,
  "step_type": "split",
  "urls": [
    "/api/v1/files/a1b2c3d4/vocals.wav",
    "/api/v1/files/a1b2c3d4/drums.wav"
  ]
}
```

- `step_index`（int，必填）：Pipeline 内的步骤索引（从 0 开始）
- `step_type`（string，必填）：该步骤的能力 ID（如 `"split"`、`"midi"`）
- `urls`（string[]，必填）：该步骤输出文件的下载 URL

### 5.5 步骤错误对象

用于 `errors` 数组，表示每个步骤的失败信息。

```json
{
  "step_index": 1,
  "error": {
    "code": "MODEL_LOAD_FAILED",
    "message": "basic-pitch model not found"
  }
}
```

- `step_index`（int，必填）：失败步骤的索引（从 0 开始）
- `error`（Error，必填）：错误详情（见[第 5.1 节](#51-错误对象)）

### 5.6 设备信息对象

```json
{
  "device_type": "npu",
  "device_index": 0,
  "name": "Ascend 910B",
  "available": true,
  "memory_total_mb": 32768,
  "memory_used_mb": 0
}
```

---

## 6. REST API 端点

### 6.1 健康检查

```
GET /api/v1/health
```

**响应：**

```json
{
  "status": "ok",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "model_loaded": "htdemucs",
  "active_tasks": 1,
  "queued_tasks": 0
}
```

### 6.2 获取能力列表

```
GET /api/v1/capabilities
```

返回完整的服务端配置：能力、设备、参数 Schema 和调度器配置。

**响应：**

```json
{
  "protocol_version": "1.0.0",
  "server_version": "0.1.0",
  "capabilities": [
    {
      "id": "split",
      "label": "Stem Separation",
      "description": "Separate audio into individual stems",
      "status": "implemented",
      "models": ["htdemucs", "htdemucs_6s"],
      "default_model": "htdemucs_6s",
      "param_defs": []
    },
    {
      "id": "midi",
      "label": "MIDI Transcription",
      "description": "Convert audio to MIDI",
      "status": "not_implemented",
      "models": [],
      "default_model": null,
      "param_defs": []
    },
    {
      "id": "generate",
      "label": "AI Composition",
      "description": "AI-assisted MIDI generation",
      "status": "not_implemented",
      "models": [],
      "default_model": null,
      "param_defs": []
    },
    {
      "id": "detect",
      "label": "Note Detection",
      "description": "Detect notes from audio",
      "status": "not_implemented",
      "models": [],
      "default_model": null,
      "param_defs": []
    }
  ],
  "devices": [
    {
      "device_type": "npu",
      "available": true,
      "count": 2,
      "install_hint": null,
      "units": [
        {"device_index": 0, "name": "Ascend 910B", "memory_total_mb": 32768},
        {"device_index": 1, "name": "Ascend 910B", "memory_total_mb": 32768}
      ]
    },
    {
      "device_type": "cuda",
      "available": false,
      "count": 0,
      "install_hint": "pip install torch --index-url https://download.pytorch.org/whl/cu128",
      "units": []
    }
  ],
  "scheduler": {
    "max_concurrent_tasks": 4,
    "max_queue_size": 20
  },
  "output_formats": ["wav", "flac", "mp3"],
  "output_packages": ["separate", "zip"],
  "max_upload_bytes": 524288000
}
```

**字段说明：**

- `capabilities[].status`：`"implemented"` | `"not_implemented"`
  - `"implemented"`：功能完整，可正常使用
  - `"not_implemented"`：已规划但尚未实现；客户端应显示为禁用/灰色，提示"尚未实现"
- `capabilities[].param_defs`（array）：参数 Schema 定义（见[第 9 节](#9-参数-schemaui-驱动)）
- `devices[].install_hint`：设备不可用时显示的安装提示（可用时为 null）
- `devices[].units`：该设备类型下的各个设备单元
- `scheduler.max_concurrent_tasks`：同时执行的最大任务数
- `scheduler.max_queue_size`：等待队列的最大长度；超出后提交返回 `QUOTA_EXCEEDED`

### 6.3 提交 Pipeline（创建任务）

```
POST /api/v1/tasks
```

根据 `Content-Type` 支持两种提交模式：

#### 模式一：文件上传

```
Content-Type: multipart/form-data
```

| 字段                  | 类型          | 必填  | 描述                                      |
| ------------------- | ----------- | --- | --------------------------------------- |
| `file`              | file        | 是   | 音频文件上传                                  |
| `pipeline`          | JSON string | 是   | Pipeline 定义（见[第 10 节](#10-pipeline-模型)） |
| `device_preference` | string      | 否   | 首选设备类型（`"npu"`、`"cuda"`、`"cpu"` 等）      |
| `device_index`      | int         | 否   | 指定设备索引（默认：自动）                           |
| `priority`          | int         | 否   | 优先级 1-10，默认 5（数值越大越优先）                  |
| `output_format`     | string      | 否   | `"wav"`（默认）                             |
| `output_package`    | string      | 否   | `"separate"`（默认）                        |
| `force_refresh`     | bool        | 否   | 跳过缓存，强制重新执行（默认：false）                   |
| `callback_url`      | string      | 否   | 预留，未来 Webhook 支持                        |

#### 模式二：URL 引用

```
Content-Type: application/json
```

```json
{
  "input_url": "https://example.com/song.wav",
  "pipeline": {
    "steps": [
      {
        "type": "split",
        "model": "htdemucs_6s",
        "params": {"shifts": 1},
        "model_params": {}
      }
    ]
  },
  "device_preference": "npu",
  "priority": 5,
  "output_format": "wav",
  "output_package": "separate",
  "force_refresh": false
}
```

模式一的所有字段均可作为 JSON 键使用，但 `file` 替换为 `input_url`。

#### Pipeline JSON 示例

**单步骤（仅 split）：**

```json
{
  "steps": [
    {
      "type": "split",
      "model": "htdemucs_6s",
      "params": {
        "shifts": 1,
        "overlap": 0.25
      },
      "model_params": {}
    }
  ]
}
```

**多步骤（split → midi → detect）：**

```json
{
  "steps": [
    {
      "type": "split",
      "model": "htdemucs_6s",
      "params": {"shifts": 1},
      "model_params": {}
    },
    {
      "type": "midi",
      "model": "basic-pitch",
      "input": {"from_step": 0, "stem": "vocals"},
      "params": {},
      "model_params": {}
    },
    {
      "type": "detect",
      "input": {"from_step": 0, "stem": "vocals"},
      "params": {},
      "model_params": {}
    }
  ]
}
```

#### 响应

**成功（201 Created）：**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "queued",
  "pipeline": { },
  "created_at": "2026-04-15T12:00:00Z"
}
```

**缓存命中（200 OK）：**

```json
{
  "task_id": "previous-task-id-from-cache",
  "status": "done",
  "cached": true,
  "created_at": "2026-04-15T11:00:00Z"
}
```

**错误响应：**

| HTTP | 错误码                      | 触发条件                    |
| ---- | ------------------------ | ----------------------- |
| 400  | INPUT_FORMAT_UNSUPPORTED | 音频格式不受支持                |
| 400  | INPUT_TOO_LARGE          | 文件超过 max_upload_bytes   |
| 400  | INPUT_CORRUPT            | 文件已损坏                   |
| 400  | PIPELINE_INVALID         | Pipeline 定义无效           |
| 429  | QUOTA_EXCEEDED           | 并发任务过多或队列已满             |
| 503  | SERVER_BUSY              | 服务端过载（包含 `retry_after`） |

### 6.4 获取任务状态（快照）

```
GET /api/v1/tasks/{task_id}
```

**响应（处理中）：**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "processing",
  "current_step": 0,
  "step_type": "split",
  "percent": 50,
  "completed_steps": [],
  "completed_urls": [],
  "created_at": "2026-04-15T12:00:00Z",
  "started_at": "2026-04-15T12:00:01Z"
}
```

**状态值：** `"queued"` | `"processing"` | `"done"` | `"partial_error"` | `"error"` | `"cancelled"`

**响应（已完成）：**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "done",
  "current_step": 2,
  "percent": 100,
  "completed_steps": [0, 1, 2],
  "completed_urls": [
    {
      "step_index": 0,
      "step_type": "split",
      "urls": [
        "/api/v1/files/a1b2c3d4/vocals.wav",
        "/api/v1/files/a1b2c3d4/drums.wav",
        "/api/v1/files/a1b2c3d4/bass.wav",
        "/api/v1/files/a1b2c3d4/other.wav",
        "/api/v1/files/a1b2c3d4/guitar.wav",
        "/api/v1/files/a1b2c3d4/piano.wav"
      ]
    },
    {
      "step_index": 1,
      "step_type": "midi",
      "urls": [
        "/api/v1/files/a1b2c3d4/vocals.mid"
      ]
    },
    {
      "step_index": 2,
      "step_type": "detect",
      "urls": []
    }
  ],
  "created_at": "2026-04-15T12:00:00Z",
  "started_at": "2026-04-15T12:00:01Z",
  "finished_at": "2026-04-15T12:05:30Z"
}
```

**响应（部分错误）：**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "partial_error",
  "completed_steps": [0, 2],
  "failed_steps": [1],
  "completed_urls": [
    {
      "step_index": 0,
      "step_type": "split",
      "urls": ["/api/v1/files/a1b2c3d4/vocals.wav", "/api/v1/files/a1b2c3d4/drums.wav", "/api/v1/files/a1b2c3d4/bass.wav", "/api/v1/files/a1b2c3d4/other.wav", "/api/v1/files/a1b2c3d4/guitar.wav", "/api/v1/files/a1b2c3d4/piano.wav"]
    },
    {
      "step_index": 2,
      "step_type": "detect",
      "urls": []
    }
  ],
  "errors": [
    {
      "step_index": 1,
      "error": {"code": "MODEL_LOAD_FAILED", "message": "basic-pitch model not found"}
    }
  ]
}
```

### 6.5 获取任务事件（SSE 流）

```
GET /api/v1/tasks/{task_id}/events
Accept: text/event-stream
```

见[第 7 节](#7-sse-事件流)。

### 6.6 下载文件

```
GET /api/v1/files/{task_id}/{filename}
```

**响应：** 带有适当 Content-Type 的二进制文件。

**支持的文件名：** 任务产生的任何文件（如 `vocals.wav`、`drums.flac`、`vocals.mid`）。

### 6.7 取消/删除任务

```
DELETE /api/v1/tasks/{task_id}
```

**响应：**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "cancelled",
  "message": "Task cancelled. Input and output files deleted."
}
```

取消正在运行的任务会停止执行并删除所有关联文件。删除已完成的任务会移除任务及其文件。

### 6.8 任务列表

```
GET /api/v1/tasks?status=processing&limit=20&offset=0
```

**查询参数：**

| 参数       | 类型     | 默认值 | 描述    |
| -------- | ------ | --- | ----- |
| `status` | string | all | 按状态筛选 |
| `limit`  | int    | 20  | 最大返回数 |
| `offset` | int    | 0   | 分页偏移  |

**响应：**

```json
{
  "tasks": [
    {"task_id": "...", "status": "processing", "current_step": 0, "percent": 50}
  ],
  "total": 1,
  "limit": 20,
  "offset": 0
}
```

---

## 7. SSE 事件流

### 7.1 连接

```
GET /api/v1/tasks/{task_id}/events
Accept: text/event-stream
```

服务端保持连接打开，按事件发生顺序推送。

### 7.2 事件类型

所有 SSE JSON 负载包含 `type` 字段，指示业务级事件类型。这与 SSE 协议层的 `event:` 行是分离的——`event:` 行负责传输路由，`type` 负责应用层分发。

#### 7.2.1 `partial_result` — 细粒度数据事件

每个独立输出项（如每个分轨）可用时触发。

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "type": "partial_result",
  "step_index": 0,
  "step_type": "split",
  "track": {
    "track_type": "audio",
    "stem": "vocals",
    "label": "Vocals",
    "url": "/api/v1/files/a1b2c3d4/vocals.wav",
    "format": "wav",
    "sample_rate": 44100,
    "duration": 180.5,
    "size_bytes": 63580444
  }
}
```

**当前实现说明：** 对于基于 demucs 的 split，所有分轨是一次性计算完成的。服务端快速连续发送 `partial_result` 事件（每个分轨一个），然后发送 `status: "completed"` 的 `progress` 事件。这种"伪流式"方式确保客户端代码以统一的流式处理逻辑编写，不受后端实现差异影响。

**未来的后端** 可能会在每个分轨独立完成时以真实时间间隔发送 `partial_result` 事件。

#### 7.2.2 `progress` — 步骤状态事件

步骤生命周期变更时触发：运行中、已完成或失败。

**运行中：**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "type": "progress",
  "step_index": 0,
  "step_type": "split",
  "status": "running",
  "percent": 50
}
```

**已完成：**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "type": "progress",
  "step_index": 0,
  "step_type": "split",
  "status": "completed",
  "urls": [
    "/api/v1/files/a1b2c3d4/vocals.wav",
    "/api/v1/files/a1b2c3d4/drums.wav",
    "/api/v1/files/a1b2c3d4/bass.wav",
    "/api/v1/files/a1b2c3d4/other.wav",
    "/api/v1/files/a1b2c3d4/guitar.wav",
    "/api/v1/files/a1b2c3d4/piano.wav"
  ]
}
```

**失败：**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "type": "progress",
  "step_index": 0,
  "step_type": "split",
  "status": "failed",
  "error": {
    "code": "DEVICE_OOM",
    "message": "NPU out of memory",
    "details": {
      "device": "npu:0",
      "required_mb": 2048,
      "available_mb": 1024
    }
  }
}
```

**`status` 枚举：** `"running"` | `"completed"` | `"failed"`

#### 7.2.3 `final_result` — 终结事件

整个 Pipeline 结束时触发一次（成功、部分错误、错误或已取消）。

**成功：**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "type": "final_result",
  "status": "done",
  "urls": [
    {
      "step_index": 0,
      "step_type": "split",
      "urls": [
        "/api/v1/files/a1b2c3d4/vocals.wav",
        "/api/v1/files/a1b2c3d4/drums.wav",
        "/api/v1/files/a1b2c3d4/bass.wav",
        "/api/v1/files/a1b2c3d4/other.wav",
        "/api/v1/files/a1b2c3d4/guitar.wav",
        "/api/v1/files/a1b2c3d4/piano.wav"
      ]
    },
    {
      "step_index": 1,
      "step_type": "midi",
      "urls": ["/api/v1/files/a1b2c3d4/vocals.mid"]
    },
    {
      "step_index": 2,
      "step_type": "detect",
      "urls": []
    }
  ]
}
```

**部分错误：**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "type": "final_result",
  "status": "partial_error",
  "completed_steps": [0, 2],
  "failed_steps": [1],
  "urls": [
    {
      "step_index": 0,
      "step_type": "split",
      "urls": ["/api/v1/files/a1b2c3d4/vocals.wav", "..."]
    },
    {
      "step_index": 2,
      "step_type": "detect",
      "urls": []
    }
  ],
  "errors": [
    {
      "step_index": 1,
      "error": {"code": "MODEL_LOAD_FAILED", "message": "basic-pitch model not found"}
    }
  ]
}
```

**错误：**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "type": "final_result",
  "status": "error",
  "error": {
    "code": "DEVICE_OOM",
    "message": "NPU out of memory"
  }
}
```

**已取消：**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "type": "final_result",
  "status": "cancelled"
}
```

**`status` 枚举：** `"done"` | `"partial_error"` | `"error"` | `"cancelled"`

### 7.3 事件流示例

单步骤 Pipeline（仅 split）：

```
→ partial_result  {type:"partial_result", step_index:0, stem:"vocals", url:"..."}
→ partial_result  {type:"partial_result", step_index:0, stem:"drums", url:"..."}
→ partial_result  {type:"partial_result", step_index:0, stem:"bass", url:"..."}
→ partial_result  {type:"partial_result", step_index:0, stem:"other", url:"..."}
→ partial_result  {type:"partial_result", step_index:0, stem:"guitar", url:"..."}
→ partial_result  {type:"partial_result", step_index:0, stem:"piano", url:"..."}
→ progress        {type:"progress", step_index:0, status:"completed", urls:[...]}
→ final_result    {type:"final_result", status:"done", urls:[{step_index:0, step_type:"split", urls:[...]}]}
```

多步骤 Pipeline（split → midi → detect）：

```
→ progress        {type:"progress", step_index:0, status:"running", percent:30}
→ progress        {type:"progress", step_index:0, status:"running", percent:60}
→ partial_result  {type:"partial_result", step_index:0, stem:"vocals", url:"..."}
→ partial_result  {type:"partial_result", step_index:0, stem:"drums", url:"..."}
→ ...             （其余分轨）
→ progress        {type:"progress", step_index:0, status:"completed", urls:[...]}
→ progress        {type:"progress", step_index:1, status:"running", percent:50}
→ progress        {type:"progress", step_index:1, status:"completed", urls:["...vocals.mid"]}
→ progress        {type:"progress", step_index:2, status:"running", percent:40}
→ progress        {type:"progress", step_index:2, status:"completed", urls:[]}
→ final_result    {type:"final_result", status:"done", urls:[{step_index:0, ...}, {step_index:1, ...}, {step_index:2, ...}]}
```

---

## 8. 能力声明

### 8.1 能力对象

每个 AI 能力声明如下：

```json
{
  "id": "split",
  "label": "Stem Separation",
  "description": "Separate audio into individual stems",
  "status": "implemented",
  "models": ["htdemucs", "htdemucs_6s"],
  "default_model": "htdemucs_6s",
  "param_defs": []
}
```

- `id`（string）：能力标识符，用于 Pipeline 步骤的 `type`
- `status`（string）：`"implemented"` | `"not_implemented"`
  - `"implemented"`：功能完整，可正常使用
  - `"not_implemented"`：已规划但尚未实现；客户端应显示为禁用/灰色，提示"尚未实现"
- `models`（string[]）：该能力可用的模型名称
- `default_model`（string | null）：默认模型选择
- `param_defs`（array）：参数 Schema 定义（见[第 9 节](#9-参数-schemaui-驱动)）

### 8.2 标准能力 ID

| ID         | 标签      | 描述            |
| ---------- | ------- | ------------- |
| `split`    | 分轨分离    | 将音频分离为独立分轨    |
| `midi`     | MIDI 转写 | 将音频转换为 MIDI   |
| `generate` | AI 作曲   | AI 辅助 MIDI 生成 |
| `detect`   | 音符检测    | 从音频中检测音符      |

---

## 9. 参数 Schema（UI 驱动）

### 9.1 概述

服务端返回参数 Schema，客户端据此动态生成 UI 控件。这允许在不更新客户端的情况下添加新参数。

### 9.2 Schema 定义

每个参数定义如下：

```json
{
  "key": "shifts",
  "type": "int",
  "label": "Shifts",
  "description": "Number of random shifts for equivariant stabilization",
  "default": 1,
  "min": 1,
  "max": 20,
  "group": "advanced"
}
```

### 9.3 参数类型 → Qt Widget 映射

| Schema 类型    | Qt Widget        | 附加字段                          |
| ------------ | ---------------- | ----------------------------- |
| `int`        | QSpinBox         | `min`、`max`、`step`            |
| `float`      | QDoubleSpinBox   | `min`、`max`、`step`、`decimals` |
| `string`     | QLineEdit        | `placeholder`、`pattern`       |
| `enum`       | QComboBox        | `options: [{value, label}]`   |
| `bool`       | QCheckBox        | —                             |
| `multi_enum` | QListWidget（可勾选） | `options: [{value, label}]`   |

### 9.4 完整示例

```json
{
  "id": "split",
  "param_defs": [
    {
      "key": "model",
      "type": "enum",
      "label": "Model",
      "description": "Separation model to use",
      "default": "htdemucs_6s",
      "options": [
        {"value": "htdemucs", "label": "HTDemucs (4 stems)"},
        {"value": "htdemucs_6s", "label": "HTDemucs 6-stem"}
      ],
      "group": "basic"
    },
    {
      "key": "shifts",
      "type": "int",
      "label": "Shifts",
      "description": "Number of random shifts for better quality",
      "default": 1,
      "min": 1,
      "max": 20,
      "step": 1,
      "group": "advanced"
    },
    {
      "key": "overlap",
      "type": "float",
      "label": "Overlap",
      "description": "Overlap between chunks",
      "default": 0.25,
      "min": 0.0,
      "max": 0.99,
      "step": 0.01,
      "decimals": 2,
      "group": "advanced"
    },
    {
      "key": "stems",
      "type": "multi_enum",
      "label": "Stems to Extract",
      "description": "Select which stems to separate",
      "default": ["vocals", "drums", "bass", "other", "guitar", "piano"],
      "options": [
        {"value": "vocals", "label": "Vocals"},
        {"value": "drums", "label": "Drums"},
        {"value": "bass", "label": "Bass"},
        {"value": "other", "label": "Other"},
        {"value": "guitar", "label": "Guitar"},
        {"value": "piano", "label": "Piano"}
      ],
      "group": "basic"
    }
  ]
}
```

### 9.5 分组排序

参数按分组进行 UI 布局。标准分组：

| 分组             | 显示顺序 | 描述          |
| -------------- | ---- | ----------- |
| `basic`        | 1    | 默认显示的主要参数   |
| `advanced`     | 2    | 隐藏在"高级"开关后  |
| `experimental` | 3    | 隐藏在"实验性"开关后 |

---

## 10. Pipeline 模型

### 10.1 概述

所有任务都是 Pipeline。单步骤任务是只有一个步骤的 Pipeline。

### 10.2 Pipeline 定义

```json
{
  "steps": [
    {
      "type": "split",
      "model": "htdemucs_6s",
      "input": null,
      "params": {"shifts": 1},
      "model_params": {}
    },
    {
      "type": "midi",
      "model": "basic-pitch",
      "input": {"from_step": 0, "stem": "vocals"},
      "params": {},
      "model_params": {}
    }
  ]
}
```

### 10.3 步骤字段

| 字段             | 类型     | 必填   | 描述                                                       |
| -------------- | ------ | ---- | -------------------------------------------------------- |
| `type`         | string | 是    | 能力 ID（如 `"split"`、`"midi"`）                              |
| `model`        | string | 否    | 模型名称（默认为能力的 default_model）                               |
| `input`        | object | null | 输入源规格（见下文）                                               |
| `params`       | object | 否    | 参数值，以 `param_defs[].key` 为键（见[第 9 节](#9-参数-schemaui-驱动)） |
| `model_params` | object | 否    | 模型特定参数（直接透传，服务端不校验）                                      |

### 10.4 输入规格

步骤 0（第一步）始终接收上传的音频文件，其 `input` 必须为 `null`。

后续步骤必须显式指定输入：

```json
{
  "from_step": 0,
  "stem": "vocals"
}
```

- `from_step`（int，必填）：源步骤索引（从 0 开始）
- `stem`（string，可选）：源步骤输出中的特定分轨
  - 若省略，使用源步骤的完整输出
  - 对于 `midi` 和 `detect` 能力，`stem` 指定要处理的音频分轨

### 10.5 Pipeline 执行规则

1. 步骤按顺序依次执行
2. 若某步骤失败，依赖该步骤的后续步骤被跳过
3. 不依赖失败步骤的后续步骤继续执行
4. 依赖关系由 `input.from_step` 决定：若指向失败步骤，则依赖步骤被跳过
5. Pipeline 在所有可能的步骤执行完毕或跳过后终止

### 10.6 Pipeline 进度计算

- **步骤进度**：每个步骤报告自身的 `percent`（0-100）
- **总进度**：`((已完成步骤数 * 100) + 当前步骤百分比) / 总步骤数`

---

## 11. 错误码体系

### 11.1 错误码格式

`类别_具体` — 大写字母、下划线分隔、类别前缀。

### 11.2 错误类别

#### INPUT（4xx）

| 错误码                            | HTTP | 描述                      |
| ------------------------------ | ---- | ----------------------- |
| `INPUT_FORMAT_UNSUPPORTED`     | 400  | 音频格式不受任何后端支持            |
| `INPUT_TOO_LARGE`              | 400  | 文件大小超过 max_upload_bytes |
| `INPUT_CORRUPT`                | 400  | 文件已损坏或无法解码              |
| `INPUT_SAMPLERATE_UNSUPPORTED` | 400  | 采样率无法处理（预留，当前自动重采样）     |
| `INPUT_CHANNEL_UNSUPPORTED`    | 400  | 声道布局不受支持                |

#### MODEL（4xx/5xx）

| 错误码                  | HTTP | 描述                |
| -------------------- | ---- | ----------------- |
| `MODEL_NOT_FOUND`    | 400  | 请求的模型不存在          |
| `MODEL_LOAD_FAILED`  | 500  | 模型加载失败（文件损坏、不兼容等） |
| `MODEL_INCOMPATIBLE` | 400  | 模型与所选设备不兼容        |

#### DEVICE（5xx）

| 错误码                  | HTTP | 描述       |
| -------------------- | ---- | -------- |
| `DEVICE_OOM`         | 503  | 设备内存不足   |
| `DEVICE_UNAVAILABLE` | 503  | 请求的设备不可用 |
| `DEVICE_TIMEOUT`     | 504  | 设备操作超时   |
| `DEVICE_ERROR`       | 500  | 通用设备错误   |

#### SERVER（5xx）

| 错误码                   | HTTP | 描述      |
| --------------------- | ---- | ------- |
| `SERVER_BUSY`         | 503  | 服务端过载   |
| `SERVER_ERROR`        | 500  | 内部服务端错误 |
| `SERVER_CONFIG_ERROR` | 500  | 服务端配置错误 |

#### QUOTA（429）

| 错误码                | HTTP | 描述          |
| ------------------ | ---- | ----------- |
| `QUOTA_EXCEEDED`   | 429  | 并发任务过多或队列已满 |
| `QUOTA_RATE_LIMIT` | 429  | 请求频率过高      |

#### PIPELINE（400）

| 错误码                            | HTTP | 描述            |
| ------------------------------ | ---- | ------------- |
| `PIPELINE_INVALID`             | 400  | Pipeline 定义无效 |
| `PIPELINE_CIRCULAR_DEPENDENCY` | 400  | 步骤输入形成循环依赖    |
| `PIPELINE_MISSING_INPUT`       | 400  | 步骤需要输入但未指定    |

#### CAPABILITY（400）

| 错误码                          | HTTP | 描述        |
| ---------------------------- | ---- | --------- |
| `CAPABILITY_NOT_IMPLEMENTED` | 400  | 请求的能力尚未实现 |

#### AUTH（401/403）

| 错误码             | HTTP | 描述              |
| --------------- | ---- | --------------- |
| `AUTH_REQUIRED` | 401  | 需要 API Key 但未提供 |
| `AUTH_INVALID`  | 403  | 提供的 API Key 无效  |

### 11.3 带重试信息的错误响应

对于 `QUOTA_EXCEEDED` 和 `SERVER_BUSY`：

```json
{
  "error": {
    "code": "QUOTA_EXCEEDED",
    "message": "Too many concurrent tasks",
    "details": {
      "retry_after": 30,
      "queue_position": 3,
      "max_concurrent": 4,
      "current_tasks": 4
    }
  }
}
```

---

## 12. 文件传输

### 12.1 上传

- 方法：`POST /api/v1/tasks`
- **文件上传模式**：`Content-Type: multipart/form-data`，字段名 `file`
- **URL 引用模式**：`Content-Type: application/json`，字段 `input_url`
- 最大大小：在 `/api/v1/capabilities` → `max_upload_bytes` 中声明
- 支持格式：由服务端能力决定（通常为 wav、mp3、flac、ogg、m4a、aac）

### 12.2 下载

- URL 格式：`/api/v1/files/{task_id}/{filename}`
- URL 相对于服务端根路径
- Content-Type 根据文件扩展名自动设置
- 文件直接提供（无 base64 编码）

### 12.3 输出格式

通过 `output_format` 参数按任务指定：

| 格式     | 扩展名     | 描述          |
| ------ | ------- | ----------- |
| `wav`  | `.wav`  | 无压缩 PCM（默认） |
| `flac` | `.flac` | 无损压缩        |
| `mp3`  | `.mp3`  | 有损压缩        |

### 12.4 输出打包

通过 `output_package` 参数按任务指定：

| 打包方式       | 描述                    |
| ---------- | --------------------- |
| `separate` | 独立文件，每个通过其 URL 下载（默认） |
| `zip`      | 单个 ZIP 压缩包包含所有输出文件    |

### 12.5 断点续传（预留）

HTTP `Range` 头的断点续传支持预留，当前版本不支持。

---

## 13. 服务发现

### 13.1 手动配置

主要方式：RMMS 配置文件（`rmmsrc.xml` → `<ai>` 节点）存储 AI Server URL：

```xml
<ai>
  <server_url>http://192.168.1.100:8420</server_url>
  <api_key>optional-key</api_key>
</ai>
```

### 13.2 mDNS 自动发现

AI Server 通过 mDNS/DNS-SD 广播：

- **服务类型**：`_rmms-ai._tcp`
- **端口**：8420（默认）
- **TXT 记录**：
  - `protocol_version=1.0.0`
  - `devices=npu,cuda`

RMMS 可自动发现局域网内的 AI Server。发现的服务器与手动配置的一同显示在 AI 配置对话框中。

---

## 14. 幂等性与缓存

### 14.1 缓存键

缓存键由以下内容计算：

- 文件内容哈希（上传文件的 SHA-256）
- Pipeline 定义哈希（规范 JSON 的 SHA-256）
- 输出格式和打包设置

### 14.2 缓存行为

- 相同输入 + 相同 Pipeline + 相同设置 → 立即返回缓存结果
- 缓存响应包含 `"cached": true`
- 客户端可设置 `force_refresh: true` 跳过缓存

### 14.3 缓存失效

缓存失效条件：

- 服务端重启（内存缓存）
- 通过 `DELETE /api/v1/tasks/{task_id}` 删除任务
- 服务端模型版本变更

---

## 15. SSE 重连策略

### 15.1 设计决策

**不支持 `Last-Event-ID` 事件回放。** 事件回放需要服务端缓冲事件，导致内存膨胀和架构复杂化。

### 15.2 REST 快照 + SSE 增量

SSE 断连时：

1. **检测**：客户端检测到 SSE 连接丢失
2. **快照**：客户端调用 `GET /api/v1/tasks/{task_id}` 获取当前状态
3. **恢复 UI**：客户端使用快照恢复进度条、已完成分轨列表等
4. **重连**：客户端打开新的 SSE 连接到 `GET /api/v1/tasks/{task_id}/events`
5. **增量**：客户端仅接收重连后发出的事件

### 15.3 快照要求

`GET /api/v1/tasks/{task_id}` 必须返回足够的状态以完整恢复 UI：

- 当前步骤索引和类型
- 当前进度百分比
- 已完成步骤及其输出 URL 列表（步骤结果数组）
- 失败步骤及错误信息（步骤错误数组）
- 任务状态

---

## 16. 轨道类型系统

### 16.1 轨道类型

| 类型       | 描述                   | 当前状态           |
| -------- | -------------------- | -------------- |
| `audio`  | 音频波形文件（wav/flac/mp3） | 已实现            |
| `midi`   | MIDI 文件（.mid）或音符事件数据 | 已实现（文件），预留（事件） |
| `hybrid` | 波形 + 音高信息组合轨道        | 未实现            |

### 16.2 混合轨道（预留）

`hybrid` 轨道类型为未来 RMMS 轨道格式预留，该格式将结合波形和音高信息。

**当前行为：** 当能力将产生 `hybrid` 轨道时，服务端返回：

```json
{
  "track_type": "hybrid",
  "status": "not_implemented",
  "message": "Hybrid tracks are not yet supported"
}
```

**客户端行为：** 显示"混合轨道尚未支持"并跳过该轨道。

**未来扩展：** 实现后，`hybrid` 轨道将包含：

```json
{
  "track_type": "hybrid",
  "status": "implemented",
  "audio_url": "/api/v1/files/{task_id}/vocals.wav",
  "midi_url": "/api/v1/files/{task_id}/vocals.mid",
  "note_events": [ ]
}
```

---

## 17. 未来预留

以下功能在协议设计中预留。字段和事件类型已定义但尚未实现。

### 17.1 预留 SSE 事件类型

| 事件               | 描述          | 状态       |
| ---------------- | ----------- | -------- |
| `partial_result` | 逐分轨流式       | 已实现（伪流式） |
| `progress`       | 步骤状态        | 已实现      |
| `final_result`   | Pipeline 终结 | 已实现      |

### 17.2 预留 API 端点

| 端点                               | 描述      | 状态  |
| -------------------------------- | ------- | --- |
| `POST /api/v1/tasks/{id}/pause`  | 暂停任务    | 预留  |
| `POST /api/v1/tasks/{id}/resume` | 恢复暂停的任务 | 预留  |

### 17.3 预留请求字段

| 字段             | 端点          | 描述                | 状态  |
| -------------- | ----------- | ----------------- | --- |
| `callback_url` | POST /tasks | 完成通知的 Webhook URL | 预留  |
| `Range` 头      | POST /tasks | 断点续传              | 预留  |

### 17.4 预留响应字段

| 字段            | 端点   | 描述                   | 状态  |
| ------------- | ---- | -------------------- | --- |
| `note_events` | 轨道对象 | 内联音符事件数据（替代 MIDI 文件） | 预留  |
| `hybrid` 数据   | 轨道对象 | 音频 + MIDI 组合轨道数据     | 预留  |

---

## 18. 变更日志

| 版本          | 日期         | 变更       |
| ----------- | ---------- | -------- |
| 1.0.0-draft | 2026-04-15 | 初始协议规范草案 |

---

## 附录 A：完整 API 端点汇总

| 方法     | 路径                                   | 描述                                                  |
| ------ | ------------------------------------ | --------------------------------------------------- |
| GET    | `/api/v1/health`                     | 健康检查                                                |
| GET    | `/api/v1/capabilities`               | 服务端能力、设备、参数 Schema、调度器配置                            |
| POST   | `/api/v1/tasks`                      | 提交 Pipeline（multipart/form-data 或 application/json） |
| GET    | `/api/v1/tasks`                      | 任务列表                                                |
| GET    | `/api/v1/tasks/{task_id}`            | 获取任务状态（快照）                                          |
| GET    | `/api/v1/tasks/{task_id}/events`     | SSE 事件流                                             |
| DELETE | `/api/v1/tasks/{task_id}`            | 取消/删除任务                                             |
| GET    | `/api/v1/files/{task_id}/{filename}` | 下载输出文件                                              |

## 附录 B：SSE 事件汇总

| 事件               | 触发条件        | 关键字段                                               |
| ---------------- | ----------- | -------------------------------------------------- |
| `partial_result` | 单个输出项就绪     | `step_index`、`track`                               |
| `progress`       | 步骤状态变更      | `step_index`、`status`、`percent` / `urls` / `error` |
| `final_result`   | Pipeline 完成 | `status`、`urls`（步骤结果数组）、`errors`（步骤错误数组）           |

## 附录 C：客户端实现清单

RMMS 客户端必须实现：

- [ ] 解析 `/api/v1/capabilities` 并根据参数 Schema 生成 UI
- [ ] 将不可用能力显示为禁用状态，提示"尚未实现"
- [ ] 将不可用设备显示安装提示
- [ ] 通过 `POST /api/v1/tasks` 提交 Pipeline（支持 multipart 和 JSON 两种模式）
- [ ] 连接 SSE 流并处理全部三种事件类型
- [ ] 使用 JSON 负载中的 `type` 字段进行事件分发（而非 SSE `event:` 行）
- [ ] 处理 `partial_result`，在分轨到达时导入
- [ ] 处理 `status: "failed"` 的 `progress`，显示错误信息
- [ ] 处理 `status: "partial_error"` 的 `final_result`，显示部分结果
- [ ] 将 `urls` 和 `errors` 解析为步骤结果/步骤错误对象数组
- [ ] SSE 断连时实现 REST 快照恢复
- [ ] 通过 `DELETE /api/v1/tasks/{task_id}` 支持任务取消/删除
- [ ] 忽略未知 JSON 字段（前向兼容）
- [ ] 忽略未知 `track_type` 值并通知用户
- [ ] 优雅处理 `CAPABILITY_NOT_IMPLEMENTED` 错误
