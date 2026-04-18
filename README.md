# rmms-ai-server

RMMS 项目的官方 AI 推理后端，基于 Demucs 实现音频多轨分离，遵循 **RMMS AI 服务端协议 v1.0.0-alpha**。

> 协议规范详见 [rmms-ai-protocol](https://github.com/The-Sunder-islands/rmms-ai-protocol)

## 概述

rmms-ai-server 是 **RMMS (Rip & MultiMedia Studio)** 的独立 AI 后端，负责音频分轨分离、AI 推理任务调度与文件处理。它通过 **REST + SSE** 与 RMMS 客户端通信，实现**后端可远程部署、可插拔替换、跨设备兼容**的架构设计。

## 已经完整实现的功能

- 遵循 **RMMS AI 服务端协议 v1.0.0-alpha**，与前端无缝对接
- 基于 **Demucs** 实现高质量人声/鼓/贝斯/其他分轨分离
- 在 PyTorch 环境部署正确的情况下，支持多设备自动适配（CUDA / NPU / XPU / MPS / CPU）
- **Schema 驱动参数**，前端可动态生成 UI
- **SSE 实时进度推送**，断连可自动恢复
- 支持任务缓存、并发调度、文件管理
- 可本地运行、可局域网部署、可云端部署
- 提供 Docker 一键部署，开箱即用

## 架构

- 框架：FastAPI
- 推理引擎：Demucs（htdemucs / htdemucs_6s）
- 通信：REST API + SSE 事件流
- 协议：RMMS AI Server Protocol v1.0.0-alpha
- 部署：Python + Uvicorn + Docker

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/The-Sunder-islands/rmms-ai-server.git
cd rmms-ai-server
```

### 2. 安装

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate.ps1   # Windows

# ⚠️ PyTorch 需要根据你的设备手动安装对应版本：
# CUDA:
#   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126
# ROCm:
#   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm7.2
# 否则默认安装 CPU 版本

# 安装服务端及依赖
pip install -e .
```

### 3. 启动

```bash
# 直接启动（默认端口 8420）
rmms-ai-server

# 或通过 Python 模块启动
python -m rmms_ai_server

# 自定义参数
rmms-ai-server --port 9000 --log-level debug --no-mdns
```

启动成功后，AI 服务默认运行在 `http://127.0.0.1:8420`

- 测试页面：`http://127.0.0.1:8420/`
- API 文档：`http://127.0.0.1:8420/docs`

### 4. Docker 部署

```bash
# 构建镜像
docker build -t rmms-ai-server .

# 运行（GPU）
docker run -d --name rmms-ai-server --gpus all \
  -p 8420:8420 \
  -v /path/to/models:/data/models \
  rmms-ai-server

# 运行（CPU）
docker run -d --name rmms-ai-server \
  -p 8420:8420 \
  rmms-ai-server
```

## 与 RMMS 客户端对接

本服务端已完整实现 RMMS AI 服务端协议，任何遵循该协议的客户端均可直接对接。RMMS DAW 客户端的集成开发正在进行中。

## 协议支持

本后端完整实现以下协议能力：

- 健康检查 `/api/v1/health`
- 能力发现 `/api/v1/capabilities`
- 任务提交 `/api/v1/tasks`
- SSE 实时进度推送
- 分轨结果流式返回
- 任务取消与文件清理
- 输入校验与标准错误码
- 前向兼容设计

## 支持模型

| 模型 | 分轨数 | 说明 |
|------|--------|------|
| htdemucs | 4 | drums / bass / other / vocals |
| htdemucs_6s | 6 | drums / bass / other / vocals / guitar / piano |

## 实验性功能

以下功能已有引擎实现，尚未在默认 Pipeline 中暴露：

- MIDI 转写（basic-pitch）
- 音符检测
- AI 辅助处理

安装可选依赖以启用：

```bash
pip install -e ".[midi]"   # MIDI 转写
pip install -e ".[npu]"    # 华为 NPU 支持
pip install -e ".[fftw]"   # FFT 加速
```

## 项目状态

v1.0.0a1 — 早期可用版本

- 已实现：**音频分轨分离完整流程**
- 已对接：RMMS AI 服务端协议完整实现
- 稳定运行：本地 / 局域网 / 云端部署

## 许可证

[MIT](LICENSE)
