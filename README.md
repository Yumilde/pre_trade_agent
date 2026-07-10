# 融资融券交易试算 Agent

基于 LangGraph 构建的融资融券（两融）交易试算智能 Agent，支持多轮对话式交易计算与场景分析。

## 快速开始

### 1. 拉取 Docker 镜像

```bash
docker pull yumilde/pre-trade-agent@sha256:7abd1dad3e44f47cffb69fcd5ef353a58bf5858ce372b24e7eb0b16a368d89e4
```

### 2. 克隆项目

```bash
git clone https://github.com/Yumilde/pre_trade_agent.git
# 或 Gitee
git clone https://gitee.com/ln4444/pre_trade_agent.git
```

### 3. 进入项目目录

```bash
cd pre_trade_agent
```

### 4. 创建环境变量文件

```bash
cp .env_example .env
```

### 5. 编辑 .env 配置文件

```bash
vim .env
```

填入配置 → 按 `ESC` 确保退出编辑模式 → 输入 `:wq` 后回车，保存并退出。

> 如果你不熟悉 vim，也可以用其他编辑器打开，如 `nano .env` 或 `code .env`。

### 6. 启动 LangGraph 服务

```bash
langgraph up
```

### 7. 打开前端页面

在浏览器中打开 `pre_trade_agent/frontend/index.html`，即可开始使用。

---

## .env 配置说明

`.env` 文件是项目的核心配置文件，以下逐项说明：

### LLM 基础配置（必填）

```env
OPENAI_MODEL=<your-model-name>          # 使用的 LLM 模型名称（例如 deepseek-v4-flash）
OPENAI_BASE_URL=<your-llm-api-url>     # LLM API 地址（例如 https://api.deepseek.com）
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  # LLM API 密钥
```

> 支持任意兼容 OpenAI 接口的服务，如 DeepSeek、OpenAI、其他国产模型等。修改 `OPENAI_BASE_URL` 和 `OPENAI_MODEL` 即可切换。

### 代理绕过（内网地址直连）

```env
INTRANET_HOSTS=<your-internal-ips>,127.0.0.1,localhost
NO_PROXY=<your-internal-ips>,127.0.0.1,localhost
no_proxy=<your-internal-ips>,127.0.0.1,localhost
```

如果你的 MCP 服务或其他依赖部署在内网，在此填写内网 IP 以绕过代理。仅使用本地服务时保留 `127.0.0.1,localhost` 即可。

### MCP 服务地址（融资融券业务工具）

```env
MCP_URL=http://<your-mcp-server-url>        # MCP 服务端地址
MCP_API_KEY=eyJ...                          # MCP 认证 JWT Token
MCP_API_HEADER=Authorization                # 认证头字段名
MCP_API_PREFIX=Bearer                       # 认证头前缀
MCP_TOOLS_TIMEOUT_SECONDS=15                # 工具调用超时（秒）
MCP_REQUIRED=false                          # 是否强制要求 MCP 可用（false=可选）
```

### LangSmith 可观测性（可选，推荐开启）

```env
LANGSMITH_TRACING=true                                  # 是否开启链路追踪
LANGSMITH_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxxxxxx      # LangSmith API Key
LANGSMITH_ENDPOINT=https://api.smith.langchain.com       # LangSmith 端点
LANGSMITH_PROJECT=pre_trade_agent                       # 项目名称
```

### 沙箱后端

```env
SANDBOX_PROVIDER=local          # 沙箱类型：local=本地文件系统 / daytona=云端沙箱
SANDBOX_SEED_DIR=./sandbox_seed # 沙箱种子数据目录
RULES_DIR=./rules               # 规则文件目录
```

### Daytona 云端沙箱（仅 SANDBOX_PROVIDER=daytona 时必填）

```env
DAYTONA_API_KEY=
DAYTONA_API_URL=https://app.daytona.io/api
DAYTONA_SNAPSHOT_ID=
DAYTONA_MAX_SANDBOXES=10
DAYTONA_ACTIVE_WINDOW_SECONDS=300
```

### 其他配置

```env
DEEPAGENT_DEBUG=false     # 是否开启调试模式
MAX_CLARIFY_ROUNDS=4      # 最大澄清轮次
```

---

## 项目结构

```
pre_trade_agent/
├── agent.py              # LangGraph Agent 主入口
├── prompt.py             # Prompt 模板
├── daytona_sandbox.py    # Daytona 沙箱适配
├── langgraph.json        # LangGraph 配置
├── pyproject.toml        # Python 项目配置
├── requirements.txt      # Python 依赖
├── .env_example          # 环境变量模板
├── rules/                # 业务规则文件
├── sandbox_seed/         # 沙箱种子数据
│   ├── formulas/         # 计算公式
│   ├── scenarios/        # 业务场景
│   ├── outputs/          # 输出模板
│   └── scratch/          # 临时工作区
└── frontend/             # 前端页面
    ├── index.html        # 主页面
    └── server.py         # 前端服务
```
