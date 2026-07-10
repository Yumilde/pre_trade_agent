import asyncio
import json
import os
import traceback
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pulp
from deepagents import create_deep_agent, SubAgent
from deepagents.backends import LocalShellBackend
from dotenv import load_dotenv
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from daytona_sandbox import PerRunDaytonaBackend, SandboxQuotaManager, DAYTONA_SANDBOX_HOME
from prompt import get_system_prompt, _PROMPT_REWRITER_PROMPT_TEMPLATE


# 代理环境变量配置
_INTRANET_HOSTS = os.environ.get("INTRANET_HOSTS", "127.0.0.1,localhost")

def _configure_proxy_env() -> None:
    for key in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(key, "")
        missing = [h for h in _INTRANET_HOSTS.split(",") if h not in existing]
        if missing:
            os.environ[key] = existing + ("," if existing else "") + ",".join(missing)

_configure_proxy_env()


# ---------------------------------------------------------------------------
# 工具裁剪中间件
# ---------------------------------------------------------------------------
class ExcludeToolsMiddleware(AgentMiddleware):
    """在模型看到工具前过滤掉指定名字的工具。

    deepagents 的 create_deep_agent 会无条件注入 TodoListMiddleware（提供 write_todos），
    而本 Agent 的规划/进度跟踪已统一收敛到受反思护栏监管的 reflective_thinking，
    write_todos 与之功能重复且不受护栏约束。此中间件将其从模型可见工具中剔除，
    避免出现第二条不受监管的规划通路造成 run 间漂移。
    """

    def __init__(self, excluded: frozenset[str]) -> None:
        super().__init__()
        self._excluded = excluded

    @staticmethod
    def _tool_name(tool) -> Optional[str]:
        if isinstance(tool, dict):
            name = tool.get("name")
            return name if isinstance(name, str) else None
        name = getattr(tool, "name", None)
        return name if isinstance(name, str) else None

    def _filter(self, request):
        if self._excluded:
            filtered = [t for t in request.tools if self._tool_name(t) not in self._excluded]
            return request.override(tools=filtered)
        return request

    def wrap_model_call(self, request, handler):
        return handler(self._filter(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._filter(request))


# prompt_rewriter 子代理（问题重写与歧义检测，LLM 驱动）
# 模板定义见 prompt.py 的 _PROMPT_REWRITER_PROMPT_TEMPLATE。

def _load_rules_sync(rules_dir: str) -> Dict[str, str]:
    """同步读取本地 rules/ 目录下的规则 JSON（供 asyncio.to_thread 调用）。"""
    base = Path(rules_dir)
    commonsense_parts: List[str] = []
    dimension_parts: List[str] = []

    commonsense_dir = base / "commonsense"
    if commonsense_dir.is_dir():
        for fp in sorted(commonsense_dir.glob("*.json")):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            commonsense_parts.append(json.dumps(data, ensure_ascii=False, indent=2))

    dimension_dir = base / "dimensions"
    if dimension_dir.is_dir():
        for fp in sorted(dimension_dir.glob("*.json")):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            dimension_parts.append(json.dumps(data, ensure_ascii=False, indent=2))

    return {
        "commonsense_rules": "\n\n".join(commonsense_parts) or "（未找到常识规则文件）",
        "dimension_rules": "\n\n".join(dimension_parts) or "（未找到维度定义文件）",
    }


async def _load_rules(rules_dir: str) -> Dict[str, str]:
    return await asyncio.to_thread(_load_rules_sync, rules_dir)


async def _build_prompt_rewriter_subagent(rules_dir: str) -> SubAgent:
    """构建 prompt_rewriter 子代理规格，将外置规则注入其 system_prompt。"""
    rules = await _load_rules(rules_dir)
    system_prompt = _PROMPT_REWRITER_PROMPT_TEMPLATE.format(
        commonsense_rules=rules["commonsense_rules"],
        dimension_rules=rules["dimension_rules"],
    )
    return {
        "name": "prompt_rewriter",
        "description": (
            "问题重写与歧义检测子代理。主代理在第一轮 reflective_thinking 之前，"
            "必须先通过 task 调用本子代理：传入用户原始问题（及上一轮澄清上下文，如有），"
            "本子代理返回重写后的问题或 [CLARIFY] 澄清模板。"
        ),
        "system_prompt": system_prompt,
        "tools": [],
    }


# ---------------------------------------------------------------------------
# reflective_thinking
# ---------------------------------------------------------------------------
@dataclass
class ReflectiveGuardResult:
    """反思守卫校验结果，_validate_reflective_guard() 的返回类型"""
    ok: bool
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# optimize_trading_plan
# ---------------------------------------------------------------------------
class StockOptimizationInput(BaseModel):
    """单只股票优化的输入参数模型，并自动校验取值合法性"""
    stock_code: str = Field(..., description="股票代码")
    stock_price: float = Field(..., gt=0, description="当前股价")
    margin_ratio: float = Field(1.0, gt=0, description="融资保证金比例，100%填1.0")
    weight: float = Field(1.0, gt=0, description="目标函数权重，越高越优先")
    existing_position_value: float = Field(0.0, ge=0, description="该股票当前持仓市值")
    max_position_value: Optional[float] = Field(
        None,
        ge=0,
        description="该股票允许的最大持仓市值上限（通常由集中度规则折算）",
    )


class PortfolioOptimizationInput(BaseModel):
    """组合优化的总输入模型，供优化器求解"""
    cash_available: float = Field(..., ge=0, description="可用现金")
    credit_available_limit: float = Field(..., ge=0, description="可新增融资负债上限")
    lot_size: int = Field(100, ge=1, description="最小交易股数单位")
    objective_mode: Literal["max_buy_value", "max_finance_used"] = Field(
        "max_buy_value",
        description="优化目标：max_buy_value 最大化买入市值；max_finance_used 最大化融资使用额",
    )
    stocks: List[StockOptimizationInput] = Field(..., min_length=1, description="待优化股票集合")


@dataclass
class StockOptimizationOutput:
    """单只股票优化后的结果输出模型"""
    stock_code: str
    financed_amount: float
    margin_cash_used: float
    self_cash_used: float
    total_buy_value: float
    buy_shares: int


def _solve_lp_with_fallback(model: pulp.LpProblem) -> tuple[int, str, List[str]]:
    """按 HiGHS→HiGHS_CMD→PULP_CBC_CMD 顺序依次尝试求解 LP 模型，返回首个可用求解器的求解结果"""
    solver_errors: List[str] = []

    for solver_name in ("HiGHS", "HiGHS_CMD", "PULP_CBC_CMD"):
        solver_cls = getattr(pulp, solver_name, None)
        if solver_cls is None:
            continue

        try:
            solver = solver_cls(msg=False)
            if hasattr(solver, "available") and not solver.available():
                solver_errors.append(f"{solver_name}: unavailable")
                continue
            return model.solve(solver), solver_name, solver_errors
        except pulp.PulpSolverError as exc:
            solver_errors.append(f"{solver_name}: {exc}")

    raise pulp.PulpSolverError("No available PuLP solver found")


@tool(args_schema=PortfolioOptimizationInput)
def optimize_trading_plan(cash_available: float, credit_available_limit: float, lot_size: int, stocks: List[StockOptimizationInput], objective_mode: str = "max_buy_value") -> str:
    """最优化求解器：在现金、融资额度、集中度约束下按整数手数最大化目标"""
    if objective_mode not in {"max_buy_value", "max_finance_used"}:
        return json.dumps(
            {"status": "failed", "message": f"unsupported objective_mode={objective_mode}"},
            ensure_ascii=False,
        )

    model = pulp.LpProblem("pre_trade_opt", pulp.LpMaximize)
    finance_lots: Dict[str, pulp.LpVariable] = {}
    self_lots: Dict[str, pulp.LpVariable] = {}

    for s in stocks:
        finance_lots[s.stock_code] = pulp.LpVariable(
            f"finance_lots_{s.stock_code}", lowBound=0, cat=pulp.LpInteger
        )
        self_lots[s.stock_code] = pulp.LpVariable(
            f"self_lots_{s.stock_code}", lowBound=0, cat=pulp.LpInteger
        )

    def financed_amount_expr(s: StockOptimizationInput) -> pulp.LpAffineExpression:
        return s.stock_price * lot_size * finance_lots[s.stock_code]

    def self_cash_expr(s: StockOptimizationInput) -> pulp.LpAffineExpression:
        return s.stock_price * lot_size * self_lots[s.stock_code]

    if objective_mode == "max_finance_used":
        model += pulp.lpSum(s.weight * financed_amount_expr(s) for s in stocks)
    else:
        model += pulp.lpSum(
            s.weight * (financed_amount_expr(s) + self_cash_expr(s)) for s in stocks
        )

    model += (
        pulp.lpSum(s.margin_ratio * financed_amount_expr(s) + self_cash_expr(s) for s in stocks)
        <= cash_available
    )
    model += pulp.lpSum(financed_amount_expr(s) for s in stocks) <= credit_available_limit

    for s in stocks:
        total_position_expr = s.existing_position_value + financed_amount_expr(s) + self_cash_expr(s)
        if s.max_position_value is not None:
            model += total_position_expr <= s.max_position_value

    try:
        status, solver_name, solver_errors = _solve_lp_with_fallback(model)
    except pulp.PulpSolverError as exc:
        return json.dumps({"status": "failed", "message": str(exc)}, ensure_ascii=False)

    if pulp.LpStatus[status] != "Optimal":
        return json.dumps(
            {
                "status": "failed",
                "message": f"optimization status={pulp.LpStatus[status]}",
                "solver": solver_name,
                "solver_errors": solver_errors,
            },
            ensure_ascii=False,
        )

    rows: List[StockOptimizationOutput] = []
    total_margin_cash = 0.0
    total_self_cash = 0.0
    total_finance = 0.0

    for s in stocks:
        finance_lot_count = max(0, int(round(finance_lots[s.stock_code].value() or 0)))
        self_lot_count = max(0, int(round(self_lots[s.stock_code].value() or 0)))
        financed_amount = s.stock_price * lot_size * finance_lot_count
        self_cash_used = s.stock_price * lot_size * self_lot_count
        margin_cash_used = s.margin_ratio * financed_amount
        buy_shares = lot_size * (finance_lot_count + self_lot_count)
        buy_value = financed_amount + self_cash_used

        rows.append(
            StockOptimizationOutput(
                stock_code=s.stock_code,
                financed_amount=round(financed_amount, 2),
                margin_cash_used=round(margin_cash_used, 2),
                self_cash_used=round(self_cash_used, 2),
                total_buy_value=round(buy_value, 2),
                buy_shares=buy_shares,
            )
        )
        total_margin_cash += margin_cash_used
        total_self_cash += self_cash_used
        total_finance += financed_amount

    payload = {
        "status": "success",
        "objective": round(float(pulp.value(model.objective)), 2),
        "objective_mode": objective_mode,
        "solver": solver_name,
        "cash_used_estimated": round(total_margin_cash + total_self_cash, 2),
        "cash_available": round(cash_available, 2),
        "credit_used_estimated": round(total_finance, 2),
        "credit_available_limit": round(credit_available_limit, 2),
        "positions": [asdict(r) for r in rows],
        "notes": [
            "buy_shares 由整数手数变量直接求解，天然满足 lot_size 约束。",
            "financed_amount 表示新增融资负债，并等于融资买入成交金额。",
            "margin_cash_used = margin_ratio * financed_amount。",
        ],
    }
    if solver_errors:
        payload["solver_fallback_notes"] = solver_errors
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# PreTradeAgent
# ---------------------------------------------------------------------------
class PreTradeAgent:
    """封装 agent"""
    def __init__(
        self,
        model: str = "",
        base_url: str = "",
        api_key: Optional[str] = None,
        reflective_url: str = "",
        trading_url: str = "",
        reflective_mcp_api_key: Optional[str] = None,
        trading_mcp_api_key: Optional[str] = None,
        reflective_mcp_api_header: str = "Authorization",
        trading_mcp_api_header: str = "Authorization",
        reflective_mcp_api_prefix: str = "Bearer",
        trading_mcp_api_prefix: str = "Bearer",
        sandbox_seed_dir: str = "./sandbox_seed",
        rules_dir: str = "./rules",
        daytona_api_key: str = "",
        daytona_api_url: str = "https://app.daytona.io/api",
        daytona_snapshot_id: str = "",
    ):
        self.model_name = model
        self.base_url = base_url
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "no-key-required")
        self.reflective_url = reflective_url
        self.trading_url = trading_url
        self.reflective_mcp_api_key = reflective_mcp_api_key or os.getenv("MCP_API_KEY")
        self.trading_mcp_api_key = trading_mcp_api_key or os.getenv("MCP_API_KEY")
        self.reflective_mcp_api_header = reflective_mcp_api_header
        self.trading_mcp_api_header = trading_mcp_api_header
        self.reflective_mcp_api_prefix = reflective_mcp_api_prefix
        self.trading_mcp_api_prefix = trading_mcp_api_prefix
        self.max_guard_retries = 2
        self.agent = None
        self.sandbox_seed_dir = sandbox_seed_dir or os.getenv("SANDBOX_SEED_DIR", "./sandbox_seed")
        self.rules_dir = rules_dir or os.getenv("RULES_DIR", "./rules")
        self.daytona_api_key = daytona_api_key or os.getenv("DAYTONA_API_KEY", "")
        self.daytona_api_url = daytona_api_url or os.getenv("DAYTONA_API_URL", "https://app.daytona.io/api")
        self.daytona_snapshot_id = daytona_snapshot_id or os.getenv("DAYTONA_SNAPSHOT_ID", "")
        self._build_lock = asyncio.Lock()
        # 方案 B：按 thread_id 累积本轮对话的完整澄清历史（原始问题 + 历次 [CLARIFY] + 用户历次回答），
        # 供 prompt_rewriter 每轮重新收敛"最新重写"、并据以计澄清轮次。CLI 与服务端按同一 thread_id 共用语义。
        self._clarify_threads: Dict[str, List[Dict[str, str]]] = {}
        self.max_clarify_rounds = int(os.getenv("MAX_CLARIFY_ROUNDS", "4"))

    @staticmethod
    def _build_mcp_headers(api_key: Optional[str], header_name: str, prefix: str) -> Optional[Dict[str, str]]:
        """根据 API 密钥、自定义 header 名与前缀构造 MCP 请求的认证头字典"""
        if not api_key:
            return None
        key = api_key.strip()
        if not key:
            return None
        header = header_name.strip() or "Authorization"
        p = prefix.strip()
        return {header: f"{p} {key}" if p else key}

    def _mcp_connections(self) -> Dict[str, Dict[str, Any]]:
        """组装单路 MCP 连接配置：reflective_thinking 与业务工具同挂在一台 MCP 服务上，故只建一个连接条目，
        避免同一批工具被 reflector_/trading_ 多个前缀重复加载（4 工具变 8）。URL/认证沿用现有配置（两套在
        当前部署下均回退到同一 MCP_URL/MCP_API_*）；如未来真拆两台，再恢复多条目即可。"""
        url = self.trading_url or self.reflective_url
        conn: Dict[str, Any] = {"transport": "http", "url": url}

        headers = self._build_mcp_headers(
            self.trading_mcp_api_key or self.reflective_mcp_api_key,
            self.trading_mcp_api_header,
            self.trading_mcp_api_prefix,
        )
        if headers:
            conn["headers"] = headers

        return {"mcp": conn}

    async def build(self):
        """构建 agent 图"""
        if self.agent is not None:
            return self.agent

        async with self._build_lock:
            if self.agent is not None:
                return self.agent

            # 配置代理
            _configure_proxy_env()

            # 创建 llm
            llm = ChatOpenAI(
                model=self.model_name,
                base_url=self.base_url,
                api_key=self.api_key,
                temperature=0,
            )

            # 加载 MCP 工具
            mcp_client = MultiServerMCPClient(self._mcp_connections(), tool_name_prefix=True)
            timeout = float(os.getenv("MCP_TOOLS_TIMEOUT_SECONDS", "15"))  # 防止 MCP 端点宕机时 LangGraph Studio 无限等待
            mcp_required = os.getenv("MCP_REQUIRED", "false").lower() == "true"
            try:
                if timeout > 0:
                    mcp_tools = await asyncio.wait_for(mcp_client.get_tools(), timeout=timeout)
                else:
                    mcp_tools = await mcp_client.get_tools()
            except Exception as exc:
                if mcp_required:
                    raise
                print(
                    f"[agent] WARNING: MCP 工具加载失败，已降级为仅本地工具继续构建。"
                    f"（{type(exc).__name__}: {exc}）"
                )
                mcp_tools = []

            # 沙箱后端选择：读取 SANDBOX_PROVIDER 环境变量决定走本地还是 Daytona
            sandbox_provider = os.getenv("SANDBOX_PROVIDER", "daytona").lower()

            if sandbox_provider == "local":
                # ---- 本地文件系统后端（开发 / Docker 部署） ----
                import asyncio as _asyncio
                from pathlib import Path as _Path

                def _setup_local_root() -> _Path:
                    """在独立线程中解析路径并创建可写目录，避免阻塞事件循环。"""
                    d = _Path(self.sandbox_seed_dir).resolve()
                    (d / "scratch").mkdir(exist_ok=True)
                    (d / "outputs").mkdir(exist_ok=True)
                    return d

                seed_dir = await _asyncio.to_thread(_setup_local_root)
                print(
                    f"[agent] using LocalShellBackend, root_dir={seed_dir}  "
                    f"virtual_mode=True"
                )
                backend = LocalShellBackend(
                    root_dir=str(seed_dir),
                    virtual_mode=True,
                )
                sandbox_home = "/"

            else:
                # ---- Daytona 云沙箱后端（现有代码不变） ----
                if not self.daytona_api_key:
                    raise RuntimeError(
                        "DAYTONA_API_KEY 未配置：SANDBOX_PROVIDER=daytona 时必须在环境变量"
                        "或 .env 中设置 DAYTONA_API_KEY。"
                    )

                _backend_by_run: dict[str, PerRunDaytonaBackend] = {}

                quota_manager = SandboxQuotaManager(
                    api_key=self.daytona_api_key,
                    api_url=self.daytona_api_url,
                    max_sandboxes=int(os.getenv("DAYTONA_MAX_SANDBOXES", "10")),
                    active_window_seconds=int(os.getenv("DAYTONA_ACTIVE_WINDOW_SECONDS", "300")),
                )

                def _backend_factory(runtime, /) -> PerRunDaytonaBackend:
                    cfg = None
                    try:
                        cfg = runtime.config
                    except AttributeError:
                        try:
                            from langgraph.config import get_config
                            cfg = get_config()
                        except LookupError:
                            pass
                    key = None
                    if cfg:
                        key = cfg.get("configurable", {}).get("thread_id")
                        if not key:
                            key = cfg.get("run_id")
                    if not key:
                        key = "default"
                    if key not in _backend_by_run:
                        print(f"[agent] new PerRunDaytonaBackend for key={key}")
                        _backend_by_run[key] = PerRunDaytonaBackend(
                            api_key=self.daytona_api_key,
                            api_url=self.daytona_api_url,
                            snapshot_id=self.daytona_snapshot_id,
                            seed_dir=self.sandbox_seed_dir,
                            manager=quota_manager,
                        )
                    return _backend_by_run[key]

                backend = _backend_factory
                sandbox_home = DAYTONA_SANDBOX_HOME

            # 组装 agent 图
            all_tools = [*mcp_tools, optimize_trading_plan]
            prompt_rewriter = await _build_prompt_rewriter_subagent(self.rules_dir)
            self.agent = create_deep_agent(
                model=llm,
                tools=all_tools,
                system_prompt=get_system_prompt(sandbox_home),
                backend=backend,
                subagents=[prompt_rewriter],
                middleware=[ExcludeToolsMiddleware(frozenset({"write_todos"}))],
                debug=os.getenv("DEEPAGENT_DEBUG", "false").lower() == "true",
                name="pre_trade_agent",
            )
            return self.agent

    @staticmethod
    def _extract_text(result: Any) -> str:
        """从 agent 返回结果中提取最终文本"""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            messages = result.get("messages")
            if isinstance(messages, list):
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        content = msg.content
                        if isinstance(content, str):
                            return content
                        if isinstance(content, list):
                            parts: List[str] = []
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    parts.append(str(item.get("text", "")))
                                else:
                                    parts.append(str(item))
                            return "\n".join(p for p in parts if p)
            return json.dumps(result, ensure_ascii=False)
        return str(result)

    @staticmethod
    def _as_text_blob(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            return "\n".join(p for p in parts if p)
        return str(content)

    @staticmethod
    def _try_parse_json_payload(text: str) -> Optional[Dict[str, Any]]:
        """将单条消息的 content 解析为 JSON 字典"""
        text = text.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        left = text.find("{")
        right = text.rfind("}")
        if left >= 0 and right > left:
            try:
                parsed = json.loads(text[left : right + 1])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return None
        return None

    @staticmethod
    def _is_reflective_tool_name(name: Optional[str]) -> bool:
        """判断给定的工具名是否为 reflective_thinking"""
        if not name:
            return False
        return "reflective_thinking" in name

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        """将任意值转为非负整数，而非抛出异常"""
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return int(stripped)
        return None

    def _validate_reflective_guard(self, result: Any) -> ReflectiveGuardResult:
        """强制检查 agent 在给出最终答案前，是否完成 reflective_thinking 全流程"""
        # 结果格式校验
        if not isinstance(result, dict):
            return ReflectiveGuardResult(
                ok=False,
                reason="reflective_guard_violation",
                details={"message": "agent result is not dict; cannot inspect reflective trace"},
            )
        messages = result.get("messages")
        if not isinstance(messages, list):
            return ReflectiveGuardResult(
                ok=False,
                reason="reflective_guard_violation",
                details={"message": "missing messages in result; cannot inspect reflective trace"},
            )

        # 遍历消息，提取所有 reflective_thinking 工具调用
        reflective_calls: List[Dict[str, Any]] = []
        reflective_call_id_to_name: Dict[str, str] = {}
        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            tool_calls = getattr(msg, "tool_calls", None)
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                name = call.get("name")
                if not self._is_reflective_tool_name(name):
                    continue
                call_id = str(call.get("id", ""))
                if call_id:
                    reflective_call_id_to_name[call_id] = str(name)
                args = call.get("args", {})
                if not isinstance(args, dict):
                    args = {}
                reflective_calls.append(
                    {
                        "name": name,
                        "id": call.get("id"),
                        "thought_number": self._coerce_int(args.get("thought_number")),
                        "total_thoughts": self._coerce_int(args.get("total_thoughts")),
                    }
                )

        # 完全没调用过反思工具 → 直接拦截
        if not reflective_calls:
            return ReflectiveGuardResult(
                ok=False,
                reason="reflective_guard_violation",
                details={"message": "no reflective_thinking tool call observed"},
            )

        # 逐个检查每次反思调用，校验思考过程的逻辑合法性
        prev_thought_number = 0
        for idx, call in enumerate(reflective_calls, start=1):
            thought_number = call["thought_number"]
            total_thoughts = call["total_thoughts"]
            if thought_number is None or total_thoughts is None:
                return ReflectiveGuardResult(
                    ok=False,
                    reason="reflective_guard_violation",
                    details={
                        "message": "reflective_thinking args missing thought_number/total_thoughts",
                        "call_index": idx,
                        "call": call,
                    },
                )
            if thought_number < 1 or thought_number <= prev_thought_number:
                return ReflectiveGuardResult(
                    ok=False,
                    reason="reflective_guard_violation",
                    details={
                        "message": "thought_number must start from 1 and be strictly increasing",
                        "call_index": idx,
                        "call": call,
                    },
                )
            if total_thoughts <= 1:
                return ReflectiveGuardResult(
                    ok=False,
                    reason="reflective_guard_violation",
                    details={
                        "message": "total_thoughts must be greater than 1",
                        "call_index": idx,
                        "call": call,
                    },
                )
            if total_thoughts < thought_number:
                return ReflectiveGuardResult(
                    ok=False,
                    reason="reflective_guard_violation",
                    details={
                        "message": "total_thoughts must be >= thought_number",
                        "call_index": idx,
                        "call": call,
                    },
                )
            prev_thought_number = thought_number

        # 检查最近一次反思工具的结果中是否 next_thought_needed=False
        reflective_tool_payloads: List[Tuple[Optional[str], Optional[Dict[str, Any]]]] = []
        for msg in messages:
            tool_name = getattr(msg, "name", None)
            tool_call_id = getattr(msg, "tool_call_id", None)
            name_from_call_id = reflective_call_id_to_name.get(str(tool_call_id), "")
            if self._is_reflective_tool_name(tool_name) or self._is_reflective_tool_name(name_from_call_id):
                payload = self._try_parse_json_payload(self._as_text_blob(getattr(msg, "content", "")))
                reflective_tool_payloads.append((tool_name or name_from_call_id, payload))

        # 有调用却没有任何返回 → 直接拦截
        if not reflective_tool_payloads:
            return ReflectiveGuardResult(
                ok=False,
                reason="reflective_guard_violation",
                details={
                    "message": "no reflective_thinking tool response observed",
                    "last_call": reflective_calls[-1],
                },
            )

        # 检查最后一次反思工具的结果中是否 next_thought_needed=False
        _, latest_payload = reflective_tool_payloads[-1] # 取最后一次调用的返回结果
        latest_next_needed = None
        if isinstance(latest_payload, dict):
            latest_next_needed = latest_payload.get("next_thought_needed")
        if latest_next_needed is not False:
            return ReflectiveGuardResult(
                ok=False,
                reason="reflective_guard_violation",
                details={
                    "message": "final gate failed: latest next_thought_needed must be False",
                    "last_next_thought_needed": latest_next_needed,
                    "last_reflective_payload": latest_payload,
                },
            )

        return ReflectiveGuardResult(ok=True)

    def _build_guard_retry_hint(self, violation: ReflectiveGuardResult, retry_idx: int) -> str:
        """reflective_thinking 工具调用校验失败时的重试提示"""
        details_text = json.dumps(violation.details or {}, ensure_ascii=False)
        return (
            "系统运行时校验未通过，请严格纠偏后重做本轮推理并重新给出最终回答。\n"
            "必须满足：\n"
            "1) reflective_thinking 的 thought_number 从 1 开始且严格递增；\n"
            "2) 每次 reflective_thinking 都满足 total_thoughts > 1 且 total_thoughts >= thought_number；\n"
            "3) 输出最终结论前，最后一次 reflective_thinking 的 next_thought_needed 必须为 False。\n"
            f"当前第 {retry_idx} 次重试，违规详情：{details_text}"
        )

    @staticmethod
    def _guard_failure_payload(reason: str, details: Dict[str, Any]) -> str:
        """将reflective_thinking 工具调用校验失败信息打包成统一的 JSON 失败响应"""
        return json.dumps(
            {
                "status": "failed",
                "reason": reason,
                "details": details,
            },
            ensure_ascii=False,
        )

    async def ask(
        self,
        question: str,
        clarification_context: Optional[str] = None,
        *,
        thread_id: str = "default",
    ) -> str:
        """对外主入口：构建 agent 并提问，每轮回答都经反思护栏校验；通过则返回最终文本，不通过则把纠错提示累积注入后重试，直至通过或重试耗尽（返回结构化失败 JSON）。

        方案 B：按 thread_id 累积本轮对话的完整澄清历史并随每次提问重放，使 prompt_rewriter 能据完整历史
        迭代收敛"最新重写"，并据历史中的 [CLARIFY] 计数施加澄清预算（默认 4 轮）。
        """
        agent = await self.build()

        # 取出该 thread 至今的完整历史；据其中以 [CLARIFY] 开头的助手消息数计已澄清轮次
        history: List[Dict[str, str]] = list(self._clarify_threads.get(thread_id, []))
        clarify_rounds = sum(
            1
            for m in history
            if m.get("role") == "assistant"
            and str(m.get("content", "")).lstrip().startswith("[CLARIFY]")
        )

        # 本轮重放给模型的基础消息 = 历史 + 本次提问（历史里已含原始问题与历次 [CLARIFY]/回答）
        base_msgs: List[Dict[str, str]] = list(history) + [{"role": "user", "content": question}]
        if clarification_context:
            base_msgs.append({
                "role": "user",
                "content": f"[系统注入的上文] 上一次对话中：\n{clarification_context}",
            })
        # 澄清预算提示（让模型可靠地知道已澄清轮次，避免无限追问）
        if clarify_rounds > 0:
            if clarify_rounds >= self.max_clarify_rounds:
                base_msgs.append({
                    "role": "user",
                    "content": (
                        f"[系统提示] 本轮对话已进行 {clarify_rounds} 轮澄清，已达预算上限 "
                        f"{self.max_clarify_rounds} 轮。请不要再发起 [CLARIFY]：改为按可确定部分给出最可能结论，"
                        f"或按 [STOP] 收口（reason_code 取 unresolved_ambiguity 或 missing_critical_input）。"
                    ),
                })
            else:
                base_msgs.append({
                    "role": "user",
                    "content": (
                        f"[系统提示] 本轮对话已进行 {clarify_rounds} 轮澄清，预算上限 "
                        f"{self.max_clarify_rounds} 轮（剩余 {self.max_clarify_rounds - clarify_rounds} 轮）。"
                    ),
                })

        retry_hints: List[str] = []
        max_attempts = self.max_guard_retries + 1

        # 主循环
        for attempt in range(1, max_attempts + 1):
            # 重新拼装消息列表（基础消息 + 本次重试的纠错提示；纠错提示不入库）
            msgs = list(base_msgs)
            for hint in retry_hints:
                msgs.append({"role": "user", "content": hint})

            # 调用 agent 并获取结果（& 校验）
            result = await agent.ainvoke({"messages": msgs})
            guard = self._validate_reflective_guard(result)
            if guard.ok:
                answer = self._extract_text(result)
                # 仅把"本次提问 + 最终回答"提交进该 thread 历史（不含纠错提示/预算提示/注入上文）
                self._clarify_threads[thread_id] = history + [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
                return answer

            # 重试耗尽
            if attempt >= max_attempts:
                return self._guard_failure_payload(
                    "reflective_guard_retry_exhausted",
                    {
                        "attempts": attempt,
                        "last_violation_reason": guard.reason,
                        "last_violation_details": guard.details,
                        "recommendation": "请重试，并确保 reflective_thinking 参数与 next_thought_needed 约束满足要求。",
                    },
                )
            
            # 累积纠错提示并重试下一轮
            retry_hints.append(self._build_guard_retry_hint(guard, retry_idx=attempt))

        # 兜底
        return self._guard_failure_payload(
            "reflective_guard_violation",
            {"message": "unexpected guard flow termination"},
        )

    async def loop(self) -> None:
        """交互主循环"""
        print("=" * 60)
        print("交易试算 Agent 已启动（输入 quit 退出）")
        print("=" * 60)
        thread_id = "cli-session"  # 单会话；ask() 按此 thread_id 累积完整澄清历史
        # 读取输入 & 处理退出条件
        while True:
            try:
                question = input("\nUser: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[退出]")
                break
            if not question:
                continue
            if question.lower() in {"quit", "exit", "退出"}:
                break
            try:
                answer = await self.ask(question, thread_id=thread_id)
                print(f"\nAI: {answer}")

                # 澄清机制：回答仍是 [CLARIFY] 则保留历史等待用户回应继续迭代重写；
                # 一旦本题收口（非 [CLARIFY] 的最终回答/[STOP]），清空该 thread，下一题重新开始。
                if not answer.strip().startswith("[CLARIFY]"):
                    self._clarify_threads.pop(thread_id, None)

            # 异常隔离
            except Exception:
                print("\n[错误] 发生异常：")
                traceback.print_exc()


# ---------------------------------------------------------------------------
# factory function
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_env_once() -> None:
    """加载环境变量，保证只加载一次"""
    load_dotenv()
    _configure_proxy_env()

@lru_cache(maxsize=1)
def create_agent_from_env() -> PreTradeAgent:
    """从环境变量创建 PreTradeAgent 实例，保证只创建一次"""
    _load_env_once()
    return PreTradeAgent(
        model=os.getenv("OPENAI_MODEL", ""),
        base_url=os.getenv("OPENAI_BASE_URL", ""),
        api_key=os.getenv("OPENAI_API_KEY", "no-key-required"),
        reflective_url=os.getenv("REFLECTIVE_MCP_URL", os.getenv("MCP_URL", "")),
        trading_url=os.getenv("TRADING_MCP_URL", os.getenv("MCP_URL", "")),
        reflective_mcp_api_key=os.getenv("REFLECTIVE_MCP_API_KEY", os.getenv("MCP_API_KEY")),
        trading_mcp_api_key=os.getenv("TRADING_MCP_API_KEY", os.getenv("MCP_API_KEY")),
        reflective_mcp_api_header=os.getenv("REFLECTIVE_MCP_API_HEADER", os.getenv("MCP_API_HEADER", "Authorization")),
        trading_mcp_api_header=os.getenv("TRADING_MCP_API_HEADER", os.getenv("MCP_API_HEADER", "Authorization")),
        reflective_mcp_api_prefix=os.getenv("REFLECTIVE_MCP_API_PREFIX", os.getenv("MCP_API_PREFIX", "Bearer")),
        trading_mcp_api_prefix=os.getenv("TRADING_MCP_API_PREFIX", os.getenv("MCP_API_PREFIX", "Bearer")),
        sandbox_seed_dir=os.getenv("SANDBOX_SEED_DIR", "./sandbox_seed"),
        rules_dir=os.getenv("RULES_DIR", "./rules"),
        daytona_api_key=os.getenv("DAYTONA_API_KEY", ""),
        daytona_api_url=os.getenv("DAYTONA_API_URL", "https://app.daytona.io/api"),
        daytona_snapshot_id=os.getenv("DAYTONA_SNAPSHOT_ID", ""),
    )


# ---------------------------------------------------------------------------
# LangGraph Studio 
# ---------------------------------------------------------------------------
_graph = None
_graph_lock: Optional[asyncio.Lock] = None

async def make_graph(config: Optional[RunnableConfig] = None):
    """LangGraph Studio 入口"""
    global _graph, _graph_lock

    if _graph is not None:
        return _graph

    if _graph_lock is None:
        _graph_lock = asyncio.Lock()

    async with _graph_lock:
        if _graph is None:
            app = create_agent_from_env()
            _graph = await app.build()
        return _graph


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
async def main() -> None:
    """CLI 入口"""
    agent = create_agent_from_env()
    await agent.loop()