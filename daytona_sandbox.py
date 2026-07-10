from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    FileInfo,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.sandbox import (
    BaseSandbox,
    _EDIT_COMMAND_TEMPLATE,
    _EDIT_INLINE_MAX_BYTES,
    _EDIT_TMPFILE_TEMPLATE,
    _GLOB_COMMAND_TEMPLATE,
    _READ_COMMAND_TEMPLATE,
    _WRITE_CHECK_TEMPLATE,
)
from deepagents.backends.utils import _get_file_type

from daytona_sdk import AsyncDaytona, DaytonaConfig
from daytona_sdk.common.filesystem import FileDownloadRequest, FileUpload

if TYPE_CHECKING:
    from daytona_sdk import AsyncSandbox

logger = logging.getLogger(__name__)

DAYTONA_SANDBOX_HOME = "/home/daytona/"


class DaytonaSandbox(BaseSandbox):
    """符合 `SandboxBackendProtocol` 的 Daytona 云沙箱后端。

    所有异步方法都直接调用 Daytona 的异步 SDK——没有 ``asyncio.run()``，也不绕道线程池。同步方法仅为兼容性提供，不应在异步上下文中调用。
    """

    def __init__(self, sandbox: AsyncSandbox, sandbox_home_dir: str = DAYTONA_SANDBOX_HOME) -> None:
        self._sandbox = sandbox
        self.sandbox_home_dir = sandbox_home_dir
        self._default_timeout: int = 30

    @property
    def id(self) -> str:
        return self._sandbox.id

    # ------------------------------------------------------------------
    # execute（同步 + 异步）
    # ------------------------------------------------------------------

    def execute(
        self, command: str, *, timeout: int | None = None
    ) -> ExecuteResponse:
        raise RuntimeError(
            "DaytonaSandbox.execute() called synchronously. "
            "Use aexecute() instead."
        )

    async def aexecute(
        self, command: str, *, timeout: int | None = None
    ) -> ExecuteResponse:
        effective = timeout if timeout is not None else self._default_timeout
        result = await self._sandbox.process.exec(command, timeout=effective)
        output = result.result or ""
        return ExecuteResponse(
            output=output,
            exit_code=result.exit_code,
            truncated=False,
        )

    # ------------------------------------------------------------------
    # ls / grep / glob —— 直接调用 aexecute 的异步重写版本
    # ------------------------------------------------------------------

    async def als(self, path: str) -> LsResult:
        path_b64 = base64.b64encode(path.encode("utf-8")).decode("ascii")
        cmd = f"""python3 -c "
import os
import json
import base64

path = base64.b64decode('{path_b64}').decode('utf-8')

try:
    with os.scandir(path) as it:
        for entry in it:
            result = {{
                'path': os.path.join(path, entry.name),
                'is_dir': entry.is_dir(follow_symlinks=False)
            }}
            print(json.dumps(result))
except FileNotFoundError:
    pass
except PermissionError:
    pass
" 2>/dev/null"""
        result = await self.aexecute(cmd)
        file_infos: list[FileInfo] = []
        for line in result.output.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                file_infos.append({"path": data["path"], "is_dir": data["is_dir"]})
            except json.JSONDecodeError:
                continue
        return LsResult(entries=file_infos)

    async def agrep(
        self, pattern: str, path: str | None = None, glob: str | None = None
    ) -> GrepResult:
        import shlex
        search_path = shlex.quote(path or ".")
        grep_opts = "-rHnF"
        glob_pattern = ""
        if glob:
            glob_pattern = f"--include='{glob}'"
        pattern_escaped = shlex.quote(pattern)
        cmd = f"grep {grep_opts} {glob_pattern} -e {pattern_escaped} {search_path} 2>/dev/null || true"
        result = await self.aexecute(cmd)
        output = result.output.rstrip()
        if not output:
            return GrepResult(matches=[])
        matches: list[GrepMatch] = []
        for line in output.split("\n"):
            parts = line.split(":", 2)
            if len(parts) >= 3:
                matches.append({
                    "path": parts[0],
                    "line": int(parts[1]),
                    "text": parts[2],
                })
        return GrepResult(matches=matches)

    async def aglob(self, pattern: str, path: str = "/") -> GlobResult:
        pattern_b64 = base64.b64encode(pattern.encode("utf-8")).decode("ascii")
        path_b64 = base64.b64encode(path.encode("utf-8")).decode("ascii")
        cmd = _GLOB_COMMAND_TEMPLATE.format(path_b64=path_b64, pattern_b64=pattern_b64)
        result = await self.aexecute(cmd)
        output = result.output.strip()
        if not output:
            return GlobResult(matches=[])
        file_infos: list[FileInfo] = []
        for line_text in output.split("\n"):
            try:
                data = json.loads(line_text)
                file_infos.append({
                    "path": data["path"],
                    "is_dir": data.get("is_dir", False),
                })
            except json.JSONDecodeError:
                continue
        return GlobResult(matches=file_infos)

    # ------------------------------------------------------------------
    # read（异步重写版本）
    # ------------------------------------------------------------------

    async def aread(
        self, file_path: str, offset: int = 0, limit: int = 2000
    ) -> ReadResult:
        file_type = _get_file_type(file_path)
        path_b64 = base64.b64encode(file_path.encode("utf-8")).decode("ascii")

        cmd = _READ_COMMAND_TEMPLATE.format(
            path_b64=path_b64,
            file_type=file_type,
            offset=int(offset),
            limit=int(limit),
        )
        return self._parse_read_result(file_path, await self.aexecute(cmd))

    @staticmethod
    def _parse_read_result(file_path: str, result: ExecuteResponse) -> ReadResult:
        import json
        output = result.output.rstrip()
        try:
            data = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            detail = output[:200] if output else "(empty)"
            return ReadResult(error=f"File '{file_path}': unexpected server response: {detail}")
        if not isinstance(data, dict):
            return ReadResult(error=f"File '{file_path}': unexpected server response: {output[:200]}")
        if "error" in data:
            return ReadResult(error=f"File '{file_path}': {data['error']}")
        from deepagents.backends.protocol import FileData
        return ReadResult(
            file_data=FileData(
                content=data["content"],
                encoding=data.get("encoding", "utf-8"),
            )
        )

    # ------------------------------------------------------------------
    # write（异步重写版本）
    # ------------------------------------------------------------------

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        path_b64 = base64.b64encode(file_path.encode("utf-8")).decode("ascii")
        check_cmd = _WRITE_CHECK_TEMPLATE.format(path_b64=path_b64)
        result = await self.aexecute(check_cmd)
        if result.exit_code != 0 or "Error:" in result.output:
            error_msg = result.output.strip() or f"Failed to write file '{file_path}'"
            return WriteResult(error=error_msg)

        responses = await self.aupload_files(
            [(file_path, content.encode("utf-8"))]
        )
        if not responses:
            return WriteResult(error=f"Upload returned no response for '{file_path}'")
        if responses[0].error:
            return WriteResult(
                error=f"Failed to write file '{file_path}': {responses[0].error}"
            )
        return WriteResult(path=file_path)

    # ------------------------------------------------------------------
    # edit（异步重写版本）
    # ------------------------------------------------------------------

    async def aedit(
        self, file_path: str, old_string: str, new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        import json

        payload_size = (
            len(old_string.encode("utf-8")) + len(new_string.encode("utf-8"))
        )

        if payload_size <= _EDIT_INLINE_MAX_BYTES:
            payload = json.dumps({
                "path": file_path,
                "old": old_string,
                "new": new_string,
                "replace_all": replace_all,
            })
            payload_b64 = base64.b64encode(
                payload.encode("utf-8")
            ).decode("ascii")
            cmd = _EDIT_COMMAND_TEMPLATE.format(payload_b64=payload_b64)
            result = await self.aexecute(cmd)
            return self._parse_edit_result(file_path, old_string, result)

        import os as _os
        uid = base64.b32encode(_os.urandom(10)).decode("ascii").lower()
        old_tmp = f"/tmp/.deepagents_edit_{uid}_old"
        new_tmp = f"/tmp/.deepagents_edit_{uid}_new"

        resps = await self.aupload_files([
            (old_tmp, old_string.encode("utf-8")),
            (new_tmp, new_string.encode("utf-8")),
        ])
        if len(resps) < 2 or any(r.error for r in resps):
            return EditResult(
                error=f"Error editing file '{file_path}': temp upload failed"
            )

        cmd = _EDIT_TMPFILE_TEMPLATE.format(
            old_path_b64=base64.b64encode(old_tmp.encode("utf-8")).decode("ascii"),
            new_path_b64=base64.b64encode(new_tmp.encode("utf-8")).decode("ascii"),
            target_b64=base64.b64encode(file_path.encode("utf-8")).decode("ascii"),
            replace_all=replace_all,
        )
        result = await self.aexecute(cmd)
        return self._parse_edit_result(file_path, old_string, result)

    @staticmethod
    def _parse_edit_result(
        file_path: str, old_string: str, result: ExecuteResponse
    ) -> EditResult:
        import json
        output = result.output.rstrip()
        try:
            data = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            detail = output[:200] if output else "(empty)"
            return EditResult(error=f"Error editing file '{file_path}': unexpected server response: {detail}")
        if not isinstance(data, dict):
            return EditResult(error=f"Error editing file '{file_path}': unexpected server response: {output[:200]}")
        if "error" in data:
            return BaseSandbox._map_edit_error(data["error"], file_path, old_string)
        return EditResult(path=file_path, occurrences=data.get("count", 1))

    # ------------------------------------------------------------------
    # upload / download（同步 + 异步）
    # ------------------------------------------------------------------

    def upload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        raise RuntimeError(
            "DaytonaSandbox.upload_files() called synchronously. "
            "Use aupload_files() instead."
        )

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        daytona_files = [
            FileUpload(source=content, destination=path)
            for path, content in files
        ]
        try:
            await self._sandbox.fs.upload_files(daytona_files)
        except Exception as exc:
            logger.debug("Daytona upload_files failed: %s", exc)
            return [
                FileUploadResponse(
                    path=path, error=f"{type(exc).__name__}: {exc}"
                )
                for path, _ in files
            ]
        return [FileUploadResponse(path=path) for path, _ in files]

    def download_files(
        self, paths: list[str]
    ) -> list[FileDownloadResponse]:
        raise RuntimeError(
            "DaytonaSandbox.download_files() called synchronously. "
            "Use adownload_files() instead."
        )

    async def adownload_files(
        self, paths: list[str]
    ) -> list[FileDownloadResponse]:
        requests = [FileDownloadRequest(source=p) for p in paths]
        try:
            results = await self._sandbox.fs.download_files(requests)
        except Exception as exc:
            logger.debug("Daytona download_files failed: %s", exc)
            return [
                FileDownloadResponse(
                    path=p, content=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
                for p in paths
            ]

        responses: list[FileDownloadResponse] = []
        for i, r in enumerate(results):
            if r.error:
                responses.append(
                    FileDownloadResponse(
                        path=paths[i], content=None, error=r.error
                    )
                )
            else:
                content = r.result
                if isinstance(content, str):
                    content = content.encode("utf-8")
                responses.append(
                    FileDownloadResponse(path=paths[i], content=content)
                )
        return responses


# ============================================================================
# 沙箱配额管理 —— 账号级硬上限（FIFO 淘汰，跳过活跃中的）
# ============================================================================

AGENT_SANDBOX_LABELS = {"app": "pre_trade_agent"}  # 仅统计/淘汰带此 label 的沙箱，避免误删账号内其它沙箱
DEFAULT_MAX_SANDBOXES = 10
DEFAULT_ACTIVE_WINDOW_SECONDS = 300  # 最近这么多秒内被本进程访问过的沙箱视为“活跃”，淘汰时跳过


def _parse_created_at(value: object) -> datetime:
    """把 Daytona 的 created_at 字符串解析为可比较的 datetime；无法解析时按“最新”处理（最不易被删）。"""
    _newest = datetime.max.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value:
        return _newest
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return _newest
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


class SandboxQuotaManager:
    """账号级沙箱配额管理器。

    维持账号内带 ``labels`` 的沙箱总数不超过 ``max_sandboxes``。需要新建沙箱时，若已达上限，
    先按 ``created_at`` 从最早开始删除【非活跃】沙箱（活跃 = 最近 ``active_window_seconds`` 秒内被
    本进程访问过），腾出名额后再创建；若所有候选都活跃却仍超限，则兜底删除最早的以守住硬上限。

    “列出 → 淘汰 → 创建”整段在一把 ``asyncio.Lock`` 内完成，避免多个 thread 并发建沙箱时数错、超卖。
    口径为账号级（通过 Daytona ``list`` 实时查询），因此能把上一轮 ``langgraph dev`` 残留的孤儿沙箱
    也计入上限并优先清理（孤儿不在本进程活跃表中，天然非活跃）。
    """

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str,
        labels: dict[str, str] | None = None,
        max_sandboxes: int = DEFAULT_MAX_SANDBOXES,
        active_window_seconds: int = DEFAULT_ACTIVE_WINDOW_SECONDS,
    ) -> None:
        self._api_key = api_key
        self._api_url = api_url
        self._labels = dict(labels or AGENT_SANDBOX_LABELS)
        self._max = max(1, int(max_sandboxes))
        self._active_window = max(0, int(active_window_seconds))
        self._lock = asyncio.Lock()
        self._client: AsyncDaytona | None = None
        self._by_sandbox_id: dict[str, PerRunDaytonaBackend] = {}  # 沙箱 id → 对应 backend，淘汰时用于失效其委托
        self._last_access: dict[str, float] = {}  # 沙箱 id → 最后访问时间（monotonic）

    def _ensure_client(self) -> AsyncDaytona:
        if self._client is None:
            self._client = AsyncDaytona(DaytonaConfig(api_key=self._api_key, api_url=self._api_url))
        return self._client

    def touch(self, sandbox_id: str | None) -> None:
        """标记某沙箱刚被访问（刷新活跃时间）。由 backend 在每次操作前调用。"""
        if sandbox_id:
            self._last_access[sandbox_id] = time.monotonic()

    def _is_active(self, sandbox_id: str) -> bool:
        ts = self._last_access.get(sandbox_id)
        return ts is not None and (time.monotonic() - ts) < self._active_window

    async def create_under_cap(self, *, snapshot_id: str, backend: "PerRunDaytonaBackend") -> AsyncSandbox:
        """在不超过账号配额的前提下创建并返回一个带 label 的沙箱。"""
        from daytona_sdk import CreateSandboxFromSnapshotParams

        async with self._lock:
            await self._evict_to_make_room()
            client = self._ensure_client()
            if snapshot_id:
                params = CreateSandboxFromSnapshotParams(snapshot_id=snapshot_id, labels=dict(self._labels))
            else:
                params = CreateSandboxFromSnapshotParams(labels=dict(self._labels))
            sandbox = await client.create(params)
            print(f"[quota] sandbox created (id={sandbox.id}) labels={self._labels}")
            self._by_sandbox_id[sandbox.id] = backend
            self.touch(sandbox.id)
            return sandbox

    async def _evict_to_make_room(self) -> None:
        """若带 label 的沙箱数已达上限，删除最早创建的非活跃沙箱直至能再容纳一个。须在持锁状态下调用。"""
        client = self._ensure_client()
        try:
            # client.list() 是 async def，返回 AsyncPaginatedSandboxes（Pydantic 模型，
            # 无 __aiter__），必须 await 后读 .items，不能用 async for。逐页拉全，避免漏算。
            labels = dict(self._labels)
            first = await client.list(labels=labels)
            sandboxes = list(first.items)
            for page_num in range(2, int(getattr(first, "total_pages", 1) or 1) + 1):
                nxt = await client.list(labels=labels, page=page_num)
                sandboxes.extend(nxt.items)
        except Exception as exc:  # 列举失败不应阻断创建，跳过本次淘汰
            print(f"[quota] WARNING: 列举沙箱失败，跳过本次淘汰：{type(exc).__name__}: {exc}")
            return

        overflow = len(sandboxes) - (self._max - 1)  # 需要删除多少个才能在创建后仍 <= max
        if overflow <= 0:
            return

        sandboxes.sort(key=lambda sb: _parse_created_at(getattr(sb, "created_at", None)))  # 最早创建在前
        deleted: set[str] = set()

        # 第一轮：从最早开始，只删非活跃的
        for sb in sandboxes:
            if len(deleted) >= overflow:
                break
            if not self._is_active(sb.id) and await self._delete(sb):
                deleted.add(sb.id)

        # 第二轮兜底：若候选全是活跃的但仍超限，按 FIFO 删最早的以守住硬上限
        if len(deleted) < overflow:
            for sb in sandboxes:
                if len(deleted) >= overflow:
                    break
                if sb.id in deleted:
                    continue
                print(f"[quota] WARNING: 候选沙箱均处于活跃中，为守住上限 {self._max} 删除活跃沙箱 id={sb.id}")
                if await self._delete(sb):
                    deleted.add(sb.id)

    async def _delete(self, sandbox: AsyncSandbox, *, retries: int = 2, retry_delay: float = 1.0) -> bool:
        sid = sandbox.id
        last_exc: Exception | None = None
        for attempt in range(retries + 1):  # 限流/瞬时网络错误时有限重试，提高逐出成功率
            try:
                await self._ensure_client().delete(sandbox)
                print(f"[quota] evicted sandbox id={sid} created_at={getattr(sandbox, 'created_at', None)}")
                self._last_access.pop(sid, None)
                backend = self._by_sandbox_id.pop(sid, None)
                if backend is not None:
                    backend._invalidate()  # 让对应 backend 下次访问时惰性重建一个新沙箱
                return True
            except Exception as exc:
                last_exc = exc
                if attempt < retries:
                    await asyncio.sleep(retry_delay)
        print(f"[quota] WARNING: 删除沙箱失败 id={sid}（重试 {retries} 次后仍失败）：{type(last_exc).__name__}: {last_exc}")
        return False


# ============================================================================
# 按运行封装 —— 每次 LangGraph 运行（线程）对应一个全新沙箱
# ============================================================================

class PerRunDaytonaBackend(BaseSandbox):
    """惰性创建单个 ``DaytonaSandbox`` 的后端。

    每个实例确保只创建一个 Daytona 沙箱（通过带双重检查的 ``asyncio.Lock``
    保护）。调用方负责为每次 LangGraph 运行创建一个 ``PerRunDaytonaBackend``
    实例。推荐的模式是将一个工厂函数（以 ``run_id`` 为键）作为
    ``backend`` 参数传给 ``create_deep_agent``。
    """

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str,
        snapshot_id: str = "",
        seed_dir: str = "./sandbox_seed",
        sandbox_home_dir: str = DAYTONA_SANDBOX_HOME,
        manager: "SandboxQuotaManager | None" = None,
    ) -> None:
        self._api_key = api_key
        self._api_url = api_url
        self._snapshot_id = snapshot_id
        self._seed_dir = seed_dir
        self.sandbox_home_dir = sandbox_home_dir
        self._default_timeout: int = 30
        self._delegate: DaytonaSandbox | None = None
        self._init_lock = asyncio.Lock()
        self._manager = manager  # 账号级配额管理器；为 None 时退回旧的“直接创建、不限额”行为

    # -- 惰性创建委托对象 ----------------------------------------------

    def _invalidate(self) -> None:
        """配额管理器淘汰了本 backend 对应的沙箱后调用：清空委托，下次访问时惰性重建一个新沙箱。"""
        self._delegate = None

    async def _ensure_delegate(self) -> None:
        if self._delegate is not None:
            if self._manager is not None:
                self._manager.touch(self._delegate.id)  # 刷新活跃时间，避免被当作非活跃淘汰
            return
        async with self._init_lock:
            if self._delegate is not None:
                if self._manager is not None:
                    self._manager.touch(self._delegate.id)
                return
            self._delegate = await self._create_sandbox()
            if self._manager is not None:
                self._manager.touch(self._delegate.id)

    async def _create_sandbox(self) -> DaytonaSandbox:
        if self._manager is not None:
            # 经配额管理器创建：账号级限额 + 打 label + FIFO（跳过活跃）淘汰
            sandbox = await self._manager.create_under_cap(
                snapshot_id=self._snapshot_id,
                backend=self,
            )
        else:
            # 兜底：未接入配额管理器时退回旧行为（直接创建、不限额）
            config = DaytonaConfig(api_key=self._api_key, api_url=self._api_url)
            daytona = AsyncDaytona(config=config)
            if self._snapshot_id:
                from daytona_sdk import CreateSandboxFromSnapshotParams
                params = CreateSandboxFromSnapshotParams(snapshot_id=self._snapshot_id)
                sandbox = await daytona.create(params)
            else:
                sandbox = await daytona.create()
            print(f"[agent] daytona workspace created (id={sandbox.id})")

        # 将公式与场景文件预置到沙箱中
        seed_dir = Path(self._seed_dir)
        resolved = await asyncio.to_thread(seed_dir.resolve)
        if await asyncio.to_thread(resolved.exists):
            uploads = []
            for file_path in await asyncio.to_thread(lambda: list(resolved.rglob("*"))):
                if await asyncio.to_thread(file_path.is_file):
                    rel = str(file_path.relative_to(resolved))
                    content = await asyncio.to_thread(file_path.read_bytes)
                    uploads.append(FileUpload(source=content, destination=f"{self.sandbox_home_dir}{rel}"))
            if uploads:
                await sandbox.fs.upload_files(uploads)
                print(f"[agent] daytona sandbox seeded with {len(uploads)} files")

        return DaytonaSandbox(sandbox, sandbox_home_dir=self.sandbox_home_dir)

    # -- 标识 -----------------------------------------------------------

    @property
    def id(self) -> str:
        return self._delegate.id if self._delegate is not None else "pending"

    # -- execute ------------------------------------------------------------

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        raise RuntimeError(
            "PerRunDaytonaBackend.execute() called synchronously. "
            "Use aexecute() instead."
        )

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        await self._ensure_delegate()
        return await self._delegate.aexecute(command, timeout=timeout)

    # -- ls / grep / glob ----------------------------------------------------

    async def als(self, path: str) -> LsResult:
        await self._ensure_delegate()
        return await self._delegate.als(path)

    async def agrep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        await self._ensure_delegate()
        return await self._delegate.agrep(pattern, path, glob)

    async def aglob(self, pattern: str, path: str = "/") -> GlobResult:
        await self._ensure_delegate()
        return await self._delegate.aglob(pattern, path)

    # -- read / write / edit -------------------------------------------------

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        await self._ensure_delegate()
        return await self._delegate.aread(file_path, offset, limit)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        await self._ensure_delegate()
        return await self._delegate.awrite(file_path, content)

    async def aedit(self, file_path: str, old_string: str, new_string: str,
                    replace_all: bool = False) -> EditResult:
        await self._ensure_delegate()
        return await self._delegate.aedit(file_path, old_string, new_string, replace_all)

    # -- upload / download ---------------------------------------------------

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        raise RuntimeError(
            "PerRunDaytonaBackend.upload_files() called synchronously. "
            "Use aupload_files() instead."
        )

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        await self._ensure_delegate()
        return await self._delegate.aupload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        raise RuntimeError(
            "PerRunDaytonaBackend.download_files() called synchronously. "
            "Use adownload_files() instead."
        )

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        await self._ensure_delegate()
        return await self._delegate.adownload_files(paths)
