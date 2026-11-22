"""最小可运行的 agent 轨迹收集骨架：单样本多轮工具调用 + 批量并发。"""

import asyncio
import inspect
import json
import re
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# LLM 接口：传入 messages，返回一段文本。厂商 SDK 由外部注入。
LLMClient = Callable[[list[dict[str, Any]]], Awaitable[str]]
ToolFn = Callable[..., Any]

SYSTEM_PROMPT = (
    "你可以在需要时调用工具。需要调用时只输出一行 JSON："
    '{"tool": "<工具名>", "arguments": {...}}；'
    "不需要调用时直接输出最终答案。"
)


@dataclass
class ToolCall:
    """一次工具调用。documents 为该次检索命中的文档标识，供后续挖正负样本。"""

    name: str
    arguments: dict[str, Any]
    result: str
    documents: list[str] = field(default_factory=list)


@dataclass
class Trajectory:
    """单条样本的完整轨迹。"""

    prompt: str
    final_answer: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    turns: int = 0
    error: Optional[str] = None
    elapsed: float = 0.0

    @property
    def documents(self) -> list[str]:
        """所有轮命中文档的并集（按首次出现顺序去重）。"""
        seen: set[str] = set()
        merged: list[str] = []
        for call in self.tool_calls:
            for doc in call.documents:
                if doc not in seen:
                    seen.add(doc)
                    merged.append(doc)
        return merged


def _iter_json_objects(text: str):
    """扫出文本里所有括号配平的 {...} 片段（跳过字符串里的括号）。"""
    i, n = 0, len(text)
    while i < n:
        if text[i] == "{":
            depth, in_str, esc, j = 0, False, False, i
            while j < n:
                ch = text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        yield text[i : j + 1]
                        break
                j += 1
            i = j + 1
        else:
            i += 1


_DSML_INVOKE = re.compile(r'invoke\s+name="([^"]+)"')
_DSML_PARAM = re.compile(r'parameter\s+name="([^"]+)"[^>]*>(.*?)</', re.S)


def _parse_dsml(text: str) -> Optional[tuple[str, dict[str, Any]]]:
    """解析 deepseek 的原生工具调用格式：

    <｜｜DSML｜｜ calls>
    <｜｜DSML｜｜ invoke name="search">
    <｜｜DSML｜｜ parameter name="query" string="true">关键词</｜｜DSML｜｜ parameter>
    </｜｜DSML｜｜ invoke>
    </｜｜DSML｜｜ calls>

    这个模型有一半概率走这套格式，不认就会把工具调用当成最终答案。
    """
    invoke = _DSML_INVOKE.search(text)
    if not invoke:
        return None
    tool = invoke.group(1).strip()
    args: dict[str, Any] = {}
    for name, raw in _DSML_PARAM.findall(text):
        value = raw.strip().strip('"').strip("'").strip()
        if value:
            args[name] = value
    return tool, args


def _parse_tool_call(text: str) -> Optional[tuple[str, dict[str, Any]]]:
    """从回复里提取工具调用；没有则返回 None（视为最终答案）。

    两种格式都认：
      1. 约定的 JSON（允许前后有脏字符，例如 <｜｜DSML｜｜...> 尾巴）
      2. 模型原生的 DSML（invoke/parameter 标签）
    """
    s = text.strip()
    if not s:
        return None
    try:
        payload = json.loads(s)
        candidates = [payload]
    except (json.JSONDecodeError, ValueError):
        candidates = []
        for chunk in _iter_json_objects(s):
            try:
                candidates.append(json.loads(chunk))
            except (json.JSONDecodeError, ValueError):
                continue

    for payload in candidates:
        if isinstance(payload, dict) and isinstance(payload.get("tool"), str):
            arguments = payload.get("arguments") or {}
            return payload["tool"], arguments if isinstance(arguments, dict) else {}

    return _parse_dsml(s)


class ToolAgentLoop:
    """单个 prompt 的多轮工具调用循环，全程 async。无 per-run 可变状态，可安全共享。"""

    def __init__(
        self,
        llm: LLMClient,
        tools: Optional[dict[str, ToolFn]] = None,
        max_turns: int = 6,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.tools: dict[str, ToolFn] = dict(tools or {})
        self.max_turns = max_turns
        names = ", ".join(sorted(self.tools)) or "(none)"
        base = system_prompt or SYSTEM_PROMPT
        self.system_prompt = f"{base}\n\n可用工具: {names}"

    async def run(self, prompt: str, max_turns: Optional[int] = None) -> Trajectory:
        """跑完一条轨迹：LLM → 解析工具调用 → 执行工具 → 回填上下文，直到给出答案或轮数用尽。"""
        limit = max_turns if max_turns is not None else self.max_turns
        trajectory = Trajectory(prompt=prompt)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        start = time.perf_counter()
        try:
            for turn in range(1, limit + 1):
                trajectory.turns = turn
                reply = await self.llm(messages)
                messages.append({"role": "assistant", "content": reply})

                parsed = _parse_tool_call(reply)
                if parsed is None:  # 没有工具调用即最终答案
                    trajectory.final_answer = reply.strip()
                    break

                name, arguments = parsed
                call = await self._call_tool(name, arguments)
                trajectory.tool_calls.append(call)
                messages.append({"role": "tool", "name": name, "content": call.result})
        except Exception as exc:  # 保留已完成的部分轨迹，错误只落在本条上
            trajectory.error = f"{type(exc).__name__}: {exc}"
        finally:
            trajectory.elapsed = time.perf_counter() - start
        return trajectory

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCall:
        """同步工具放进线程池执行，异常转成工具结果回给模型。"""
        tool = self.tools.get(name)
        if tool is None:
            return ToolCall(name, arguments, f"error: unknown tool '{name}'")

        # 模型常会多给参数（比如 max_results），按签名过滤掉，别让整次调用白跑
        try:
            sig = inspect.signature(tool)
            accepts_kwargs = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
            if not accepts_kwargs:
                allowed = {
                    p.name
                    for p in sig.parameters.values()
                    if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                }
                arguments = {k: v for k, v in arguments.items() if k in allowed}
        except (TypeError, ValueError):
            pass

        try:
            raw = await asyncio.to_thread(tool, **arguments)
        except Exception as exc:
            return ToolCall(name, arguments, f"error: {type(exc).__name__}: {exc}")

        if isinstance(raw, dict):  # 约定：{"text": ..., "documents": [...]}
            text = str(raw.get("text", ""))
            documents = [str(doc) for doc in raw.get("documents") or []]
        else:
            text = str(raw)
            documents = []
        return ToolCall(name, arguments, text, documents)


class AgentWorker:
    """批量跑 prompt：每个 prompt 一个 task，Semaphore 限流，单条失败不拖垮整批。"""

    def __init__(
        self,
        llm: LLMClient,
        tools: Optional[dict[str, ToolFn]] = None,
        max_concurrency: int = 8,
        max_turns: int = 6,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.max_concurrency = max_concurrency
        self.agent = ToolAgentLoop(
            llm=llm, tools=tools, max_turns=max_turns, system_prompt=system_prompt
        )

    async def run_batch(self, prompts: list[str]) -> list[Trajectory]:
        """并发跑一批 prompt，返回与 prompts 顺序一致的轨迹列表。"""
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run_one(prompt: str) -> Trajectory:
            async with semaphore:
                try:
                    return await self.agent.run(prompt)
                except Exception as exc:  # 兜底：run 内部已自己记录错误
                    logger.exception("trajectory failed: %s", prompt[:50])
                    return Trajectory(prompt=prompt, error=f"{type(exc).__name__}: {exc}")

        tasks = [asyncio.create_task(run_one(prompt)) for prompt in prompts]
        return await asyncio.gather(*tasks)


class MockLLM:
    """假客户端：按 assistant 轮数依次发出检索调用，检索完给出最终答案。"""

    def __init__(
        self,
        queries: Optional[list[str]] = None,
        answer: str = "mock final answer",
    ) -> None:
        self.queries = list(queries or ["mock query"])
        self.answer = answer

    async def __call__(self, messages: list[dict[str, Any]]) -> str:
        await asyncio.sleep(0)  # 占位，保持真实客户端的 await 语义
        used = sum(1 for message in messages if message["role"] == "assistant")
        if used < len(self.queries):
            return json.dumps(
                {"tool": "search", "arguments": {"query": self.queries[used]}},
                ensure_ascii=False,
            )
        return self.answer


async def _smoke() -> None:
    def fake_search(query: str, top_k: int = 2) -> dict[str, Any]:
        time.sleep(0.01)  # 模拟同步阻塞的检索
        documents = [f"doc{i}:{query}" for i in range(1, top_k + 1)]
        return {"text": " / ".join(documents), "documents": documents}

    mock = MockLLM(queries=["agentic search", "query adapter"], answer="最终答案：query 侧适配")

    async def llm(messages: list[dict[str, Any]]) -> str:
        if "崩溃" in messages[1]["content"]:
            raise RuntimeError("fake llm failure")
        return await mock(messages)

    worker = AgentWorker(llm=llm, tools={"search": fake_search}, max_concurrency=2, max_turns=4)
    prompts = ["agentic search 是什么？", "这条会崩溃", "为什么要冻结文档侧？"]
    trajectories = await worker.run_batch(prompts)

    for trajectory in trajectories:
        print(
            f"prompt={trajectory.prompt!r} turns={trajectory.turns} "
            f"tool_calls={len(trajectory.tool_calls)} docs={len(trajectory.documents)} "
            f"error={trajectory.error} elapsed={trajectory.elapsed:.3f}s"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_smoke())
