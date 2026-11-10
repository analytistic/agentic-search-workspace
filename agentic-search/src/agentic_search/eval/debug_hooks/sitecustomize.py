"""Sitecustomize hooks for debugging the agent loop inside its Ray actor.

Two independent features, both off unless the launcher asks for them:

1. IDE attach (``EVAL_DEBUG_WORKER=1``)
   ``ToolAgentLoop`` / ``AgentLoopWorker`` run inside a Ray actor, i.e. a
   process raylet spawns. An IDE debugger can only instrument processes it
   starts itself (or children patched via subprocess/multiprocessing -- Ray
   workers are neither), so "F5 on rollout_eval.py" can never break inside the
   loop. This hook makes the actor open a plain debugpy port instead, so the
   IDE's "Remote Attach" works with full breakpoints.

2. Generation trace (``EVAL_TRACE_GEN=1``)
   Wraps ``ToolAgentLoop._handle_generating_state`` and dumps, per turn, the
   exact ``prompt_ids`` handed to the inference engine and the ``token_ids``
   that come back. This is the "just print it" alternative to a breakpoint and
   is what you want when the question is "is the input wrong, or is the engine
   returning garbage?".

Why sitecustomize: Ray workers are ordinary CPython processes and therefore
execute ``sitecustomize`` at startup. ``run_eval_debug.sh`` puts this directory
first on ``PYTHONPATH``, so every process in the job imports this module -- but
both features arm themselves only where ``/proc/self/cmdline`` contains
``AgentLoopWorker``, so the driver, sglang's scheduler children and the reward
workers stay silent.

Env vars
--------
EVAL_DEBUG_WORKER=1      enable IDE attach
EVAL_DEBUG_PORT=5679     port for debugpy
EVAL_DEBUG_WAIT=1        block until the IDE attaches (default: no)

EVAL_TRACE_GEN=1         enable generation trace
EVAL_TRACE_FILE=<path>   where to write it (default /root/eval_trace.log)
EVAL_TRACE_MAX_CHARS=N   chars of decoded text per dump (default 400)
"""

import os
import sys
import threading
import time


def _cmdline() -> str:
    """Best-effort command line of the current process."""
    try:
        with open("/proc/self/cmdline", "rb") as fh:
            return fh.read().replace(b"\x00", b" ").decode("utf-8", "ignore")
    except OSError:
        return ""


def _is_agent_loop_actor() -> bool:
    return "AgentLoopWorker" in _cmdline()


# ---------------------------------------------------------------------------
# 1. IDE attach
# ---------------------------------------------------------------------------
def _arm_debugger() -> None:
    if os.environ.get("EVAL_DEBUG_WORKER") != "1":
        return
    if not _is_agent_loop_actor():
        return

    try:
        import debugpy
    except ImportError:
        print("[eval-debug] debugpy not installed; skipping", flush=True)
        return

    port = int(os.environ.get("EVAL_DEBUG_PORT", "5679"))
    try:
        debugpy.listen(("0.0.0.0", port))
    except Exception as exc:  # port already taken (e.g. a second worker)
        print(f"[eval-debug] listen on :{port} failed: {exc}", flush=True)
        return

    print(
        f"[eval-debug] debugpy listening on 0.0.0.0:{port} (pid {os.getpid()})",
        flush=True,
    )
    if os.environ.get("EVAL_DEBUG_WAIT") == "1":
        print("[eval-debug] waiting for the IDE to attach...", flush=True)
        debugpy.wait_for_client()
        print("[eval-debug] client attached", flush=True)


# ---------------------------------------------------------------------------
# 2. Generation trace
# ---------------------------------------------------------------------------
_TRACE_PATH = os.environ.get("EVAL_TRACE_FILE", "/root/eval_trace.log")
_TRACE_MAX = int(os.environ.get("EVAL_TRACE_MAX_CHARS", "400"))


def _w(msg: str) -> None:
    line = f"[pid {os.getpid()}] {msg}\n"
    try:
        with open(_TRACE_PATH, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        pass
    try:
        sys.stdout.write(line)
        sys.stdout.flush()
    except Exception:
        pass


def _find_tokenizer(obj):
    """ToolAgentLoop keeps its tokenizer under one of a few attributes."""
    for name in ("tokenizer", "processor", "tokenizer_manager"):
        cand = getattr(obj, name, None)
        if cand is None:
            continue
        if hasattr(cand, "decode"):
            return cand
        inner = getattr(cand, "tokenizer", None)
        if inner is not None and hasattr(inner, "decode"):
            return inner
    server = getattr(obj, "server_manager", None)
    if server is not None:
        inner = getattr(server, "tokenizer", None)
        if inner is not None and hasattr(inner, "decode"):
            return inner
    return None


def _dec(tok, ids) -> str:
    if tok is None:
        return "<no tokenizer>"
    try:
        text = tok.decode(list(ids), skip_special_tokens=False)
    except Exception as exc:
        return f"<decode failed: {exc}>"
    if len(text) > _TRACE_MAX:
        text = text[:_TRACE_MAX] + f"...<+{len(text) - _TRACE_MAX}>"
    return text


def _to_list(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def _dump_input(loop, agent_data) -> None:
    tok = _find_tokenizer(loop)
    ids = _to_list(agent_data.prompt_ids) if agent_data.prompt_ids is not None else []
    msgs = agent_data.messages or []
    _w(f"--- turn={agent_data.assistant_turns} INPUT ---")
    _w(f"  prompt_len={len(ids)}")
    _w(f"  prompt_ids[:32]={ids[:32]}")
    _w(f"  prompt_head={_dec(tok, ids[:160])!r}")
    _w(f"  prompt_tail={_dec(tok, ids[-64:])!r}")
    _w(f"  msg_roles={[m.get('role') for m in msgs]}")
    if msgs:
        tail = msgs[-1]
        content = tail.get("content")
        if isinstance(content, (list, dict)):
            content = str(content)
        _w(f"  last_msg({tail.get('role')})={str(content)[:300]!r}")


def _dump_output(loop, agent_data, output) -> None:
    tok = _find_tokenizer(loop)
    ids = _to_list(output.token_ids) if getattr(output, "token_ids", None) is not None else []
    _w(f"--- turn={agent_data.assistant_turns} OUTPUT ---")
    _w(f"  out_len={len(ids)}")
    _w(f"  out_ids[:32]={ids[:32]}")
    _w(f"  out_head={_dec(tok, ids[:160])!r}")
    _w(f"  out_tail={_dec(tok, ids[-48:])!r}")


def _install_trace(module) -> None:
    cls = getattr(module, "ToolAgentLoop", None)
    if cls is None or getattr(cls, "_eval_traced", False):
        return

    orig = cls._handle_generating_state

    async def traced(self, agent_data, sampling_params, *args, **kwargs):
        try:
            _dump_input(self, agent_data)
        except Exception as exc:
            _w(f"  input dump failed: {exc!r}")
        out = await orig(self, agent_data, sampling_params, *args, **kwargs)
        try:
            _dump_output(self, agent_data, out)
        except Exception as exc:
            _w(f"  output dump failed: {exc!r}")
        return out

    cls._handle_generating_state = traced
    cls._eval_traced = True
    _w("[trace] hooked ToolAgentLoop._handle_generating_state")


def _watch_for_module() -> None:
    """Poll sys.modules -- the actor imports verl after sitecustomize runs."""
    target = "verl.experimental.agent_loop.tool_agent_loop"
    deadline = time.time() + 180.0
    while time.time() < deadline:
        module = sys.modules.get(target)
        if module is not None and getattr(module, "ToolAgentLoop", None) is not None:
            _install_trace(module)
            return
        time.sleep(0.02)
    _w("[trace] gave up waiting for tool_agent_loop to be imported")


def _arm_trace() -> None:
    if os.environ.get("EVAL_TRACE_GEN") != "1":
        return
    if not _is_agent_loop_actor():
        return
    _w(f"[trace] armed (file={_TRACE_PATH})")
    threading.Thread(target=_watch_for_module, daemon=True).start()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(0)

try:
    _arm_debugger()
except Exception as exc:  # never break the job because of a debug hook
    print(f"[eval-debug] arm failed: {exc}", flush=True)

try:
    _arm_trace()
except Exception as exc:
    print(f"[eval-trace] arm failed: {exc}", flush=True)
