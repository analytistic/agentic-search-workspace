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
    """True inside a ray worker process.

    Parsing /proc/self/cmdline turned out to be racy -- ray rewrites the
    proctitle via setproctitle while we are reading it, so the same process
    could look like an IDLE worker in one call and an AgentLoopWorker in the
    next (observed: the log printed a cmdline containing AgentLoopWorker while
    the predicate had just returned False). global_worker.mode is set once at
    worker startup and does not move: 0 = driver, 1 = worker.
    """
    try:
        from ray._private.worker import global_worker

        return int(getattr(global_worker, "mode", -1)) == 1
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 1. IDE attach
# ---------------------------------------------------------------------------
def _arm_debugger() -> None:
    """Called from the watcher, i.e. once ray has renamed this process."""
    if os.environ.get("EVAL_DEBUG_WORKER") != "1":
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


def _dump_input(loop, agent_data, sampling_params=None) -> None:
    tok = _find_tokenizer(loop)
    ids = _to_list(agent_data.prompt_ids) if agent_data.prompt_ids is not None else []
    msgs = agent_data.messages or []
    _w(f"--- turn={agent_data.assistant_turns} INPUT ---")
    _w(f"  prompt_len={len(ids)}")
    _w(f"  prompt_ids[:32]={ids[:32]}")
    _w(f"  prompt_head={_dec(tok, ids[:160])!r}")
    _w(f"  prompt_tail={_dec(tok, ids[-64:])!r}")

    # Sampling knobs matter here: an empty completion usually means the engine
    # hit a stop condition immediately, so dump what we actually asked for.
    sp = dict(sampling_params or {})
    stop = sp.pop("stop_token_ids", None)
    _w(f"  sampling_params={sp}")
    _w(f"  sampling.stop_token_ids={stop}")
    parser = getattr(loop, "tool_parser", None)
    _w(f"  parser={type(parser).__name__} parser.stop_token_ids={getattr(parser, 'stop_token_ids', None)}")

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
    for attr in ("stop_reason", "finish_reason", "num_preempted", "log_probs"):
        if hasattr(output, attr):
            val = getattr(output, attr)
            if attr == "log_probs" and val is not None:
                try:
                    val = f"len={len(val)}"
                except TypeError:
                    val = "<unlenable>"
            _w(f"  out.{attr}={val!r}")


def _install_trace(module) -> None:
    cls = getattr(module, "ToolAgentLoop", None)
    if cls is None or getattr(cls, "_eval_traced", False):
        return

    orig = cls._handle_generating_state

    async def traced(self, agent_data, sampling_params, *args, **kwargs):
        try:
            _dump_input(self, agent_data, sampling_params)
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


def _install_sglang_probe() -> None:
    """Probe sglang's tokenizer manager to see the raw completion.

    verl calls ``TokenizerManager.generate_request`` in-process (there is no
    HTTP port for the rollout engine), so this is the only place where the
    engine's own ``finish_reason`` is visible. Without it, a zero-length
    completion is indistinguishable from a stop-token hit.
    """
    try:
        from sglang.srt.managers.tokenizer_manager import TokenizerManager
    except Exception as exc:
        _w(f"[sglang] import failed: {exc!r}")
        return

    if getattr(TokenizerManager, "_eval_probed", False):
        return

    orig = TokenizerManager.generate_request

    def wrapped(self, obj, request=None, *args, **kwargs):
        gen = orig(self, obj, request, *args, **kwargs)

        async def agen():
            async for out in gen:
                try:
                    oids = out.get("output_ids") or []
                    meta = out.get("meta_info") or {}
                    otl = meta.get("output_token_logprobs")
                    first = None
                    if otl:
                        first = (type(otl[0]).__name__, len(otl[0]) if hasattr(otl[0], "__len__") else "?", repr(otl[0])[:80])
                    req_sp = getattr(obj, "sampling_params", None)
                    _w(
                        f"  [sglang] n_out={len(oids)} "
                        f"n_logprobs={len(otl) if otl is not None else 'MISSING'} "
                        f"first={first} "
                        f"finish={meta.get('finish_reason')} "
                        f"req_max_new={getattr(req_sp, 'max_new_tokens', None)}"
                    )
                except Exception as exc:
                    _w(f"  [sglang] probe failed: {exc!r}")
                yield out

        return agen()

    TokenizerManager.generate_request = wrapped
    TokenizerManager._eval_probed = True
    _w("[sglang] probed TokenizerManager.generate_request")


def _probe_sglang_when_ready() -> None:
    """sglang is imported after the actor starts, so wait for it."""
    deadline = time.time() + 180.0
    while time.time() < deadline:
        if "sglang.srt.managers.tokenizer_manager" in sys.modules:
            _install_sglang_probe()
            return
        time.sleep(0.2)
    _w("[sglang] tokenizer_manager never showed up")


def _install_server_generate_probe() -> None:
    """Wrap SGLangHttpServer.generate: this is where TokenOutput is built.

    If the logprob payload has an unexpected shape, the list comprehension
    inside it raises and the caller ends up with an empty TokenOutput while
    sglang itself reported thousands of tokens.
    """
    try:
        from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer
    except Exception as exc:
        _w(f"[server] import failed: {exc!r}")
        return
    if getattr(SGLangHttpServer, "_eval_probed", False):
        return

    orig = SGLangHttpServer.generate

    async def wrapped(self, *args, **kwargs):
        try:
            out = await orig(self, *args, **kwargs)
        except Exception as exc:
            _w(f"[server.generate] RAISED {type(exc).__name__}: {exc}")
            raise
        tids = getattr(out, "token_ids", None)
        _w(
            f"[server.generate] token_ids={len(tids) if tids is not None else None} "
            f"stop={getattr(out, 'stop_reason', None)}"
        )
        return out

    SGLangHttpServer.generate = wrapped
    SGLangHttpServer._eval_probed = True
    _w("[server] probed SGLangHttpServer.generate")


def _watch_for_module() -> None:
    """Wait for the modules of interest, then arm the matching hooks.

    Two different Ray actors matter and they import different modules:

    * AgentLoopWorker  -> verl.experimental.agent_loop.tool_agent_loop
      (where _handle_generating_state lives)
    * SGLangHttpServer -> verl.workers.rollout.sglang_rollout.async_sglang_server
      (the actor that actually owns the sglang engine; the agent loop only
      reaches it through ``server.generate.remote(...)`` in llm_server.py, so
      probing the tokenizer manager inside the agent loop process never fires)

    Deferring the check also matters for the proctitle: ray only renames the
    process once the actor is scheduled, which is after these imports.
    """
    loop_mod = "verl.experimental.agent_loop.tool_agent_loop"
    server_mod = "verl.workers.rollout.sglang_rollout.async_sglang_server"

    deadline = time.time() + 180.0
    trace_done = False
    probe_done = False
    while time.time() < deadline:
        if not trace_done:
            module = sys.modules.get(loop_mod)
            if module is not None and getattr(module, "ToolAgentLoop", None) is not None:
                trace_done = True
                _w(f"[hook] agent-loop actor: {_cmdline()[:90]!r}")
                try:
                    _install_trace(module)
                except Exception as exc:
                    _w(f"[hook] trace install failed: {exc!r}")
                threading.Thread(target=_arm_debugger, daemon=True).start()

        if not probe_done and server_mod in sys.modules:
            probe_done = True
            _w(f"[hook] rollout-server actor: {_cmdline()[:90]!r}")
            try:
                _install_server_generate_probe()
            except Exception as exc:
                _w(f"[hook] server probe failed: {exc!r}")
            threading.Thread(target=_probe_sglang_when_ready, daemon=True).start()

        if trace_done and probe_done:
            return
        time.sleep(0.05)

    _w(f"[hook] timeout (trace={trace_done} probe={probe_done}) cmd={_cmdline()[:90]!r}")


def _arm_watcher() -> None:
    want_debug = os.environ.get("EVAL_DEBUG_WORKER") == "1"
    want_trace = os.environ.get("EVAL_TRACE_GEN") == "1"
    if not (want_debug or want_trace):
        return
    threading.Thread(target=_watch_for_module, daemon=True).start()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(0)

try:
    _arm_watcher()
except Exception as exc:  # never break the job because of a debug hook
    print(f"[eval-hook] arm failed: {exc}", flush=True)
