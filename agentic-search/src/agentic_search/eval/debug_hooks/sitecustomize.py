"""Sitecustomize hook that lets an IDE (VSCode / PyCharm) attach to the Ray
actor that runs the agent loop.

Why this exists
---------------
``ToolAgentLoop`` / ``AgentLoopWorker`` execute inside a Ray actor, i.e. a
separate Python process spawned by raylet. An IDE debugger can only instrument
processes it starts itself (or a child process it can patch via
``subprocess``/``multiprocessing`` -- Ray workers are neither), so "just hit F5
and break inside the loop" does not work out of the box.

Ray's own debugger (``ray.util.rpdb.set_trace()`` + ``ray debug``) is one
option; this hook is the other: it makes the actor open a regular debugpy port
so the IDE's "Remote Attach" can connect with full breakpoints/variables.

How it is loaded
----------------
Ray workers are normal CPython processes and therefore run ``sitecustomize``
on startup. ``run_eval_debug.sh`` puts this directory first on ``PYTHONPATH``
when ``WORKER_DEBUG=1``, so every process in the job imports this module --
but it only arms itself inside an ``AgentLoopWorker`` process, which is why
the ``/proc/self/cmdline`` check below exists.

Env vars
--------
EVAL_DEBUG_WORKER=1   enable (set by the launcher)
EVAL_DEBUG_PORT=5679  port to listen on
EVAL_DEBUG_WAIT=1     block until a client attaches (default: no)
"""

import os


def _cmdline() -> str:
    """Best-effort command line of the current process."""
    try:
        with open("/proc/self/cmdline", "rb") as fh:
            return fh.read().replace(b"\x00", b" ").decode("utf-8", "ignore")
    except OSError:
        return ""


def _arm() -> None:
    if os.environ.get("EVAL_DEBUG_WORKER") != "1":
        return

    # Only the agent-loop actor should take the port. sglang's scheduler
    # subprocesses, the reward workers and the driver all import this module
    # too, and they must stay silent.
    if "AgentLoopWorker" not in _cmdline():
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


_arm()
