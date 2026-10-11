"""Three specialized Strands agents for testing and demoing the app.

Speed-oriented design (all aimed at CPU-only inference):
  * thinking mode is switched off for Qwen3 (it can emit thousands of hidden tokens per turn)
  * small context window, hard cap on generated tokens per agent, model kept loaded
  * EVERY phase uses ONE coarse tool; the code gathers the evidence, the model only summarises
  * every tool returns compact JSON (see tools.py)
  * a per-run tool-call budget stops a small model from looping

Why the end-to-end agent used to fail: it had four fine-grained tools plus a long
multi-step prompt, so a 4B model kept writing long tool calls / plans and hit
max_tokens (MaxTokensReachedException).  It now has a single tool, `e2e_journey`,
exactly like phases 1 and 2.
"""
import os
import time

import httpx
from pydantic import BaseModel, Field
from strands import Agent
from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent
from strands.models.ollama import OllamaModel

from tools import api_conformance_suite, e2e_journey, ui_smoke_test

# ── LLM settings (all overridable from .env) ──────────────────────────────


def _env(name: str, default: str) -> str:
    return (os.getenv(name) or default).strip()


def model_id() -> str:
    return _env("OLLAMA_MODEL", "qwen3:4b-instruct")


def ollama_host() -> str:
    return _env("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _num_ctx() -> int:
    return int(_env("OLLAMA_NUM_CTX", "4096"))


def _keep_alive() -> str:
    return _env("OLLAMA_KEEP_ALIVE", "30m")


def _ollama_options() -> dict:
    options: dict = {"num_ctx": _num_ctx()}
    threads = _env("OLLAMA_NUM_THREAD", "")
    if threads.isdigit():
        options["num_thread"] = int(threads)
    return options


def _think_args() -> dict:
    """Qwen3 'thinking' models burn most of their time on hidden reasoning. Turn it off."""
    setting = _env("OLLAMA_THINK", "auto").lower()
    if setting in ("0", "false", "off", "no"):
        return {"think": False}
    mid = model_id().lower()
    if setting == "auto" and "qwen3" in mid and "instruct" not in mid:
        return {"think": False}
    return {}


def _model(max_tokens: int) -> OllamaModel:
    config = dict(
        model_id=model_id(),
        temperature=0.0,
        max_tokens=max_tokens,
        keep_alive=_keep_alive(),
        options=_ollama_options(),
    )
    think = _think_args()
    if think:
        config["additional_args"] = think
    return OllamaModel(
        host=ollama_host(),
        ollama_client_args={"timeout": float(_env("LLM_TIMEOUT_S", "600"))},
        **config,
    )


# ── Ollama helpers used by the orchestrator (no agent needed) ─────────────


def _model_matches(wanted: str, installed: str) -> bool:
    return installed == wanted or installed == f"{wanted}:latest" or (":" not in wanted and installed.startswith(f"{wanted}:"))


async def ollama_preflight() -> tuple[bool, str]:
    """Is Ollama up and is the model pulled? Returns (ok, message)."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{ollama_host()}/api/tags")
        names = [m.get("name", "") for m in r.json().get("models", [])]
    except Exception as exc:  # noqa: BLE001
        return False, f"Ollama is not reachable at {ollama_host()} ({type(exc).__name__}). Start it with:  ollama serve"
    if not any(_model_matches(model_id(), n) for n in names):
        return False, f"Model '{model_id()}' is not pulled. Run:  ollama pull {model_id()}   (installed: {', '.join(names) or 'none'})"
    return True, f"Ollama OK, model '{model_id()}' is installed"


async def warm_up_model() -> str:
    """Load the model into RAM once, up front, with the same options the agents use."""
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=float(_env("LLM_TIMEOUT_S", "600"))) as client:
            await client.post(f"{ollama_host()}/api/generate", json={
                "model": model_id(), "prompt": "", "stream": False,
                "keep_alive": _keep_alive(), "options": _ollama_options(),
            })
            ps = (await client.get(f"{ollama_host()}/api/ps")).json().get("models", [])
    except Exception as exc:  # noqa: BLE001
        return f"warm-up skipped ({type(exc).__name__}); the first agent call will load the model instead"
    where = "CPU"
    for m in ps:
        if _model_matches(model_id(), m.get("name", "")) or m.get("name", "").startswith(model_id()):
            size, vram = m.get("size", 0) or 0, m.get("size_vram", 0) or 0
            if size and vram:
                where = f"GPU {round(100 * vram / size)}%"
    return f"model loaded in {time.monotonic() - t0:.1f}s, running on {where}"


# ── Agent construction ────────────────────────────────────────────────────

MAX_TOOL_CALLS = int(_env("MAX_TOOL_CALLS", "8"))
SUMMARY_MAX_TOKENS = int(_env("SUMMARY_MAX_TOKENS", "900"))


def _build_agent(tools: list, system_prompt: str, max_tokens: int = SUMMARY_MAX_TOKENS) -> Agent:
    """Shared constructor: model, tools, prompt, silent console, tool-call trace + budget."""
    agent = Agent(
        model=_model(max_tokens),
        tools=tools,
        system_prompt=system_prompt,
        callback_handler=None,
    )
    calls = {"n": 0}

    def _before(event: BeforeToolCallEvent) -> None:
        calls["n"] += 1
        name = event.tool_use.get("name", "?")
        args = event.tool_use.get("input", {})
        preview = ", ".join(f"{k}={str(v)[:40]}" for k, v in args.items())
        print(f"  → {name}({preview})")
        if calls["n"] > MAX_TOOL_CALLS:
            event.cancel_tool = "Tool-call budget used up. Stop calling tools and write your final answer now."

    def _after(event: AfterToolCallEvent) -> None:
        duration = getattr(event, "duration", None)
        if duration is not None:
            print(f"  ← {event.tool_use.get('name', '?')} done in {duration:.1f}s")

    agent.add_hook(_before)
    agent.add_hook(_after)
    return agent


# ── Report schemas (filled by the orchestrator from tool evidence) ─────────

class CheckResult(BaseModel):
    name: str = Field(description="Short name of the check")
    passed: bool = Field(description="True if the API behaved as documented")
    detail: str = Field(description="One-sentence evidence from the response")


class ConformanceReport(BaseModel):
    endpoint_checks: list[CheckResult]
    degradation_checks: list[CheckResult]
    overall: str = Field(description="'pass', 'partial', or 'fail'")
    summary: str


class UIStep(BaseModel):
    step: int
    action: str = Field(description="What was done, e.g. 'entered city Kansas City, MO'")
    passed: bool
    page_text_snippet: str = Field(description="Evidence seen on the page after this action, or '' if none")
    error: str | None = Field(default=None, description="Error message if the step failed")


class UIAuditReport(BaseModel):
    steps: list[UIStep]
    overall: str = Field(description="'pass', 'partial', or 'fail'")
    summary: str


class E2EReport(BaseModel):
    checks: list[CheckResult]
    overall: str = Field(description="'pass', 'partial', or 'fail'")
    summary: str


# ── Phase 1: API conformance ──────────────────────────────────────────────

API_TESTER_PROMPT = """
You are an API conformance tester.

Call the tool api_conformance_suite exactly once. It runs every check
(health, geocode, events search, travel plan and graceful-degradation checks)
and returns each check with evidence. Do not call any other tool.

Then answer in at most two plain sentences: how many checks passed out of the
total, and, for every failed check, its name and the evidence from the tool
result. If the tool says the backend is unreachable, say so. Never invent values.
"""


def build_api_tester() -> Agent:
    return _build_agent([api_conformance_suite], API_TESTER_PROMPT)


# ── Phase 2: Frontend QA (real browser interaction) ───────────────────────

FRONTEND_TESTER_PROMPT = """
You are a frontend QA tester with a real browser.

Call the tool ui_smoke_test exactly once. It drives the real events web app in
one browser session (load the page, find the keyword and city boxes, enter a
city, check event cards render, check the browser can reach the backend, run a
keyword search, check the empty state, clear the search, open an event) and
returns every step with pass/fail and the evidence it saw. Do not call any
other tool.

Then answer in at most three plain sentences: the overall result, how many steps
passed out of the total, and, if a step failed, its action and exact error.
Never invent values - quote what the tool returned.
"""


def build_frontend_tester() -> Agent:
    return _build_agent([ui_smoke_test], FRONTEND_TESTER_PROMPT)


# ── Phase 3: End-to-end ───────────────────────────────────────────────────

E2E_TESTER_PROMPT = """
You are an end-to-end tester.

Call the tool e2e_journey exactly once. It runs the whole journey in code:
geocode a city, search events, build a travel plan over the API, then open the
real web app in a browser, enter the same city, and open an event. It returns
every check with pass/fail and evidence. Do not call any other tool.

Then answer in at most three plain sentences: the overall result, how many
checks passed out of the total, and, for the first failed check, its name and
the exact evidence. Never invent values.
"""


def build_e2e_tester() -> Agent:
    return _build_agent([e2e_journey], E2E_TESTER_PROMPT)