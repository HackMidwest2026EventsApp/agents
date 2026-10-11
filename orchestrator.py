"""Run selected agents and print a consolidated report.

Usage:
    python orchestrator.py            # asks which phases (Enter = quick: 1 and 2)
    python orchestrator.py 1 2        # API conformance + Frontend QA
    python orchestrator.py all        # everything
    python orchestrator.py all --no-llm   # same checks, no language model at all (seconds)
"""
import asyncio
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # read .env BEFORE importing modules that use os.getenv

import httpx  # noqa: E402

import tools  # noqa: E402
from agents import (  # noqa: E402
    CheckResult,
    ConformanceReport,
    E2EReport,
    UIAuditReport,
    UIStep,
    _env,
    build_api_tester,
    build_e2e_tester,
    build_frontend_tester,
    model_id,
    ollama_preflight,
    warm_up_model,
)

PHASE_TIMEOUT_S = float(_env("PHASE_TIMEOUT_S", "420"))

# What each phase needs to be up before it is worth starting.
NEEDS = {"1": {"backend"}, "2": {"frontend"}, "3": {"backend", "frontend"}}


# ── Pretty printers ────────────────────────────────────────────────────────

def _print_checks(title: str, checks: list[CheckResult]) -> None:
    if checks:
        print(f"{title}:")
        for c in checks:
            print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}")
            print(f"         {c.detail}")
        print()


def print_conformance_report(report: ConformanceReport) -> None:
    print(f"\nOverall: {report.overall.upper()}")
    print(f"Summary: {report.summary}\n")
    _print_checks("Endpoint checks", report.endpoint_checks)
    _print_checks("Degradation checks", report.degradation_checks)


def print_ui_audit(report: UIAuditReport) -> None:
    print(f"\nOverall: {report.overall.upper()}")
    print(f"Summary: {report.summary}\n")
    print("Steps:")
    for s in report.steps:
        print(f"  [{'PASS' if s.passed else 'FAIL'}] step {s.step}: {s.action}")
        if s.page_text_snippet:
            print(f"         page: {s.page_text_snippet[:160]}")
        if s.error:
            print(f"         error: {s.error}")


def print_e2e_report(report: E2EReport) -> None:
    print(f"\nOverall: {report.overall.upper()}")
    print(f"Summary: {report.summary}\n")
    _print_checks("Journey checks", report.checks)


# ── Pre-flight: fail fast, before any slow model work ───────────────────────

async def _probe(url: str, timeout: float, retries: int = 1) -> tuple[bool, str]:
    last = ""
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                r = await client.get(url)
            if r.status_code < 400:
                return True, f"HTTP {r.status_code}"
            return False, f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001
            last = type(exc).__name__
            if attempt < retries:
                await asyncio.sleep(1.5)
    return False, last


async def preflight(selected: list[str], use_llm: bool) -> tuple[dict[str, str], bool]:
    """Returns ({'backend'|'frontend': problem}, use_llm). An empty dict means everything needed is up."""
    needed = set().union(*(NEEDS[p] for p in selected))
    problems: dict[str, str] = {}

    if "backend" in needed:
        ok, info = await _probe(f"{tools.api_base()}/health", timeout=8)
        if ok:
            print(f"  backend   OK    {tools.api_base()}/health ({info})")
        else:
            problems["backend"] = (f"{tools.api_base()}/health -> {info}. Check API_BASE_URL. "
                                   "If it is local, start it:  uvicorn app.main:app --reload  "
                                   "(and make sure Postgres is up:  docker compose up -d)")
            print(f"  backend   DOWN  {problems['backend']}")

    if "frontend" in needed:
        print(f"  frontend  ...   warming {tools.ui_url()}")
        ok, info = await _probe(tools.ui_url(), timeout=120)
        if ok:
            print(f"  frontend  OK    {tools.ui_url()} ({info})")
        else:
            hint = ("Check FRONTEND_BASE_URL (the deployed site is https://hackmidwest2026eventsapp.github.io/events-ui) "
                    "and that GitHub Pages is serving it."
                    if info != "HTTP 404" else
                    "404 means the wrong path. Put the sub-path in FRONTEND_BASE_URL and leave UI_PATH=/ . "
                    "'/ui/' only exists on the backend (port 8000).")
            problems["frontend"] = f"{tools.ui_url()} -> {info}. {hint}"
            print(f"  frontend  DOWN  {problems['frontend']}")

    if use_llm:
        ok, msg = await ollama_preflight()
        print(f"  ollama    {'OK  ' if ok else 'DOWN'}  {msg}")
        if not ok:
            print("            -> continuing WITHOUT the language model: all phases run as plain checks")
            use_llm = False
        else:
            print(f"  ollama    ...   {await warm_up_model()}")
    return problems, use_llm


# ── Running an agent safely ─────────────────────────────────────────────────

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _usage(result) -> str:
    try:
        s = result.metrics.get_summary()
        u = s.get("accumulated_usage", {})
        return f"{u.get('inputTokens', '?')} tokens in / {u.get('outputTokens', '?')} out, {s.get('total_cycles', '?')} model call(s)"
    except Exception:  # noqa: BLE001
        return ""


async def run_agent(agent, prompt: str) -> tuple[str | None, str]:
    """Returns (text, problem). Never raises; a timeout or error (including
    MaxTokensReachedException) comes back as `problem`."""
    t0 = time.monotonic()
    try:
        result = await asyncio.wait_for(agent.invoke_async(prompt), timeout=PHASE_TIMEOUT_S)
        text = _THINK.sub("", str(result)).strip()
        print(f"  [agent finished in {time.monotonic() - t0:.0f}s; {_usage(result)}]")
        return text, ""
    except asyncio.TimeoutError:
        return None, f"agent timed out after {PHASE_TIMEOUT_S:.0f}s"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc)[:200]}"


# ── Reports built from the tools' evidence (+ the model's summary) ──────────

def _overall(passed: int, total: int) -> str:
    if total == 0 or passed == 0:
        return "fail"
    return "pass" if passed == total else "partial"


def _auto_summary(checks: list, label: str) -> str:
    failed = [c for c in checks if not c["passed"]]
    head = f"{len(checks) - len(failed)}/{len(checks)} {label} passed."
    return head if not failed else head + " Failed: " + "; ".join(f"{c['name']} ({c['detail']})" for c in failed[:3])


async def phase_api(use_llm: bool) -> tuple[str, str]:
    tools.LAST_RESULTS.pop("api_suite", None)
    llm_summary, problem = "", ""
    if use_llm:
        llm_summary, problem = await run_agent(build_api_tester(), "Run the API conformance suite and summarise the result.")
        llm_summary = llm_summary or ""
        if problem:
            print(f"  [{problem} - using the recorded check results instead]")
    data = tools.LAST_RESULTS.get("api_suite")
    if data is None:  # model never called the tool, or no LLM
        data = await tools.run_api_suite()
    ep, dg = data["endpoint_checks"], data["degradation_checks"]
    allc = ep + dg
    overall = "fail" if data.get("aborted") else _overall(sum(c["passed"] for c in allc), len(allc))
    report = ConformanceReport(
        endpoint_checks=[CheckResult(**c) for c in ep],
        degradation_checks=[CheckResult(**c) for c in dg],
        overall=overall,
        summary=(llm_summary[:500] if llm_summary else data.get("aborted") or _auto_summary(allc, "checks")),
    )
    print_conformance_report(report)
    return overall, report.model_dump_json(indent=2)


async def phase_ui(use_llm: bool) -> tuple[str, str]:
    tools.LAST_RESULTS.pop("ui_smoke", None)
    llm_summary, problem = "", ""
    if use_llm:
        llm_summary, problem = await run_agent(build_frontend_tester(), "Run the UI smoke test and summarise the result.")
        llm_summary = llm_summary or ""
        if problem:
            print(f"  [{problem} - using the recorded step results instead]")
    data = tools.LAST_RESULTS.get("ui_smoke")
    if data is None:
        data = await tools.run_ui_smoke()
    steps = data["steps"]
    auto = f"{data.get('passed', 0)}/{data.get('total', len(steps))} steps passed ({data.get('mode', '?')} UI at {data.get('url', '?')})."
    if data.get("first_failure"):
        auto += f" First failure - {data['first_failure']}"
    if data.get("hint"):
        auto += f" {data['hint']}"
    if data.get("note"):
        auto += f" Note: {data['note']}."
    report = UIAuditReport(
        steps=[UIStep(**s) for s in steps],
        overall=data["overall"],
        summary=(llm_summary[:500] if llm_summary else auto),
    )
    print_ui_audit(report)
    return report.overall, report.model_dump_json(indent=2)


async def phase_e2e(use_llm: bool) -> tuple[str, str]:
    """Same pattern as phases 1 and 2: the tool does the work, the model only summarises,
    and if the model fails (timeout, MaxTokensReachedException, ...) the recorded results are used."""
    tools.LAST_RESULTS.pop("e2e", None)
    llm_summary, problem = "", ""
    if use_llm:
        llm_summary, problem = await run_agent(build_e2e_tester(), "Run the end-to-end journey and summarise the result.")
        llm_summary = llm_summary or ""
        if problem:
            print(f"  [{problem} - using the recorded journey results instead]")
    data = tools.LAST_RESULTS.get("e2e")
    if data is None:  # model never called the tool, or no LLM
        data = await tools.run_e2e_journey()
    checks = data["checks"]
    auto = f"{data['passed']}/{data['total']} checks passed."
    if data.get("first_failure"):
        auto += f" First failure - {data['first_failure']}"
    report = E2EReport(
        checks=[CheckResult(**c) for c in checks],
        overall=data["overall"],
        summary=(llm_summary[:500] if llm_summary else auto),
    )
    print_e2e_report(report)
    return report.overall, report.model_dump_json(indent=2)


PHASES = {
    "1": ("API Conformance", lambda llm: phase_api(llm)),
    "2": ("Frontend QA", lambda llm: phase_ui(llm)),
    "3": ("End-to-End", lambda llm: phase_e2e(llm)),
}


# ── CLI ─────────────────────────────────────────────────────────────────────

def choose_phases(args: list[str]) -> list[str]:
    if args:
        raw = args
    else:
        print("\nWhich phases do you want to run?")
        print("  1 = API Conformance")
        print("  2 = Frontend QA")
        print("  3 = End-to-End           (API + real browser, same city)")
        print("  all = 1, 2 and 3")
        print("\nExamples: '1', '1,2', 'all', or press Enter for the quick pair (1,2)")
        raw = [input("> ").strip().lower()]
    tokens: list[str] = []
    for chunk in raw:
        if chunk == "":
            return ["1", "2"]
        if chunk == "all":
            return ["1", "2", "3"]
        tokens.extend(t.strip() for t in chunk.replace(",", " ").split())
    valid = [t for t in tokens if t in PHASES]
    if not valid:
        print(f"[no valid phases in {tokens!r} - running the quick pair 1,2]")
        return ["1", "2"]
    return valid


async def main() -> None:
    argv = sys.argv[1:]
    use_llm = "--no-llm" not in argv
    args = [a for a in argv if not a.startswith("--")]
    selected = choose_phases(args)
    run_start = time.monotonic()

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    print(f"\nRunning phases: {', '.join(PHASES[p][0] for p in selected)}")
    print(f"  API      {tools.api_base()}   (what phases 1 and 3 call directly)")
    print(f"  Frontend {tools.ui_url()}  (mode: {tools.ui_mode()})")
    print("           note: the deployed UI calls whatever backend it was built with (VITE_API),")
    print("                 which may differ from API_BASE_URL above.")
    if use_llm:
        print(f"  Model    {model_id()} via {_env('OLLAMA_HOST', 'http://localhost:11434')}")
    print(f"  Reports  {reports_dir.resolve()}")
    print("\nPre-flight checks:")
    problems, use_llm = await preflight(selected, use_llm)

    results: list[dict] = []
    try:
        for key in selected:
            name, runner = PHASES[key]
            print("\n" + "=" * 64)
            print(f"PHASE {key}: {name}")
            print("=" * 64)
            missing = [problems[n] for n in sorted(NEEDS[key]) if n in problems]
            if missing:
                reason = " | ".join(missing)
                print(f"SKIPPED - {reason}")
                text = f"SKIPPED: {reason}"
                results.append({"key": key, "name": name, "status": "skipped", "secs": 0.0, "text": text})
            else:
                t0 = time.monotonic()
                status, text = await runner(use_llm)
                secs = time.monotonic() - t0
                results.append({"key": key, "name": name, "status": status, "secs": secs, "text": text})

            # ── write this phase's output to its own file ──
            slug = name.lower().replace(" ", "-")
            phase_file = reports_dir / f"{stamp}-phase-{key}-{slug}.txt"
            phase_file.write_text(
                f"PHASE {key}: {name}\n"
                f"STATUS:  {results[-1]['status']}\n"
                f"TIME:    {results[-1]['secs']:.1f}s\n"
                f"{'=' * 64}\n\n"
                f"{text}\n",
                encoding="utf-8",
            )
            print(f"  [saved → {phase_file.name}]")
    finally:
        await tools.close_browser()

    # ── consolidated report: print + save ──
    print("\n" + "#" * 64)
    print("# CONSOLIDATED REPORT")
    print("#" * 64)
    print("\nSummary:")
    for r in results:
        print(f"  Phase {r['key']}  {r['name']:<18} {r['status'].upper():<8} {r['secs']:>6.0f}s")
    total = time.monotonic() - run_start
    print(f"  Total wall time: {total:.0f}s")

    for r in results:
        print(f"\n--- PHASE {r['key']}: {r['name'].upper()} ---")
        print(r["text"])

    lines = ["# CONSOLIDATED REPORT", f"run: {stamp}", f"total: {total:.0f}s", ""]
    for r in results:
        lines.append(f"--- PHASE {r['key']}: {r['name'].upper()} ---")
        lines.append(f"status: {r['status']}  time: {r['secs']:.0f}s")
        lines.append(r["text"])
        lines.append("")
    summary_file = reports_dir / f"{stamp}-summary.txt"
    summary_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[saved summary → {summary_file.name}]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[interrupted]")