# Strands Agent

Built for **Hack Midwest 2026** under the judging criterion 

The agent and the app share no code and no virtual environment. The agent knows the app only through HTTP responses and the rendered DOM.

## What it does

| Phase | Agent | What it checks |
|---|---|---|
| 1 | API Conformance | Health, geocode, events search, and travel plan responses against the documented contract, plus graceful handling of bad input |
| 2 | Frontend QA | Drives the deployed React app: load, city entry, event cards, backend reachability (no CORS or 5xx), keyword search, empty state, clearing the search, opening an event |
| 3 | End-to-End | Geocodes a city, searches events, and builds a travel plan over the API, then loads the same city in the real UI and cross-checks that both sources list events |
| Demo | None (no LLM) | Records a walkthrough of the UI as a `.webm` video plus a replayable Playwright trace |

Each phase is one Strands agent with exactly one tool. Phases 1 to 3 write a timestamped report to `reports/`.

## Quick start

Requires Python 3.11+ (tested on 3.13), Git, and [Ollama](https://ollama.com). Works on Windows, macOS, and Linux. Commands below use PowerShell.

```powershell
git clone https://github.com/<your-org>/agents.git
cd agents
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium  # one-time browser download
ollama pull qwen3:4b
copy .env.example .env                 # macOS/Linux: cp .env.example .env
```

Edit `.env` (see [Configuration](#configuration)), make sure Ollama is running (`ollama serve`), then:

```powershell
python orchestrator.py                  # interactive menu (Enter = phases 1 and 2)
python orchestrator.py all              # all three phases
python orchestrator.py 1 3              # chosen phases
python orchestrator.py all --no-llm     # same checks, no language model, takes seconds
python demo.py                          # recorded demo, no LLM
```

> **macOS/Linux note:** `requirements.txt` was frozen on Windows and includes `pywin32`, which will not install elsewhere. Delete that line before running `pip install`.

## Configuration

Copy `.env.example` to `.env` and set these. URLs must not have a trailing slash, and there must be no spaces around `=`.

## How the Strands SDK is used

- **`Agent` + `OllamaModel`**: three agents built by a shared constructor in `agents.py`, all pointed at a local model with temperature 0.
- **`@tool`**: every capability the agents call is a Strands tool in `tools.py`.
- **Hooks**: `BeforeToolCallEvent` prints each call and enforces the tool-call budget by cancelling calls past `MAX_TOOL_CALLS`. `AfterToolCallEvent` prints how long each tool took.
- **Async invocation**: the orchestrator runs agents with `invoke_async` under a timeout and reports token usage and model calls per phase.


## Design

**Coarse tools, small outputs.** On CPU a 4B model generates only a few tokens per second, and every extra tool call is another round trip. So each phase uses one "suite" tool (`api_conformance_suite`, `ui_smoke_test`, `e2e_journey`) that does all the checking in Python and returns compact JSON evidence. The model calls the tool once and writes a two to three sentence summary. It never reads a 40 KB event list.

**The model summarises; the code is the source of truth.** Suite tools store their raw results in `LAST_RESULTS`. The orchestrator builds the Pydantic report (`ConformanceReport`, `UIAuditReport`, `E2EReport`) from that evidence, so a report is still written if the model times out, hits `MaxTokensReachedException`, or never calls the tool.

**Fail fast.** Before any model work, the orchestrator probes the backend, the frontend, and Ollama. A phase whose dependency is down is marked `SKIPPED` with a fix hint. If Ollama is down, the run continues without the LLM.

**Failures are kept as evidence.** A `PARTIAL` result with a specific error (for example a 409 from the travel plan endpoint, including the response body) shows the agent reporting what actually happened rather than a green check it may have invented. After the first failed UI step, remaining steps are marked skipped instead of stacking timeouts.

**The demo is deterministic.** `record_walkthrough` is plain Playwright with no LLM, so it behaves the same on stage every time. It is still a Strands `@tool`, the same kind the agents call.

## Checks in detail

**Phase 1: API conformance.** `GET /health`, `GET /events/geocode`, `GET /events/search`, and `GET /travel/plan` (verdict, recommended mode, and every option with mode, score, and reasons, plus both Uber links). Degradation checks confirm that a travel plan with no destination returns 400, a search with latitude 999 returns a 4xx, and geocoding a nonsense place never returns a 5xx. The suite registers a throwaway account through the API for authenticated calls.

**Phase 2: Frontend QA.** One browser session walks the deployed app: the "Discover Events" page loads, the keyword and city boxes exist, entering a city produces the "N OF M EVENTS" badge, event cards with dates render, the browser's calls to the backend succeed (and none are blocked by CORS), a keyword search changes the results, a nonsense keyword shows the empty state, clearing the keyword restores results, and clicking a card opens the event.

**Phase 3: End-to-end.** The three API calls, the UI flow without the keyword steps, and a final check that the API and the UI both list events for the same city. Note that the deployed UI calls whichever backend it was built with, which may differ from `API_BASE_URL`.

Overall status per phase is `pass` (all checks), `partial` (some), or `fail` (none).

## Output

```
reports/
├── <timestamp>-phase-1-api-conformance.txt
├── <timestamp>-phase-2-frontend-qa.txt
├── <timestamp>-phase-3-end-to-end.txt
└── <timestamp>-summary.txt

recordings/
└── demo-<timestamp>/
    ├── <hash>.webm   
    └── trace.zip        


The demo loads the app, searches two cities (Kansas City, then Dallas), scrolls the results, opens a random event, and waits up to 90 seconds for the "What to wear" and "Explore nearby" sections to render before touring the page. `demo.py` prints PASS or FAIL per step and opens the recording folder when it finishes.

## Project layout

```
agents.py         Agent constructors, Ollama helpers, Pydantic report schemas
orchestrator.py   CLI phase runner: pre-flight, runs agents, writes reports/
tools.py          @tool functions: HTTP checks and Playwright flows
demo.py           Recorded walkthrough, no LLM
smoke_test.py     Minimal Strands + Ollama connectivity check
requirements.txt  Pinned dependencies
.env.example      Config template


To verify that Strands can reach Ollama before running anything else, run `python smoke_test.py`. It asks the model to use an `add` tool to compute 2 + 3 (it uses `qwen3:4b`).

