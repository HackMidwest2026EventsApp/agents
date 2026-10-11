"""Tools the agents call to exercise the app over HTTP and in a real browser.

Design notes (why this file looks the way it does)
--------------------------------------------------
* CPU inference is slow per token, so every tool returns SMALL, pre-digested
  JSON.  The model never has to read a 40 KB event list.
* Three coarse "suite" tools (`api_conformance_suite`, `ui_smoke_test`,
  `e2e_journey`) do all of the checking in code and hand back evidence.  The
  model only decides what to call and then writes a short summary - far fewer
  LLM round trips, and nothing for a small model to get lost in.
* Base URLs are read at CALL time (not import time) and normalised, so a
  trailing slash in .env can no longer produce `.../events-ui//`.
* Nothing here raises: every failure comes back as data with a hint.

The deployed UI (React + Vite on GitHub Pages) has NO login, NO "Set location"
button and NO "api online" pill.  What it does have:
    - a keyword box   (placeholder "Search events, music, food...")
    - a city box      (placeholder "Enter a city...")
    - a badge         ("20 OF 1104 EVENTS")
    - event cards     (title, ISO date like 2026-10-11T16:00:00Z, address)
The React flow below is written against exactly that.
"""

import asyncio
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from strands import tool

# ── Configuration (read from the environment at call time) ─────────────────


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return (value or default).strip()


def api_base() -> str:
    return _env("API_BASE_URL", "http://localhost:8000").rstrip("/")


def frontend_base() -> str:
    return _env("FRONTEND_BASE_URL", "http://localhost:3000").rstrip("/")


def ui_path() -> str:
    """'/' for the deployed app (the sub-path lives in FRONTEND_BASE_URL), '/ui/' for the backend's dev UI."""
    stripped = _env("UI_PATH", "/").strip("/")
    return f"/{stripped}/" if stripped else "/"


def ui_url() -> str:
    return frontend_base() + ui_path()


def ui_mode() -> str:
    """'react' (the deployed React app) or 'dev' (the backend's bundled /ui page)."""
    forced = _env("UI_MODE", "").lower()
    if forced in ("react", "dev"):
        return forced
    if forced == "next":  # old name; the Next.js app is gone, the React flow is the replacement
        return "react"
    return "dev" if ui_path().startswith("/ui") else "react"


def clean_url(url: str) -> str:
    """Join relative paths onto the frontend and collapse accidental '//' in the path."""
    url = url.strip()
    if url.startswith("/"):
        url = frontend_base() + url
    match = re.match(r"^(https?://[^/]+)(/[^?#]*)?(.*)$", url)
    if not match:
        return url
    host, path, rest = match.group(1), match.group(2) or "/", match.group(3)
    return host + re.sub(r"/{2,}", "/", path) + rest


_TIMEOUT = httpx.Timeout(float(_env("HTTP_TIMEOUT_S", "60")), connect=5.0)
_NAV_TIMEOUT_MS = int(float(_env("NAV_TIMEOUT_S", "60")) * 1000)

# Raw results of the suite tools, so the orchestrator can build a report even
# if the model never produces a usable summary.
LAST_RESULTS: dict[str, dict] = {}
_auth_token: str | None = None

# ── Small helpers ───────────────────────────────────────────────────────────


def _shrink(obj: Any, limit: int = 300) -> Any:
    text = obj if isinstance(obj, str) else repr(obj)
    return text if len(text) <= limit else text[:limit] + "…"


def _body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return response.text[:300]


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": str(detail)[:220]}


def _status(response: httpx.Response | None, exc: BaseException | None) -> str:
    if response is None:
        return f"no response ({exc!r})"[:160]
    return f"HTTP {response.status_code}"


def _overall(passed: int, total: int) -> str:
    if total == 0 or passed == 0:
        return "fail"
    return "pass" if passed == total else "partial"


async def _get(path: str, params: dict | None = None, *, retries: int = 1):
    url = f"{api_base()}{path}"
    headers = {}
    if _auth_token:
        headers["Authorization"] = f"Bearer {_auth_token}"
    last = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                return await client.get(url, params=params, headers=headers), None
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
            last = exc
            if attempt < retries:
                await asyncio.sleep(1.5)
        except Exception as exc:
            return None, exc
    return None, last


def _compact_events(body: Any, limit: int = 3) -> Any:
    if isinstance(body, dict) and isinstance(body.get("results"), list):
        results = body["results"]
        slim = [
            {
                "title": str(e.get("title", ""))[:60],
                "venue_name": e.get("venue_name"),
                "start_time": e.get("start_time"),
                "latitude": e.get("latitude"),
                "longitude": e.get("longitude"),
                "is_online": e.get("is_online"),
            }
            for e in results[:limit]
            if isinstance(e, dict)
        ]
        return {
            "total": body.get("total"),
            "returned": len(results),
            "has_more": body.get("has_more"),
            "first_results": slim,
        }
    return _shrink(body, 400)


def _compact_plan(body: Any) -> Any:
    if not (isinstance(body, dict) and isinstance(body.get("options"), list)):
        return _shrink(body, 400)
    uber = body.get("uber") or {}
    est = body.get("uber_estimate") or {}
    return {
        "verdict": body.get("verdict"),
        "recommended_mode": body.get("recommended_mode"),
        "destination_label": body.get("destination_label"),
        "distance_mi": body.get("distance_mi"),
        "weather_note": body.get("weather_note"),
        "leave_by": body.get("leave_by"),
        "options": [
            {
                "mode": o.get("mode"),
                "score": o.get("score"),
                "eta_min": o.get("eta_min"),
                "cost_display": o.get("cost_display"),
                "why": (o.get("why") or [])[:2],
            }
            for o in body["options"]
            if isinstance(o, dict)
        ],
        "uber": {
            "universal_url": _shrink(uber.get("universal_url"), 70),
            "app_url": _shrink(uber.get("app_url"), 50),
            "has_both_links": bool(uber.get("universal_url") and uber.get("app_url")),
        },
        "uber_estimate": {"source": est.get("source"), "fare_display": est.get("fare_display")} if est else None,
    }


# ── Granular API tools (kept for ad-hoc use) ────────────────────────────────


@tool
async def api_health() -> dict:
    """Check the backend is alive (GET /health, retried once for cold starts).

    Returns:
        {"alive": bool, "status": int} or {"alive": False, "error": "...", "hint": "..."}.
    """
    response, exc = await _get("/health", retries=1)
    if response is None:
        return {
            "alive": False,
            "error": repr(exc)[:160],
            "hint": f"Nothing answered at {api_base()}. Is uvicorn running and API_BASE_URL correct?",
        }
    return {"alive": response.status_code == 200, "status": response.status_code}


@tool
async def api_geocode(q: str) -> dict:
    """Resolve a place name or address to coordinates via the API.

    Args:
        q: Place name, e.g. "Kansas City, MO".
    """
    response, exc = await _get("/api/v1/events/geocode", {"q": q})
    if response is None:
        return {"error": repr(exc)[:200]}
    return {"status": response.status_code, "body": _shrink(_body(response), 300) if response.status_code != 200 else _body(response)}


@tool
async def api_search_events(
    latitude: float,
    longitude: float,
    radius_mi: int = 10,
    keyword: str = "",
    free_only: bool = False,
) -> dict:
    """Search events near a coordinate pair. Returns a short summary plus the first 3 events.

    Args:
        latitude: Center latitude.
        longitude: Center longitude.
        radius_mi: Search radius in miles.
        keyword: Optional free-text filter (empty string for none).
        free_only: Restrict to free events.
    """
    params: dict = {"latitude": latitude, "longitude": longitude, "radius_mi": radius_mi, "free_only": free_only}
    if keyword:
        params["q"] = keyword  # the backend's keyword parameter is `q`
    response, exc = await _get("/api/v1/events/search", params)
    if response is None:
        return {"error": repr(exc)[:200]}
    body = _body(response)
    return {"status": response.status_code, "body": _compact_events(body) if response.status_code == 200 else _shrink(body)}


@tool
async def api_travel_plan(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float = 0.0,
    dest_lon: float = 0.0,
    q: str = "",
) -> dict:
    """Score every travel mode between an origin and a destination.

    Supply either dest_lat/dest_lon (both non-zero) OR q (a place name).

    Args:
        origin_lat: Origin latitude.
        origin_lon: Origin longitude.
        dest_lat: Destination latitude (0 to use q instead).
        dest_lon: Destination longitude (0 to use q instead).
        q: Destination place name (empty to use dest coordinates).
    """
    params: dict = {"origin_lat": origin_lat, "origin_lon": origin_lon}
    if dest_lat and dest_lon:
        params["dest_lat"] = dest_lat
        params["dest_lon"] = dest_lon
    if q:
        params["q"] = q
    response, exc = await _get("/api/v1/travel/plan", params)
    if response is None:
        return {"error": repr(exc)[:200]}
    body = _body(response)
    return {"status": response.status_code, "body": _compact_plan(body) if response.status_code == 200 else _shrink(body)}


@tool
async def frontend_check_page(path: str = "") -> dict:
    """Fetch a frontend page over plain HTTP and check it returns 200.

    Args:
        path: URL path. Leave empty to use the configured UI path.
    """
    url = clean_url(path) if path else ui_url()
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            r = await client.get(url)
        return {"url": url, "status": r.status_code, "alive": r.status_code == 200, "snippet": r.text[:200]}
    except Exception as exc:  # noqa: BLE001
        return {"url": url, "alive": False, "error": repr(exc)[:200]}


# ── Browser plumbing (one shared Chromium for the whole run) ────────────────

_pw = None
_browser = None
_browser_lock = asyncio.Lock()


async def _get_browser():
    global _pw, _browser
    async with _browser_lock:
        if _browser is None or not _browser.is_connected():
            from playwright.async_api import async_playwright

            if _pw is None:
                _pw = await async_playwright().start()
            _browser = await _pw.chromium.launch(headless=True)
        return _browser


async def close_browser() -> None:
    """Called by the orchestrator when the run is over."""
    global _pw, _browser
    try:
        if _browser is not None:
            await _browser.close()
        if _pw is not None:
            await _pw.stop()
    except Exception:  # noqa: BLE001
        pass
    _browser = None
    _pw = None


async def _new_page(init_script: str | None = None):
    browser = await _get_browser()
    context = await browser.new_context(viewport={"width": 1280, "height": 900})

    async def _skip_heavy(route):
        # Images/fonts/media are irrelevant to a functional test and slow on a laptop.
        if route.request.resource_type in ("image", "media", "font"):
            await route.abort()
        else:
            await route.continue_()

    await context.route("**/*", _skip_heavy)
    if init_script:
        await context.add_init_script(init_script)
    page = await context.new_page()
    return context, page


def _browser_hint(exc: BaseException) -> str:
    text = str(exc)
    if "Executable doesn't exist" in text or "playwright install" in text:
        return "Chromium is not installed. Run:  python -m playwright install chromium"
    if isinstance(exc, ImportError):
        return "Playwright is not installed. Run:  pip install playwright  then  python -m playwright install chromium"
    return ""


def _err(exc: BaseException) -> str:
    first = (str(exc).splitlines() or [""])[0][:200]
    return f"{type(exc).__name__}: {first}" if first else type(exc).__name__


async def _text(page, limit: int = 300) -> str:
    try:
        raw = await page.inner_text("body")
    except Exception:  # noqa: BLE001
        return ""
    return re.sub(r"\s+", " ", raw).strip()[:limit]


async def _hints(page) -> dict:
    """What the model/user can actually click or type into - handy after a selector miss."""
    try:
        return await page.evaluate(
            """() => ({
              inputs: [...document.querySelectorAll('input,select,textarea')].slice(0, 12)
                .map(e => e.getAttribute('aria-label') || e.placeholder || e.id || e.name || e.type),
              buttons: [...document.querySelectorAll('button')].slice(0, 15)
                .map(e => (e.innerText || e.getAttribute('aria-label') || '').trim().slice(0, 30))
                .filter(Boolean)
            })"""
        )
    except Exception:  # noqa: BLE001
        return {}


def _http_error_hint(status: int, url: str) -> str:
    if status == 404 and ui_mode() == "react":
        return (
            f"{url} is a 404. For the GitHub Pages build the sub-path belongs in FRONTEND_BASE_URL, e.g. "
            "FRONTEND_BASE_URL=https://hackmidwest2026eventsapp.github.io/events-ui with UI_PATH=/ . "
            "'/ui/' only exists on the backend (port 8000)."
        )
    return f"{url} returned HTTP {status}. Is the frontend reachable?"


@tool
async def browser_flow(url: str, steps: list[dict]) -> dict:
    """Drive a real browser: load a page, optionally run actions, return a compact result.

    Args:
        url: Page to load first. A path like "/" is joined onto the frontend base URL.
        steps: List of action dicts (use [] to just load the page). Each has:
               - action: "click", "type", "wait" or "extract"
               - selector: Playwright selector, e.g. 'input[placeholder*="city" i]'
               - value: (for "type") the text to enter
    """
    target = clean_url(url)
    try:
        context, page = await _new_page()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "url": target, "error": _err(exc), "hint": _browser_hint(exc)}

    log: list[str] = []
    extracts: dict[str, str] = {}
    try:
        response = await page.goto(target, wait_until="domcontentloaded", timeout=_NAV_TIMEOUT_MS)
        if response is not None and response.status >= 400:
            return {"ok": False, "url": target, "http_status": response.status,
                    "error": f"HTTP {response.status} at {target}", "hint": _http_error_hint(response.status, target)}
        log.append(f"loaded {target}")
        if steps:
            try:
                await page.wait_for_load_state("networkidle", timeout=3000)
            except Exception:  # noqa: BLE001
                pass
        for i, step in enumerate(steps or []):
            action, sel, val = step.get("action", ""), step.get("selector", ""), step.get("value", "")
            if action == "click":
                await page.click(sel, timeout=8000)
                log.append(f"step {i}: clicked {sel}")
            elif action == "type":
                await page.fill(sel, val, timeout=8000)
                log.append(f"step {i}: typed '{val}' into {sel}")
            elif action == "wait":
                await page.wait_for_selector(sel, timeout=15000)
                log.append(f"step {i}: {sel} appeared")
            elif action == "extract":
                extracts[sel] = (await page.inner_text(sel))[:500]
                log.append(f"step {i}: extracted from {sel}")
            else:
                log.append(f"step {i}: unknown action '{action}'")
        return {"ok": True, "title": await page.title(), "log": log, "extracts": extracts,
                "final_text": await _text(page, 600)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "url": target, "log": log, "error": _err(exc), "available": await _hints(page)}
    finally:
        await context.close()


# ── Suite 1: API conformance, all checks in code ────────────────────────────

_KC = (39.0997, -94.5786)


async def _register_throwaway_token() -> tuple[str | None, str]:
    """Create a throwaway account through the API (used by the API suite for authenticated calls)."""
    email = f"qa-agent-{int(time.time())}@example.com"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.post(f"{api_base()}/api/v1/auth/register", json={"email": email, "password": "qa-agent-pw1"})
        if r.status_code in (200, 201) and isinstance(_body(r), dict) and _body(r).get("access_token"):
            return _body(r)["access_token"], email
        return None, f"register returned HTTP {r.status_code}"
    except Exception as exc:  # noqa: BLE001
        return None, _err(exc)


async def run_api_suite() -> dict:
    """Run every API contract check. Returns {"endpoint_checks": [...], "degradation_checks": [...]}."""
    global _auth_token
    endpoint: list[dict] = []
    degrade: list[dict] = []

    r, exc = await _get("/health", retries=1)
    if r is None or r.status_code != 200:
        endpoint.append(_check("GET /health", False, f"{_status(r, exc)} at {api_base()}"))
        return {"endpoint_checks": endpoint, "degradation_checks": [],
                "aborted": f"backend not reachable at {api_base()}"}
    endpoint.append(_check("GET /health", True, "HTTP 200"))

    # geocode
    r, exc = await _get("/api/v1/events/geocode", {"q": "Kansas City, MO"})
    body = _body(r) if r is not None else None
    ok = bool(r is not None and r.status_code == 200 and isinstance(body, dict)
              and _is_num(body.get("latitude")) and _is_num(body.get("longitude")))
    lat, lon = (body["latitude"], body["longitude"]) if ok else _KC
    detail = f"{_status(r, exc)} lat={lat} lon={lon}" + ("" if ok else " (using fallback coordinates)")
    endpoint.append(_check("GET /events/geocode", ok, detail))

    if _auth_token is None:
        token, _ = await _register_throwaway_token()
        if token:
            _auth_token = token

    async def search_check() -> dict:
        rr, ee = await _get("/api/v1/events/search", {"latitude": lat, "longitude": lon, "radius_mi": 10})
        b = _body(rr) if rr is not None else None
        results = b.get("results") if isinstance(b, dict) else b if isinstance(b, list) else None
        good = bool(rr is not None and rr.status_code == 200 and isinstance(results, list))
        extra = f", returned={len(results)}, total={b.get('total') if isinstance(b, dict) else '?'}" if good else ""
        return _check("GET /events/search", good, f"{_status(rr, ee)}{extra}")

    async def travel_check() -> dict:
        rr, ee = await _get("/api/v1/travel/plan", {"origin_lat": lat, "origin_lon": lon,
                                                    "dest_lat": lat + 0.02, "dest_lon": lon + 0.02})
        b = _body(rr) if rr is not None else None
        if rr is None or rr.status_code != 200 or not isinstance(b, dict):
            # include the response body: a bare "HTTP 409" tells you nothing about WHY
            why = f" - {_shrink(b, 140)}" if rr is not None else ""
            return _check("GET /travel/plan", False, f"{_status(rr, ee)}{why}")
        missing = [k for k in ("verdict", "recommended_mode") if not b.get(k)]
        options = b.get("options")
        if not isinstance(options, list) or not options:
            missing.append("options")
        else:
            if not all(isinstance(o, dict) and o.get("mode") and _is_num(o.get("score")) and o.get("why") is not None
                       for o in options):
                missing.append("options[].mode/score/why")
        uber = b.get("uber") if isinstance(b.get("uber"), dict) else {}
        missing += [f"uber.{k}" for k in ("universal_url", "app_url") if not uber.get(k)]
        if missing:
            return _check("GET /travel/plan", False, f"HTTP 200 but missing: {', '.join(missing)}")
        return _check("GET /travel/plan", True,
                      f"HTTP 200 mode={b.get('recommended_mode')}, {len(options)} options, both Uber links present")

    endpoint.extend(await asyncio.gather(search_check(), travel_check()))

    # degradation: bad input must be a clean 4xx, never a 5xx / crash
    async def no_destination() -> dict:
        rr, ee = await _get("/api/v1/travel/plan", {"origin_lat": lat, "origin_lon": lon})
        return _check("travel/plan without destination -> 400", bool(rr is not None and rr.status_code == 400), _status(rr, ee))

    async def bad_latitude() -> dict:
        rr, ee = await _get("/api/v1/events/search", {"latitude": 999, "longitude": lon})
        return _check("events/search with latitude=999 -> 4xx", bool(rr is not None and 400 <= rr.status_code < 500), _status(rr, ee))

    async def nonsense_place() -> dict:
        rr, ee = await _get("/api/v1/events/geocode", {"q": "zzzzqqqq1234xyz"})
        return _check("geocode of nonsense place -> no 5xx", bool(rr is not None and rr.status_code < 500), _status(rr, ee))

    degrade.extend(await asyncio.gather(no_destination(), bad_latitude(), nonsense_place()))
    return {"endpoint_checks": endpoint, "degradation_checks": degrade}


@tool
async def api_conformance_suite() -> dict:
    """Run the complete API contract test in one call (health, geocode, events search,
    travel plan, plus graceful-degradation checks) and return every check with evidence.

    Returns:
        {"endpoint_checks": [{name, passed, detail}], "degradation_checks": [...]}
    """
    result = await run_api_suite()
    LAST_RESULTS["api_suite"] = result
    return result


# ── Suite 2: scripted UI walk-through in ONE browser session ────────────────


class _Steps:
    """Runs steps in order; after the first failure the rest are marked skipped (no stacked timeouts)."""

    def __init__(self, page) -> None:
        self.page = page
        self.rows: list[dict] = []
        self.failed = False

    async def run(self, action: str, fn) -> None:
        n = len(self.rows) + 1
        if self.failed:
            self.rows.append({"step": n, "action": action, "passed": False, "page_text_snippet": "",
                              "error": "skipped: an earlier step failed"})
            return
        try:
            evidence = await fn()
            self.rows.append({"step": n, "action": action, "passed": True,
                              "page_text_snippet": str(evidence or "")[:200], "error": None})
        except Exception as exc:  # noqa: BLE001
            self.failed = True
            self.rows.append({"step": n, "action": action, "passed": False,
                              "page_text_snippet": await _text(self.page, 160), "error": _err(exc)})


# ---- helpers for the deployed React app --------------------------------------------------

_BADGE_RE = re.compile(r"(\d[\d,]*)\s+OF\s+(\d[\d,]*)\s+EVENTS", re.I)   # "20 OF 1104 EVENTS"
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")                  # "2026-10-11T16:00"
_CITY_INPUT = 'input[placeholder*="city" i]'                              # "Enter a city..."
_SEARCH_INPUT = 'input[placeholder*="search" i]'                          # "Search events, music, food..."
_NONSENSE = "zzzzqqqq1234xyz"


async def _badge(page) -> tuple[int, int] | None:
    """(shown, total) from the 'N OF M EVENTS' pill, or None if it is not on the page."""
    m = _BADGE_RE.search(await _text(page, 20000))
    if not m:
        return None
    return int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))


async def _wait_badge(page, ok, timeout_s: float, what: str):
    """Poll the badge until ok(badge) is true. Raises RuntimeError (with the last badge) on timeout."""
    deadline = time.monotonic() + timeout_s
    while True:
        last = await _badge(page)
        if ok(last):
            return last
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{what} (badge now: {last})")
        await asyncio.sleep(0.5)


async def _poll(fn, timeout_s: float, what: str):
    """Poll an async callable until it returns something truthy."""
    deadline = time.monotonic() + timeout_s
    while True:
        value = await fn()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise RuntimeError(what)
        await asyncio.sleep(0.5)


def _attach_net(page) -> dict:
    """Record the browser's calls to the backend: ok (<400), warn (4xx), bad (5xx / network failure / CORS)."""
    net: dict[str, list[str]] = {"ok": [], "warn": [], "bad": [], "console": []}
    ui_host = urlparse(ui_url()).netloc

    def label(url: str, method: str) -> str:
        u = urlparse(url)
        return f"{method} {u.netloc}{u.path}"

    def is_backend_call(resource_type: str, url: str) -> bool:
        host = urlparse(url).netloc
        return resource_type in ("fetch", "xhr") and host != ui_host and "google" not in host

    def on_response(resp) -> None:
        try:
            req = resp.request
            if not is_backend_call(req.resource_type, resp.url):
                return
            bucket = "bad" if resp.status >= 500 else "warn" if resp.status >= 400 else "ok"
            net[bucket].append(f"{resp.status} {label(resp.url, req.method)}")
        except Exception:  # noqa: BLE001
            pass

    def on_failed(req) -> None:
        try:
            if is_backend_call(req.resource_type, req.url):
                net["bad"].append(f"FAILED {label(req.url, req.method)} ({getattr(req, 'failure', '') or 'network error'})")
        except Exception:  # noqa: BLE001
            pass

    def on_console(msg) -> None:
        try:
            text = msg.text or ""
            if msg.type == "error" and ("cors" in text.lower() or "access-control" in text.lower()):
                net["console"].append(text[:200])
        except Exception:  # noqa: BLE001
            pass

    page.on("response", on_response)
    page.on("requestfailed", on_failed)
    page.on("console", on_console)
    return net


async def _flow_react(S: _Steps, city: str, keyword: str, net: dict, full: bool = True) -> dict:
    """Walk the deployed React app.  full=False is the short version used by the end-to-end journey.

    Returns the facts the steps learned (e.g. the badge after the city search).
    """
    page = S.page
    seen: dict = {}

    async def load():
        response = await page.goto(ui_url(), wait_until="domcontentloaded", timeout=_NAV_TIMEOUT_MS)
        if response is not None and response.status >= 400:
            raise RuntimeError(_http_error_hint(response.status, ui_url()))
        await page.wait_for_selector("text=/Discover Events/i", timeout=30000)
        return f"HTTP {response.status if response else '?'} | title: {await page.title()} | 'Discover Events' heading rendered"

    async def inputs():
        await page.wait_for_selector(_SEARCH_INPUT, timeout=10000)
        await page.wait_for_selector(_CITY_INPUT, timeout=10000)
        return "keyword box ('Search events...') and city box ('Enter a city...') are present"

    async def set_city():
        before = await _badge(page)
        box = page.locator(_CITY_INPUT).first
        last_exc: Exception | None = None
        for attempt in ("fill", "keys"):
            if attempt == "fill":
                await box.fill(city, timeout=10000)
            else:  # some apps only react to real key events
                await box.fill("", timeout=10000)
                await box.press_sequentially(city, delay=40)
            try:  # autocomplete dropdown, if the app has one; otherwise submit with Enter
                await page.locator('[role="option"]').first.click(timeout=1500)
            except Exception:  # noqa: BLE001
                await box.press("Enter")
            try:
                badge = await _wait_badge(page, lambda b: bool(b and b[1] > 0), 15 if attempt == "fill" else 45,
                                          f"no events appeared after entering '{city}'")
                seen["city_badge"] = badge
                return f"entered '{city}' ({attempt}); badge {before} -> {badge[0]} of {badge[1]}"
            except RuntimeError as exc:
                last_exc = exc
        raise last_exc  # type: ignore[misc]

    async def cards():
        text = await _text(page, 20000)
        dates = _DATE_RE.findall(text)
        shown, total = seen["city_badge"]
        if not dates:
            raise RuntimeError(f"badge says {shown} of {total} events but no event card with a date is visible")
        if shown > total:
            raise RuntimeError(f"badge is inconsistent: shows {shown} of {total}")
        title = ""
        try:
            title = (await page.locator("h3").first.inner_text(timeout=2000))[:60]
        except Exception:  # noqa: BLE001
            pass
        return f"{len(dates)} dated event cards visible (badge {shown} of {total}); first date {dates[0]}; first title '{title}'"

    async def backend():
        if net["console"]:
            raise RuntimeError("CORS error in the browser: " + net["console"][0]
                               + " -> add the frontend origin to the backend's CORS list")
        if net["bad"]:
            raise RuntimeError("backend call failed: " + "; ".join(net["bad"][:2]))
        if not net["ok"]:
            raise RuntimeError("the page made no successful cross-origin API calls (is VITE_API pointing at a live backend?)")
        warn = f"; {len(net['warn'])} returned 4xx" if net["warn"] else ""
        return f"{len(net['ok'])} backend call(s) OK, e.g. {net['ok'][0]}{warn}"

    async def keyword_search():
        before = await _badge(page)
        box = page.locator(_SEARCH_INPUT).first
        await box.fill(keyword, timeout=10000)
        await box.press("Enter")
        await _wait_badge(page, lambda b: b is not None and b != before, 30,
                          f"results did not change after searching '{keyword}'")
        await asyncio.sleep(1.5)  # let a debounced / in-flight request settle
        after = await _badge(page)
        return f"'{keyword}': badge {before} -> {after}"

    async def empty_state():
        box = page.locator(_SEARCH_INPUT).first
        await box.fill(_NONSENSE, timeout=10000)
        await box.press("Enter")

        async def empty():
            b = await _badge(page)
            if b is not None and b[1] == 0:
                return f"badge {b[0]} of {b[1]}"
            if re.search(r"no events", await _text(page, 20000), re.I):
                return "'No events' message shown"
            return ""

        return "nonsense keyword -> " + await _poll(empty, 30, "nonsense keyword still shows events (no empty state)")

    async def clear_search():
        box = page.locator(_SEARCH_INPUT).first
        await box.fill("", timeout=10000)
        await box.press("Enter")
        b = await _wait_badge(page, lambda b: bool(b and b[1] > 0), 30, "results did not come back after clearing the keyword")
        return f"results restored: {b[0]} of {b[1]}"

    async def open_event():
        before = page.url
        target = page.locator("h3").first
        if not await target.count():
            target = page.locator(r"text=/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/").first
        await target.scroll_into_view_if_needed(timeout=5000)
        await target.click(timeout=8000)

        async def changed():
            if page.url != before:
                return "url change"
            if await page.locator('[role="dialog"]').count():
                return "dialog"
            if not re.search(r"upcoming events", await _text(page, 3000), re.I):
                return "view change"
            return ""

        how = await _poll(changed, 20, "clicking an event card opened nothing (no new URL, dialog or view)")
        await asyncio.sleep(1.0)
        heading = ""
        try:
            heading = (await page.locator("h1, h2").first.inner_text(timeout=2000))[:60]
        except Exception:  # noqa: BLE001
            pass
        text = await _text(page, 6000)
        extras = [w for w in ("What to wear", "Nearby", "Weather") if re.search(w, text, re.I)]
        return f"opened via {how}: {_shrink(page.url, 70)} | heading '{heading}' | sections: {', '.join(extras) or 'none seen'}"

    await S.run(f"load {ui_url()}", load)
    await S.run("keyword and city boxes are present", inputs)
    await S.run(f"enter city '{city}'", set_city)
    await S.run("event cards render (title, date, address)", cards)
    await S.run("browser reaches the backend (no CORS / 5xx)", backend)
    if full:
        await S.run(f"keyword search '{keyword}' changes the results", keyword_search)
        await S.run("nonsense keyword shows the empty state", empty_state)
        await S.run("clearing the keyword restores results", clear_search)
    await S.run("click an event card -> detail opens", open_event)
    return seen


async def _flow_dev(S: _Steps, city: str, dest: str) -> None:
    """The backend's bundled /ui page (only used when UI_MODE=dev or UI_PATH starts with /ui)."""
    page = S.page

    async def load():
        response = await page.goto(ui_url(), wait_until="domcontentloaded", timeout=_NAV_TIMEOUT_MS)
        if response is not None and response.status >= 400:
            raise RuntimeError(_http_error_hint(response.status, ui_url()))
        return f"HTTP {response.status if response else '?'} | title: {await page.title()}"

    async def lookup():
        await page.fill("#place", city, timeout=10000)
        await page.click('button:has-text("Look up")', timeout=10000)
        await page.wait_for_function("() => !/no location set/i.test(document.querySelector('#originState')?.innerText || '')", timeout=30000)
        return _shrink(await page.inner_text("#originState"), 120)

    async def search():
        await page.click("#searchBtn", timeout=10000)
        await page.wait_for_function("() => !/set a location, then search/i.test(document.querySelector('#searchState')?.innerText || '')", timeout=60000)
        return _shrink(await page.inner_text("#searchState"), 120)

    async def travel():
        await page.fill("#dest", dest, timeout=10000)
        await page.click('button:has-text("How should I get there?")', timeout=10000)
        await page.wait_for_function("() => /uber/i.test(document.querySelector('#travel')?.innerText || '')", timeout=60000)
        return _shrink(await page.inner_text("#travel"), 160)

    await S.run(f"load {ui_url()}", load)
    await S.run(f"look up '{city}'", lookup)
    await S.run("click Search", search)
    await S.run(f"how to get to '{dest}'", travel)


async def run_ui_smoke() -> dict:
    """Scripted UI walk-through. Returns steps shaped like UIStep + an overall verdict."""
    city = _env("UI_CITY", "Kansas City, MO")
    keyword = _env("UI_KEYWORD", "music")
    dest = "Union Station, Kansas City"
    mode = ui_mode()

    try:
        context, page = await _new_page()
    except Exception as exc:  # noqa: BLE001
        return {"overall": "fail", "mode": mode, "url": ui_url(),
                "steps": [{"step": 1, "action": "start browser", "passed": False, "page_text_snippet": "",
                           "error": _err(exc)}],
                "passed": 0, "total": 1, "first_failure": f"step 1: {_err(exc)}",
                "hint": _browser_hint(exc)}

    net = _attach_net(page)
    S = _Steps(page)
    try:
        if mode == "dev":
            await _flow_dev(S, city, dest)
        else:
            await _flow_react(S, city, keyword, net, full=True)
    finally:
        await context.close()

    rows = S.rows
    passed = sum(r["passed"] for r in rows)
    first_failed = next((r for r in rows if not r["passed"]), None)
    if passed == len(rows):
        overall = "pass"
    elif rows and not rows[0]["passed"]:
        overall = "fail"
    else:
        overall = "partial"
    return {"overall": overall, "mode": mode, "url": ui_url(), "note": "",
            "passed": passed, "total": len(rows),
            "first_failure": (f"step {first_failed['step']}: {first_failed['error']}" if first_failed else None),
            "steps": rows}


@tool
async def ui_smoke_test() -> dict:
    """Run the full scripted UI walk-through in a real browser in ONE call: load the page,
    check both search boxes exist, enter a city, check event cards render, check the browser can
    reach the backend, run a keyword search, check the empty state, clear the search and open an
    event. Returns each step with pass/fail and page evidence.

    Returns:
        {"overall": "pass"|"partial"|"fail", "steps": [{step, action, passed, page_text_snippet, error}]}
    """
    result = await run_ui_smoke()
    LAST_RESULTS["ui_smoke"] = result
    return result


# ── Suite 3: end-to-end (API + browser) in ONE call ─────────────────────────


async def run_e2e_journey() -> dict:
    """Geocode -> search -> travel plan over the API, then the same city through the real UI.

    Returns {"overall", "passed", "total", "checks": [{name, passed, detail}], ...}.
    """
    city = _env("UI_CITY", "Kansas City, MO")
    dest = "Union Station, Kansas City"
    checks: list[dict] = []

    # 1. geocode
    r, exc = await _get("/api/v1/events/geocode", {"q": city})
    body = _body(r) if r is not None else None
    ok = bool(r is not None and r.status_code == 200 and isinstance(body, dict)
              and _is_num(body.get("latitude")) and _is_num(body.get("longitude")))
    lat, lon = (body["latitude"], body["longitude"]) if ok else _KC
    checks.append(_check(f"API: geocode '{city}'", ok,
                         f"{_status(r, exc)} lat={lat} lon={lon}" + ("" if ok else " (fallback coordinates used)")))

    # 2. search near that point
    r, exc = await _get("/api/v1/events/search", {"latitude": lat, "longitude": lon, "radius_mi": 10})
    b = _body(r) if r is not None else None
    results = b.get("results") if isinstance(b, dict) else None
    good = bool(r is not None and r.status_code == 200 and isinstance(results, list) and results)
    api_total = b.get("total") if good and isinstance(b, dict) else None
    checks.append(_check("API: events search near the geocoded point", good,
                         f"{_status(r, exc)}, returned={len(results) if good else 0}, total={api_total}"))

    # 3. travel plan to a named destination
    r, exc = await _get("/api/v1/travel/plan", {"origin_lat": lat, "origin_lon": lon, "q": dest})
    b = _body(r) if r is not None else None
    ok = bool(r is not None and r.status_code == 200 and isinstance(b, dict)
              and b.get("recommended_mode") and isinstance(b.get("options"), list) and b["options"])
    if ok:
        detail = f"HTTP 200 mode={b.get('recommended_mode')}, {len(b['options'])} options"
    else:
        detail = _status(r, exc) + (f" - {_shrink(b, 120)}" if r is not None else "")
    checks.append(_check(f"API: travel plan to '{dest}'", ok, detail))

    # 4. the same city through the real browser
    seen: dict = {}
    try:
        context, page = await _new_page()
    except Exception as exc2:  # noqa: BLE001
        checks.append(_check("UI: start browser", False, f"{_err(exc2)} {_browser_hint(exc2)}"))
    else:
        net = _attach_net(page)
        S = _Steps(page)
        try:
            seen = await _flow_react(S, city, "", net, full=False)
        finally:
            await context.close()
        for row in S.rows:
            checks.append(_check("UI: " + row["action"], row["passed"],
                                 row["page_text_snippet"] if row["passed"] else row["error"]))

    # 5. cross-check
    badge = seen.get("city_badge")
    ui_txt = f"{badge[0]} of {badge[1]}" if badge else "none"
    checks.append(_check("API and UI both list events for the same city",
                         bool(api_total) and bool(badge and badge[1] > 0),
                         f"API total={api_total} (10 mi); UI badge={ui_txt}"))

    passed = sum(c["passed"] for c in checks)
    first_failed = next((c for c in checks if not c["passed"]), None)
    return {"overall": _overall(passed, len(checks)), "passed": passed, "total": len(checks), "checks": checks,
            "first_failure": (f"{first_failed['name']}: {first_failed['detail']}" if first_failed else None),
            "api_total": api_total, "ui_badge": list(badge) if badge else None}


@tool
async def e2e_journey() -> dict:
    """Run the whole end-to-end journey in ONE call: geocode a city, search events, build a travel
    plan over the API, then load the real web UI, enter the same city and open an event.
    Returns every check with pass/fail and evidence.

    Returns:
        {"overall": "pass"|"partial"|"fail", "passed": int, "total": int, "checks": [{name, passed, detail}]}
    """
    result = await run_e2e_journey()
    LAST_RESULTS["e2e"] = result
    return result


# ── Recorded walkthrough for the demo (no LLM) ──────────────────────────────


@tool
async def record_walkthrough(city1: str = "Kansas City, MO", city2: str = "Dallas, TX") -> dict:
    """Record a walkthrough: two city searches, scrolls, open one event page."""
    import random
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(_env("RECORDINGS_DIR", "recordings")) / f"demo-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    steps: list[dict] = []
    browser = await _get_browser()
    context = await browser.new_context(
        viewport={"width": 1280, "height": 800},
        record_video_dir=str(run_dir),
        record_video_size={"width": 1280, "height": 800},
    )
    await context.tracing.start(screenshots=True, snapshots=True, sources=True)
    page = await context.new_page()

    async def step(label, fn):
        try:
            detail = await fn()
            steps.append({"step": label, "passed": True, "detail": str(detail or "")[:160]})
        except Exception as exc:
            steps.append({"step": label, "passed": False, "error": _err(exc)})

    async def smooth_scroll(dy: int, duration_ms: int = 700):
        """Animated scroll — looks like a human, not a jump."""
        await page.evaluate(
            """([dy, dur]) => new Promise(res => {
                const start = window.scrollY;
                const t0 = performance.now();
                function tick(now) {
                    const k = Math.min(1, (now - t0) / dur);
                    const ease = k < 0.5 ? 2*k*k : 1 - Math.pow(-2*k + 2, 2)/2;
                    window.scrollTo(0, start + dy * ease);
                    if (k < 1) requestAnimationFrame(tick); else res();
                }
                requestAnimationFrame(tick);
            })""",
            [dy, duration_ms],
        )

    async def find(selectors, action, value: str = "", timeout: int = 3000) -> str:
        last = "no match"
        for sel in selectors:
            try:
                if action == "fill":
                    await page.fill(sel, value, timeout=timeout)
                elif action == "click":
                    await page.click(sel, timeout=timeout)
                elif action == "wait":
                    await page.wait_for_selector(sel, timeout=timeout)
                return sel
            except Exception as e:
                last = f"{sel}: {type(e).__name__}"
        raise RuntimeError(last)

    CITY_SELECTORS = (
        'input[placeholder*="city" i]',
        'input[placeholder*="location" i]',
        'input[placeholder*="address" i]',
        'input[placeholder*="enter a city" i]',
        'input[type="text"]',
    )
    SEARCH_SELECTORS = (
        'button:has-text("Find events")',
        'button:has-text("Search")',
        'button:has-text("Find")',
        'button[type="submit"]',
    )

    async def settle_results(timeout: int = 8000) -> int:
        try:
            await page.wait_for_function(
                "() => document.querySelectorAll('h3').length > 0 "
                "|| /no events matched/i.test(document.body.innerText)",
                timeout=timeout)
        except Exception:
            pass
        return await page.locator("h3").count()

    # ── 1. load, then breathe for a moment so the video starts settled ──
    async def _load():
        r = await page.goto(ui_url(), wait_until="networkidle", timeout=_NAV_TIMEOUT_MS)
        await page.wait_for_timeout(1200)   # let the hero settle before moving
        return f"HTTP {r.status if r else '?'}"
    await step("load home page", _load)

    # ── 2. Kansas City ────────────────────────────────────────────────
    async def _type_1():
        sel = await find(CITY_SELECTORS, "fill", value=city1)
        await page.wait_for_timeout(600)
        return f"typed '{city1}'"
    await step(f"type '{city1}'", _type_1)

    # ── 3. search ─────────────────────────────────────────────────────
    async def _search_1():
        try:
            await page.click('button:has-text("Set location")', timeout=1200)
        except Exception:
            pass
        sel = await find(SEARCH_SELECTORS, "click", timeout=3000)
        return f"clicked {sel}"
    await step("search Kansas City", _search_1)

    # ── 4. results ────────────────────────────────────────────────────
    async def _results_1():
        n = await settle_results()
        await page.wait_for_timeout(400)
        return f"{n} event cards"
    await step("Kansas City results", _results_1)

    # ── 5. smooth scroll down ─────────────────────────────────────────
    async def _scroll_down_1():
        await smooth_scroll(700, 900)
        await smooth_scroll(500, 700)
        return "scrolled down"
    await step("scroll down results", _scroll_down_1)

    # ── 6. smooth scroll back up ──────────────────────────────────────
    async def _scroll_up_1():
        await smooth_scroll(-1300, 1100)
        return "back at top"
    await step("scroll back up", _scroll_up_1)

    # ── 7. Dallas ─────────────────────────────────────────────────────
    async def _type_2():
        sel = await find(CITY_SELECTORS, "wait", timeout=2500)
        await page.fill(sel, "")
        await page.fill(sel, city2, timeout=3000)
        await page.wait_for_timeout(600)
        return f"typed '{city2}'"
    await step(f"type '{city2}'", _type_2)

    # ── 8. search again ───────────────────────────────────────────────
    async def _search_2():
        try:
            await page.click('button:has-text("Set location")', timeout=1200)
        except Exception:
            pass
        sel = await find(SEARCH_SELECTORS, "click", timeout=3000)
        return f"clicked {sel}"
    await step("search Dallas", _search_2)

    # ── 9. Dallas results ─────────────────────────────────────────────
    async def _results_2():
        n = await settle_results()
        await page.wait_for_timeout(400)
        return f"{n} event cards"
    await step("Dallas results", _results_2)

    # ── 10. small scroll ──────────────────────────────────────────────
    async def _scroll_down_2():
        await smooth_scroll(600, 800)
        return "scrolled"
    await step("scroll results", _scroll_down_2)

    # ── 11. click a random event card ─────────────────────────────────
    async def _click_random():
        links = page.locator("h3, article a, a[href*='/event']")
        count = await links.count()
        if count == 0:
            raise RuntimeError("no event cards")
        idx = random.randrange(count)
        target = links.nth(idx)
        await target.scroll_into_view_if_needed(timeout=2000)
        await page.wait_for_timeout(500)
        await target.click(timeout=3000)
        await page.wait_for_timeout(1500)     # SPA route change — give it real time
        return f"clicked card #{idx} of {count}"
    await step("click random event", _click_random)

    # ── 12. WAIT for every key section to actually render ────────────
    async def _event_page():
        # navigation finished
        try:
            await page.wait_for_load_state("networkidle", timeout=30000)
        except Exception:
            pass

        # the strings we're waiting for (lowercase substring match on body text)
        REQUIRED = ["what to wear", "explore nearby"]
        OPTIONAL = ["nearby restaurants", "°f", "mph"]   # nice-to-have signals

        deadline = time.monotonic() + 90        # up to 90 seconds
        seen_required: set[str] = set()
        seen_optional: set[str] = set()
        last_log = 0.0

        while time.monotonic() < deadline:
            try:
                body = (await page.inner_text("body")).lower()
            except Exception:
                body = ""

            for s in REQUIRED:
                if s in body:
                    seen_required.add(s)
            for s in OPTIONAL:
                if s in body:
                    seen_optional.add(s)

            # log progress every 3s so you can see what's still loading
            now = time.monotonic()
            if now - last_log > 3:
                last_log = now
                missing = [s for s in REQUIRED if s not in seen_required]
                print(f"    … waiting for event page: {len(seen_required)}/{len(REQUIRED)} sections "
                      f"({', '.join(missing) or 'all present'})")
                await page.wait_for_timeout(0)

            # done when all required sections are visible
            if len(seen_required) == len(REQUIRED):
                # give the page one extra beat to finish animating images in
                await page.wait_for_timeout(1200)
                break

            await page.wait_for_timeout(800)

        title = await page.title()
        h1 = ""
        try:
            h1 = (await page.locator("h1").first.inner_text(timeout=2000))[:60]
        except Exception:
            pass
        return (f"required {len(seen_required)}/{len(REQUIRED)}, "
                f"optional {len(seen_optional)}/{len(OPTIONAL)}, "
                f"h1='{h1}'")
    await step("event page fully loads", _event_page)

    # ── 13. slow scroll through the event page so judges see the content ──
    async def _tour_event():
        await smooth_scroll(500, 1000)   # past the hero
        await smooth_scroll(600, 1000)   # map / weather
        await smooth_scroll(700, 1100)   # what-to-wear cards
        await smooth_scroll(600, 1000)   # restaurants
        return "toured event page"
    await step("scroll event page", _tour_event)

    # ── 14. hold on the final frame ───────────────────────────────────
    await page.wait_for_timeout(1500)

    try:
        await context.tracing.stop(path=str(run_dir / "trace.zip"))
    except Exception:
        pass
    await context.close()

    videos = list(run_dir.glob("*.webm"))
    passed = sum(1 for s in steps if s["passed"])
    return {
        "passed": passed,
        "total": len(steps),
        "failed_steps": [s["step"] for s in steps if not s["passed"]],
        "steps": steps,
        "recording_dir": str(run_dir),
        "video": str(videos[0]) if videos else None,
        "trace": str(run_dir / "trace.zip") if (run_dir / "trace.zip").exists() else None,
    }