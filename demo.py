"""Live demo: record a walkthrough of the deployed React app. No LLM."""
import asyncio
import time
import webbrowser
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from tools import record_walkthrough, ui_url, api_base


async def main():
    t0 = time.monotonic()
    print("=" * 64)
    print("  STRANDS AGENT — LIVE RECORDED WALKTHROUGH")
    print(f"  target:  {ui_url()}")
    print(f"  backend: {api_base()}")
    print("=" * 64)

    print("\n  → record_walkthrough(city='Kansas City, MO')")
    step_t0 = time.monotonic()
    result = await record_walkthrough(city1="Kansas City, MO", city2="Dallas, TX")
    print(f"  ← record_walkthrough done in {time.monotonic() - step_t0:.1f}s\n")

    for s in result.get("steps", []):
        mark = "PASS" if s.get("passed") else "FAIL"
        line = f"  [{mark}] {s['step']}"
        if not s.get("passed") and s.get("error"):
            line += f"  ({s['error']})"
        print(line)

    print(f"\n  {result.get('passed', '?')}/{result.get('total', '?')} steps passed")
    print(f"  video: {result.get('video')}")
    print(f"  trace: {result.get('trace')}")

    rec_dir = result.get("recording_dir")
    if rec_dir and Path(rec_dir).exists():
        webbrowser.open(str(Path(rec_dir).resolve()))

    print(f"\nTOTAL: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())