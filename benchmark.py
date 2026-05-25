"""
benchmark.py — Groq vs Gemini latency benchmark for Sam voice bot.

Measures TTFT, TTFR, tokens/sec, and streaming chunk consistency
using Sam's actual system prompt and real client meeting scenarios.

Requirements:
    pip install openai python-dotenv
    Add GEMINI_API_KEY to your .env file

Usage:
    python benchmark.py
"""

import asyncio
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

# ── Import Sam's actual system prompt ─────────────────────────────────────────
try:
    from Agent import PM_PROMPT_V2_BASE as SYSTEM_PROMPT

    print("✓ Loaded system prompt from Agent.py")
except Exception as e:
    print(f"⚠  Could not import Agent.py system prompt ({e})")
    print("   Using minimal fallback prompt — results will differ from production.")
    SYSTEM_PROMPT = (
        "You are Sam, a senior PM at AnavClouds Software Solutions on a live voice call. "
        "Be concise, natural, and conversational. 1-2 sentences max."
    )

from openai import AsyncOpenAI  # noqa: E402 — after dotenv load

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.1-8b-instant"  # matches Agent.py line 1919
GEMINI_MODEL = "gemini-2.5-flash"
RUNS = 3  # runs per test case per provider
DELAY = 2.0  # seconds between API calls

TTFT_IDEAL = 400  # ms — ideal for voice UX
TTFT_MAX = 800  # ms — outer limit for acceptable voice UX
CHUNK_GAP_MAX = 500  # ms — max gap between streaming chunks before TTS stutters

# ── Test cases ────────────────────────────────────────────────────────────────
TEST_CASES = [
    {
        "name": "TEST 1 — SHORT (greeting)",
        "label": "SHORT",
        "prompt": "Hey Sam, how are you doing today?",
        "max_tokens": 40,
        "stream_detail": False,
    },
    {
        "name": "TEST 2 — MEDIUM (standup update)",
        "label": "MEDIUM",
        "prompt": (
            "So I've been working on the payment gateway refactor, fixed 3 bugs, "
            "had a client call about scope changes, and reviewed two PRs."
        ),
        "max_tokens": 120,
        "stream_detail": True,  # chunk timeline printed for this case only
    },
    {
        "name": "TEST 3 — LONG (complex concern)",
        "label": "LONG",
        "prompt": (
            "We're concerned about the timeline. The auth module is 2 weeks behind, "
            "the payment flow has 3 unresolved bugs, and the client demo is in 10 days. "
            "What should we prioritize and how do we communicate this to stakeholders?"
        ),
        "max_tokens": 200,
        "stream_detail": False,
    },
]


# ── API client factories ───────────────────────────────────────────────────────
def make_groq_client() -> AsyncOpenAI:
    keys_raw = os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", ""))
    keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
    if not keys:
        raise RuntimeError("GROQ_API_KEYS not found in .env")
    return AsyncOpenAI(
        api_key=keys[0],
        base_url="https://api.groq.com/openai/v1",
    )


def make_gemini_client() -> AsyncOpenAI:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not found in .env  —  add it to run Gemini tests"
        )
    # Gemini exposes an OpenAI-compatible endpoint — same client, different base URL
    return AsyncOpenAI(
        api_key=key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


# ── Single streaming run ───────────────────────────────────────────────────────
async def run_single(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    capture_chunks: bool = False,
) -> dict:
    """
    Stream one completion and return timing + text.
    Returns dict with: ttft_ms, ttfr_ms, word_count, text, chunk_timeline.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    t0 = time.perf_counter()
    ttft_ms = None
    full_text = ""
    chunk_timeline = []  # [(elapsed_ms, token_str)]

    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
        stream=True,
    )

    async for chunk in stream:
        token = (
            chunk.choices[0].delta.content
            if chunk.choices and chunk.choices[0].delta
            else None
        )
        if not token:
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if ttft_ms is None:
            ttft_ms = elapsed_ms

        full_text += token

        if capture_chunks:
            chunk_timeline.append((elapsed_ms, token))

    ttfr_ms = (time.perf_counter() - t0) * 1000

    return {
        "ttft_ms": ttft_ms or 0.0,
        "ttfr_ms": ttfr_ms,
        "word_count": len(full_text.split()),
        "text": full_text.strip(),
        "chunk_timeline": chunk_timeline,
    }


# ── Run N times for one provider + case, return aggregated stats ──────────────
async def run_case(
    client: AsyncOpenAI,
    model: str,
    provider_label: str,
    case: dict,
) -> dict | None:
    results = []
    capture_first = case["stream_detail"]

    for i in range(RUNS):
        run_num = i + 1
        print(
            f"    [{case['label']}] {provider_label} run {run_num}/{RUNS}...",
            end=" ",
            flush=True,
        )
        try:
            r = await run_single(
                client,
                model,
                case["prompt"],
                case["max_tokens"],
                capture_chunks=(capture_first and i == 0),
            )
            results.append(r)
            print(
                f"TTFT={r['ttft_ms']:.0f}ms  TTFR={r['ttfr_ms']:.0f}ms  words={r['word_count']}"
            )
        except Exception as e:
            print(f"ERROR: {e}")

        if i < RUNS - 1:
            await asyncio.sleep(DELAY)

    if not results:
        return None

    ttfts = [r["ttft_ms"] for r in results]
    ttfrs = [r["ttfr_ms"] for r in results]
    words = [r["word_count"] for r in results]

    # Tokens per second = words generated / pure generation time (excludes TTFT)
    tps_values = []
    for r in results:
        gen_time_s = (r["ttfr_ms"] - r["ttft_ms"]) / 1000
        if gen_time_s > 0.01:
            tps_values.append(r["word_count"] / gen_time_s)

    return {
        "ttft_avg": sum(ttfts) / len(ttfts),
        "ttft_min": min(ttfts),
        "ttft_max": max(ttfts),
        "ttfr_avg": sum(ttfrs) / len(ttfrs),
        "words_avg": sum(words) / len(words),
        "tps_avg": sum(tps_values) / len(tps_values) if tps_values else 0.0,
        "last_text": results[-1]["text"],
        "chunk_timeline": results[0]["chunk_timeline"],  # first run only
    }


# ── Formatting helpers ─────────────────────────────────────────────────────────
def _ms(val: float) -> str:
    return f"{val:.0f}ms"


def _num(val: float) -> str:
    return f"{val:.1f}"


def _pf(val: float, threshold: float) -> str:
    return "✅" if val < threshold else "❌"


def print_table(case_name: str, groq: dict, gemini: dict) -> None:
    rows = [
        ("TTFT (avg)", _ms(groq["ttft_avg"]), _ms(gemini["ttft_avg"])),
        ("TTFT (min)", _ms(groq["ttft_min"]), _ms(gemini["ttft_min"])),
        ("TTFT (max)", _ms(groq["ttft_max"]), _ms(gemini["ttft_max"])),
        ("TTFR (avg)", _ms(groq["ttfr_avg"]), _ms(gemini["ttfr_avg"])),
        ("Words/sec", _num(groq["tps_avg"]), _num(gemini["tps_avg"])),
        ("Response words", _num(groq["words_avg"]), _num(gemini["words_avg"])),
    ]

    print(f"\n  {case_name}")
    print("  ┌─────────────────┬──────────────┬──────────────┐")
    print("  │ Metric          │     Groq     │    Gemini    │")
    print("  ├─────────────────┼──────────────┼──────────────┤")
    for label, gv, mv in rows:
        print(f"  │ {label:<15} │ {gv:>12} │ {mv:>12} │")
    print("  └─────────────────┴──────────────┴──────────────┘")

    # Per-provider PASS/FAIL
    for name, stats in [("Groq  ", groq), ("Gemini", gemini)]:
        t = stats["ttft_avg"]
        print(
            f"  {name}  TTFT: "
            f"{_pf(t, TTFT_IDEAL)} <400ms ideal   "
            f"{_pf(t, TTFT_MAX)} <800ms max"
        )


def print_responses(label: str, prompt: str, groq_text: str, gemini_text: str) -> None:
    print(f"\n  ── {label} responses ──")
    print(f"  User:    {prompt}")
    print(f"  [Groq  ] {groq_text}")
    print(f"  [Gemini] {gemini_text}")


def print_chunk_timeline(provider: str, timeline: list[tuple]) -> None:
    if not timeline:
        print(f"\n  ── {provider} chunk timeline: (none captured)")
        return

    print(f"\n  ── {provider} streaming chunk timeline (MEDIUM case, run 1) ──")
    prev_t = 0.0
    stutter_found = False
    shown = 0

    for i, (t_ms, token) in enumerate(timeline, 1):
        gap = t_ms - prev_t
        gap_warn = "  ⚠️  GAP" if gap > CHUNK_GAP_MAX else ""
        if gap > CHUNK_GAP_MAX:
            stutter_found = True

        tok_display = repr(token[:18])
        print(
            f"  [{t_ms:7.1f}ms]  chunk {i:>3}: {tok_display:<22}  gap={gap:5.0f}ms{gap_warn}"
        )
        prev_t = t_ms
        shown += 1

        if shown >= 50:
            remaining = len(timeline) - shown
            if remaining:
                print(f"  ... ({remaining} more chunks not shown)")
            break

    verdict = (
        "❌ STUTTER RISK — gaps > 500ms detected"
        if stutter_found
        else "✅ No chunk gaps > 500ms"
    )
    print(f"  Chunk verdict: {verdict}")


def print_verdict(all_groq: list[dict], all_gemini: list[dict]) -> None:
    avg_g = sum(s["ttft_avg"] for s in all_groq) / len(all_groq)
    avg_m = sum(s["ttft_avg"] for s in all_gemini) / len(all_gemini)
    ttft_winner = "Groq" if avg_g <= avg_m else "Gemini"

    margin = abs(avg_g - avg_m)

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  VERDICT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TTFT winner:          {ttft_winner}  (margin: {margin:.0f}ms)

  Groq   avg TTFT:      {avg_g:.0f}ms
  Gemini avg TTFT:      {avg_m:.0f}ms

  Groq   {_pf(avg_g, TTFT_IDEAL)} <400ms ideal   {_pf(avg_g, TTFT_MAX)} <800ms max
  Gemini {_pf(avg_m, TTFT_IDEAL)} <400ms ideal   {_pf(avg_m, TTFT_MAX)} <800ms max

  Quality winner:       Review responses printed above — manual assessment required

  Recommended for client calls: {ttft_winner} (based on TTFT latency)
  — Confirm quality matches Sam's voice before switching.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main() -> None:
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LATENCY BENCHMARK: Groq vs Gemini
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Groq model:    {GROQ_MODEL}
  Gemini model:  {GEMINI_MODEL}
  Runs per case: {RUNS}  |  Delay: {DELAY}s  |  Cases: {len(TEST_CASES)}
  System prompt: Sam's actual PM_PROMPT_V2_BASE from Agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

    # Build clients — fail fast if keys missing
    try:
        groq_client = make_groq_client()
    except RuntimeError as e:
        print(f"❌ Groq setup failed: {e}")
        sys.exit(1)

    try:
        gemini_client = make_gemini_client()
    except RuntimeError as e:
        print(f"❌ Gemini setup failed: {e}")
        print("   Add GEMINI_API_KEY=<your-key> to .env and re-run.")
        sys.exit(1)

    all_groq: list[dict] = []
    all_gemini: list[dict] = []

    for case in TEST_CASES:
        print(f"\n{'─' * 42}")
        print(f"  {case['name']}")
        print(f"{'─' * 42}")

        groq_stats = await run_case(groq_client, GROQ_MODEL, "Groq", case)
        await asyncio.sleep(DELAY)
        gemini_stats = await run_case(gemini_client, GEMINI_MODEL, "Gemini", case)

        if groq_stats is None or gemini_stats is None:
            print("  ⚠  Skipping table — one or both providers returned no results.")
            await asyncio.sleep(DELAY)
            continue

        print_table(case["name"], groq_stats, gemini_stats)
        print_responses(
            case["label"],
            case["prompt"],
            groq_stats["last_text"],
            gemini_stats["last_text"],
        )

        if case["stream_detail"]:
            print_chunk_timeline("Groq", groq_stats["chunk_timeline"])
            print_chunk_timeline("Gemini", gemini_stats["chunk_timeline"])

        all_groq.append(groq_stats)
        all_gemini.append(gemini_stats)

        await asyncio.sleep(DELAY)

    if all_groq and all_gemini:
        print_verdict(all_groq, all_gemini)
    else:
        print("\n⚠  Not enough results for a verdict.")


if __name__ == "__main__":
    asyncio.run(main())
