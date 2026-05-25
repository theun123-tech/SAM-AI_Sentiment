# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sam is a Python-based AI Meeting Agent that runs live voice calls via Recall.ai, transcribes speech (Deepgram Flux/Nova-3), speaks via Cartesia TTS (WebSocket streaming), and manages Jira tickets. It serves two modes: **client call** and **developer standup**.

## Running the Server

```bash
# Activate virtualenv first
venv\Scripts\activate        # Windows
source venv/bin/activate     # Unix

# Start the server (default PORT=8000)
python server.py

# Cartesia TTS A/B test harness
python test.py
```

No build step — pure Python. No standard test framework (pytest is not configured).

## Key Feature Flags (`.env`)

These control runtime behavior — check `.env` before assuming defaults:

| Variable | Effect |
|---|---|
| `USE_DIALOGUE_MANAGER` | `2` = LangGraph observer mode, `0` = disabled |
| `USE_DEEPGRAM` | `true` = Deepgram Nova-3, `false` = per-speaker Flux |
| `USE_PER_SPEAKER_FLUX` | `1` = multi-party Flux STT (client mode only) |
| `USE_OUTPUT_MEDIA` | `true` = Cartesia WebSocket PCM streaming |
| `USE_UNIFIED_RESEARCH` | `1` = unified web search flow |
| `NLU_PROVIDER_ORDER` | `groq_first` or default ordering |
| `PORT` | Server port (default 8000) |

## LLM Provider Hierarchy

- **Primary**: Groq Llama 3.3 70B — fast Q&A, NLU, routing (`GROQ_API_KEYS` holds a pool of 12 keys, round-robin rotated with 60s cooldown on 401/429)
- **Fallback**: Azure OpenAI GPT-4o mini — complex extraction (Jira action items, post-meeting summaries)
- **Web search**: Exa (primary), SerpAPI (fallback)

## Non-Obvious Gotchas

**Auto-concatenated source files**: `external_apis.py`, `dialogue.py`, `stt.py`, and `groq_client.py` are programmatically merged from multiple original files by `consolidate_to_folder.py`. Section headers in their docstrings mark original file boundaries. Do not refactor these files assuming they are standalone — the original source files are the authoritative split.

**FFmpeg hardcoded path**: Pydub is configured with the absolute path `C:\Users\user\Downloads\ffmpeg-8.1-full_build\bin\ffmpeg.exe`. If audio format conversion fails, this is the first thing to check.

**Dialogue manager silent disable**: If any dependency of the LangGraph dialogue manager fails to import, `_DIALOGUE_MANAGER_AVAILABLE` is set to `False` and the whole system silently falls back — no crash, no warning in logs.

**Recall.ai webhooks require tunnel**: `TUNNEL_URL` in `.env` must point to a live ngrok (or equivalent) URL for Recall.ai to deliver webhook events.

**Standup mode ignores `USE_PER_SPEAKER_FLUX`**: Standup always uses Recall.ai's built-in STT regardless of this flag.

**Session memory keyed by attendee hash**: Same attendees in different meetings share `conversation_id` and load prior summaries — intentional, privacy-safe design.

## File-Based Persistence

No database. All state is stored as JSON files in the project root:

- `sessions.json` — Meeting transcripts, action items, Jira keys
- `settings.json` — Jira/Azure config (editable from dashboard UI)
- `standups.json` — Developer standup records
- `pending_tickets.json` — Jira ticket creation retries
- `conversation_summaries.json` — Per-attendee-group memory (keyed by hash)
- `meeting_setups.json` — Per-session agenda/tickets/scope
- `agenda_templates.json` — Reusable agenda templates

## Auth

The REST API uses JWT auth (`JWT_SECRET` in `.env`). Admin credentials are also in `.env`. The dashboard (`index.html`) handles login flow.

## Required Environment Variables

All keys and config live in `.env` (never commit this file). Critical groups:

- Groq: `GROQ_API_KEYS` (comma-separated pool of up to 12)
- Azure: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`
- Recall.ai: `RECALL_API_KEY`, `RECALL_REGION`, `TUNNEL_URL`
- Cartesia: `CARTESIA_API_KEY`
- Deepgram: `DEEPGRAM_API_KEY`
- Jira: `JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`, `JIRA_PROJECT_KEY`
