"""
Cartesia WebSocket TTS — A/B test vs dashboard.

Sends your test paragraph through the same WebSocket API your bot uses,
saves the output as a WAV file. Play it side-by-side with the same paragraph
in Cartesia's dashboard to compare.

If WAV sounds the same as dashboard → TTS is fine, LLM-generated text is the bottleneck.
If WAV sounds flatter than dashboard → something about WS streaming differs from dashboard playback.

Usage:
    pip install websockets python-dotenv
    # set CARTESIA_API_KEY in .env or env var
    python test.py
    # then play test_output.wav and compare to dashboard
"""

import asyncio
import base64
import json
import os
import time
import wave

import websockets
from dotenv import load_dotenv

load_dotenv()

CARTESIA_API_KEY = os.environ.get("CARTESIA_API_KEY")
if not CARTESIA_API_KEY:
    raise RuntimeError("Set CARTESIA_API_KEY in env or .env file")

CARTESIA_VOICE_ID = "ee54607b-84f1-4335-b37b-51b501e8dcdb"
CARTESIA_MODEL = "sonic-3.5"
CARTESIA_WS_URL = "wss://api.cartesia.ai/tts/websocket"
SAMPLE_RATE = 44100  # match dashboard quality
OUTPUT_FILE = "test_output.wav"

PARAGRAPH = (
    "So, look... I've got mixed news about the launch. The good part? "
    "The new payment integration finally went live this morning — and honestly, "
    "it just worked! No rollbacks. No panic in the channel. Nothing broke. "
    "After all the late nights we put into this, I genuinely cannot believe we "
    "pulled it off! The team should be really proud. But... I have to be honest "
    "with you about Friday's release. We're not going to make it. The QA backlog "
    "is bigger than I realized, and I should have caught that two sprints ago. "
)


async def main():
    url = f"{CARTESIA_WS_URL}?api_key={CARTESIA_API_KEY}&cartesia_version=2024-06-10"

    word_count = len(PARAGRAPH.split())
    print(f"Voice ID:    {CARTESIA_VOICE_ID}")
    print(f"Model:       {CARTESIA_MODEL}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Words:       {word_count}")
    print(f"Connecting...")

    audio_chunks = []
    t_start = time.time()
    t_first_byte = None
    chunk_count = 0

    async with websockets.connect(url, ping_interval=None) as ws:
        request = {
            "context_id": f"test-{int(time.time())}",
            "model_id": CARTESIA_MODEL,
            "voice": {
                "mode": "id",
                "id": CARTESIA_VOICE_ID,
            },
            "transcript": PARAGRAPH,
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": SAMPLE_RATE,
            },
            "language": "en",
            "add_timestamps": False,
            "continue": False,
        }

        await ws.send(json.dumps(request))
        print(
            f"Request sent at {(time.time() - t_start) * 1000:.0f} ms — waiting for audio..."
        )

        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
            except asyncio.TimeoutError:
                print("ERROR: timed out waiting for response")
                return

            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue

            mtype = data.get("type", "")

            # Handle both "chunk" and "audio" message types (different API versions)
            if mtype in ("chunk", "audio"):
                audio_b64 = data.get("data") or data.get("audio")
                if audio_b64:
                    if t_first_byte is None:
                        t_first_byte = time.time() - t_start
                        print(f"First audio chunk: {t_first_byte * 1000:.0f} ms")
                    audio_chunks.append(base64.b64decode(audio_b64))
                    chunk_count += 1

            elif mtype == "done":
                t_done = time.time() - t_start
                print(
                    f"Stream complete:   {t_done * 1000:.0f} ms ({chunk_count} chunks)"
                )
                break

            elif mtype == "error":
                print(f"ERROR from Cartesia: {data}")
                return

            elif mtype == "timestamps":
                # ignore — we disabled them but server might send anyway
                pass

    # Save raw PCM as WAV
    raw_audio = b"".join(audio_chunks)
    if not raw_audio:
        print("ERROR: no audio received")
        return

    with wave.open(OUTPUT_FILE, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM = 2 bytes per sample
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_audio)

    bytes_received = len(raw_audio)
    seconds_audio = bytes_received / (SAMPLE_RATE * 2)
    realtime_factor = (time.time() - t_start) / seconds_audio if seconds_audio else 0

    print()
    print(f"Audio bytes:        {bytes_received:,}")
    print(f"Speech duration:    {seconds_audio:.2f}s")
    print(f"Realtime factor:    {realtime_factor:.2f}x  (lower = faster)")
    print(f"Saved to:           {OUTPUT_FILE}")
    print()
    print(
        f"Now: play {OUTPUT_FILE} and play the same paragraph in Cartesia's dashboard"
    )
    print(f"with the same voice + sonic-3.5. Listen for ellipses, em-dashes, rhythm.")
    print()
    print(
        f"  -> Sound the same?  TTS is fine. The flatness in your bot is LLM-generated text."
    )
    print(
        f"  -> WAV sounds flat? Something in the WS streaming differs from dashboard playback."
    )


if __name__ == "__main__":
    asyncio.run(main())
