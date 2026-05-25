#!/usr/bin/env python3
"""Record via ffmpeg, send to bridge, receive TTS back, save result."""
import asyncio, json, os, subprocess, struct, time, wave
import websockets

WS_URL = "ws://127.0.0.1:8765/ws"
RECORD_WAV = "/tmp/test_rec_16k.wav"
OUTPUT_WAV = "/tmp/test_out.wav"


def record():
    secs = 5
    print(f"[REC] Speak now! Recording {secs}s...")
    cmd = [
        "ffmpeg", "-y", "-f", "avfoundation", "-i", ":0",
        "-t", str(secs), "-ac", "1", "-ar", "16000",
        "-acodec", "pcm_s16le", RECORD_WAV,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print("ffmpeg err:", r.stderr.decode()[-300:])
        return False
    print("[REC] Saved", RECORD_WAV)
    return True


def wav_to_f32_chunks(path, chunk_frames=2048):
    with wave.open(path, "rb") as wf:
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        arr_int16 = struct.unpack(f"<{n_frames}h", raw)
    # float32 [-1,1]
    floats = [x / 32768.0 for x in arr_int16]
    for i in range(0, len(floats), chunk_frames):
        chunk = floats[i : i + chunk_frames]
        yield struct.pack(f"<{len(chunk)}f", *chunk)


async def pipeline():
    if not record():
        return
    async with websockets.connect(WS_URL) as ws:
        print("[TEST] WS connected, streaming audio...")
        for chunk in wav_to_f32_chunks(RECORD_WAV, 2048):
            await ws.send(chunk)
            await asyncio.sleep(0.002)
        await ws.send(json.dumps({"type": "utterance_end"}))
        print("[TEST] Sent utterance_end, waiting for response...")

        audio = b""
        started = time.time()
        ttfb = None
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=60)
            except asyncio.TimeoutError:
                print("[TEST] TIMEOUT — no response from server in 60s")
                break
            if isinstance(msg, str):
                data = json.loads(msg)
                et = time.time() - started
                print(f"[TEST] [{et:.2f}s] {data}")
                if data.get("type") == "done_speaking":
                    break
            elif isinstance(msg, bytes):
                if ttfb is None:
                    ttfb = time.time() - started
                    print(f"[TEST] *** FIRST AUDIO CHUNK at {ttfb:.2f}s ***")
                audio += msg

        total = time.time() - started
        print(f"[TEST] END total={total:.2f}s audio_bytes={len(audio)}")
        if audio:
            nsamp = len(audio) // 2
            with wave.open(OUTPUT_WAV, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)
                wf.writeframes(audio)
            print(f"[TEST] Output saved: {OUTPUT_WAV} ({nsamp / 22050:.1f}s)")
            subprocess.run([
                "ffmpeg", "-y", "-i", OUTPUT_WAV,
                "-c:a", "libopus", "-b:a", "32k", "/tmp/test_out.ogg"
            ], capture_output=True)
            print("[TEST] Also: /tmp/test_out.ogg")


asyncio.run(pipeline())
