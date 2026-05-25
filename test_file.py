#!/usr/bin/env python3
import asyncio, json, struct, wave, time
import websockets

WS_URL = "ws://127.0.0.1:8765/ws"
INPUT_WAV = "/tmp/test_16k.wav"
OUTPUT_WAV = "/tmp/test_out.wav"


async def main():
    with wave.open(INPUT_WAV, "rb") as f:
        n_frames = f.getnframes()
        raw = f.readframes(n_frames)
    # 16-bit → float32 [-1,1] → struct-packed little-endian
    ints = struct.unpack(f"<{n_frames}h", raw)
    print(f"[TEST] Frames: {n_frames} ({len(raw)} bytes)")
    # Pack floats
    chunk_size = 2048
    chunks = []
    for i in range(0, len(ints), chunk_size):
        chunk = ints[i : i + chunk_size]
        floats = [s / 32768.0 for s in chunk]
        chunks.append(struct.pack(f"<{len(floats)}f", *floats))

    async with websockets.connect(WS_URL, open_timeout=10) as ws:
        print("[TEST] Connected")
        for i, chunk in enumerate(chunks):
            await ws.send(chunk)
            await asyncio.sleep(0.001)
        await ws.send(json.dumps({"type": "utterance_end"}))
        print("[TEST] Sent utterance_end")

        audio = b""
        started = time.time()
        ttfb = None

        while True:
            elapsed = time.time() - started
            if elapsed > 120:
                print("[TEST] ABORT after 120s")
                break
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=60)
            except asyncio.TimeoutError:
                print("[TEST] No message for 60s")
                break
            if isinstance(msg, str):
                data = json.loads(msg)
                et = time.time() - started
                print(f"[TEST] [{et:.2f}s] TEXT {data}")
                if data.get("type") == "done_speaking":
                    break
            elif isinstance(msg, bytes):
                if ttfb is None:
                    ttfb = time.time() - started
                    print(f"[TEST] *** FIRST AUDIO at {ttfb:.2f}s ***")
                audio += msg

        total = time.time() - started
        print(f"[TEST] END total={total:.2f}s audio={len(audio)} bytes")
        if audio:
            with wave.open(OUTPUT_WAV, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)
                wf.writeframes(audio)
            print(f"[TEST] Saved {OUTPUT_WAV}")


asyncio.run(main())
