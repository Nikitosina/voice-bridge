#!/usr/bin/env python3
"""End-to-end test using pre-recorded WAV."""
import asyncio, json, time, struct, wave, sys
import numpy as np
import websockets

WS_URL = "ws://localhost:8765/ws"
INPUT_WAV = "/tmp/piper_test.wav"
OUTPUT_WAV = "/tmp/test_output.wav"
CHUNK_SIZE = 4096

def wav_to_f32_chunks(path, chunk_size):
    with wave.open(path, 'rb') as f:
        n_frames = f.getnframes()
        raw = f.readframes(n_frames)
        # Input is 22050Hz 16-bit mono; need 16kHz float32
        arr_int16 = np.frombuffer(raw, dtype=np.int16)
        # Resample 22050→16000 roughly
        ratio = 16000 / 22050
        new_len = int(len(arr_int16) * ratio)
        arr_f32 = np.interp(np.linspace(0, len(arr_int16)-1, new_len), np.arange(len(arr_int16)), arr_int16.astype(np.float32))
        arr_f32 /= 32768.0
        for i in range(0, len(arr_f32), chunk_size):
            yield arr_f32[i:i+chunk_size].tobytes()

async def main():
    async with websockets.connect(WS_URL) as ws:
        print("[TEST] Connected. Sending audio...")
        for chunk_bytes in wav_to_f32_chunks(INPUT_WAV, CHUNK_SIZE):
            await ws.send(chunk_bytes)
            await asyncio.sleep(0.005)
        
        await ws.send(json.dumps({"type": "utterance_end"}))
        print("[TEST] Sent utterance_end, waiting...")
        
        audio = b""
        started = time.time()
        ttfb = None
        
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=30)
            if isinstance(msg, str):
                data = json.loads(msg)
                et = time.time() - started
                print(f"[TEST] [{et:.2f}s] {data}")
                if data.get("type") == "done_speaking":
                    break
            elif isinstance(msg, bytes):
                if ttfb is None:
                    ttfb = time.time() - started
                    print(f"[TEST] *** TTFB audio: {ttfb:.2f}s ***")
                audio += msg
        
        total = time.time() - started
        print(f"[TEST] Total: {total:.2f}s | Audio: {len(audio)} bytes")
        
        if audio:
            with wave.open(OUTPUT_WAV, 'wb') as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050); w.writeframes(audio)
            print(f"[TEST] Saved: {OUTPUT_WAV}")

asyncio.run(main())
