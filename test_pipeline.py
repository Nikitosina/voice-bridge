#!/usr/bin/env python3
"""
Test: Record mic → send via WS → receive TTS back → play/save result.
"""
import asyncio
import json
import struct
import time
import sys
import wave
import subprocess
import tempfile
import os

import numpy as np
import websockets

WS_URL = "ws://localhost:8765/ws"
RECORD_SECONDS = 4
RECORD_WAV = "/tmp/test_input.wav"
OUTPUT_WAV = "/tmp/test_output.wav"
CHUNK_SIZE = 2048  # float32 frames (~128ms at 16kHz)

async def record_mic():
    """Record 4 seconds of mono 16kHz 16-bit WAV via ffmpeg avfoundation."""
    print(f"[REC] Recording {RECORD_SECONDS}s — SPEAK NOW (Russian)...")
    cmd = [
        "ffmpeg", "-y",
        "-f", "avfoundation",
        "-i", ":0",          # default mic
        "-t", str(RECORD_SECONDS),
        "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        RECORD_WAV
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        print("[REC] ffmpeg failed:", proc.stderr.decode()[-200:])
        sys.exit(1)
    print(f"[REC] Saved to {RECORD_WAV}")

def wav_to_float32_pcm(path):
    """Read mono 16-bit WAV → float32 PCM array [-1,1]."""
    with wave.open(path, 'rb') as f:
        n_channels = f.getnchannels()
        assert n_channels == 1, "Mono required"
        n_frames = f.getnframes()
        raw = f.readframes(n_frames)
        # 16-bit little-endian → int16
        arr_int16 = np.frombuffer(raw, dtype=np.int16)
        # Normalize to float32 [-1, 1]
        arr_f32 = arr_int16.astype(np.float32) / 32768.0
        return arr_f32

async def test_pipeline():
    # 1) Record
    await record_mic()
    
    # 2) Convert to f32 chunks
    pcm = wav_to_float32_pcm(RECORD_WAV)
    total_bytes = len(pcm) * 4
    print(f"[TEST] PCM frames: {len(pcm)} ({total_bytes} bytes)")
    
    # 3) Connect WS
    print(f"[TEST] Connecting to {WS_URL} ...")
    async with websockets.connect(WS_URL) as ws:
        print("[TEST] Connected. Streaming audio...")
        
        # Send audio in chunks
        for i in range(0, len(pcm), CHUNK_SIZE):
            chunk = pcm[i:i+CHUNK_SIZE]
            chunk_bytes = chunk.tobytes()
            await ws.send(chunk_bytes)
            # Small yield to avoid flooding
            if i % (CHUNK_SIZE * 8) == 0:
                await asyncio.sleep(0.01)
        
        # Send utterance_end
        await ws.send(json.dumps({"type": "utterance_end"}))
        print("[TEST] Sent utterance_end. Waiting for TTS response...")
        
        # Receive: first text messages (processing), then audio chunks, then done_speaking
        audio_buffer = b""
        started = time.time()
        ttfb = None
        
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
            except asyncio.TimeoutError:
                print("[TEST] Timeout waiting for response")
                break
            
            if isinstance(msg, str):
                data = json.loads(msg)
                et = time.time() - started
                print(f"[TEST] [{et:.2f}s] text msg: {data}")
                if data.get("type") == "done_speaking":
                    break
            elif isinstance(msg, bytes):
                if ttfb is None:
                    ttfb = time.time() - started
                    print(f"[TEST] *** TTFB audio: {ttfb:.2f}s ***")
                audio_buffer += msg
        
        total = time.time() - started
        print(f"[TEST] Total pipeline time: {total:.2f}s")
        print(f"[TEST] Audio received: {len(audio_buffer)} bytes")
        
        # Save output
        if audio_buffer:
            n_frames = len(audio_buffer) // 2
            with wave.open(OUTPUT_WAV, 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(22050)
                f.writeframes(audio_buffer)
            print(f"[TEST] Output saved: {OUTPUT_WAV} ({n_frames/22050:.1f}s @ 22050Hz)")
            
            # Convert to ogg for Telegram
            subprocess.run([
                "ffmpeg", "-y", "-i", OUTPUT_WAV,
                "-c:a", "libopus", "-b:a", "32k", "/tmp/test_output.ogg"
            ], capture_output=True)
            print("[TEST] Also saved: /tmp/test_output.ogg")
        else:
            print("[TEST] NO AUDIO RECEIVED!")

if __name__ == "__main__":
    asyncio.run(test_pipeline())
