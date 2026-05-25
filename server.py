#!/usr/bin/env python3
"""
Voice Call Bridge Server — FastAPI + WebSocket
===============================================
Session flow (per-WebSocket connection):

  LISTENING  ──(iOS sends binary PCM f32 chunks)──▶  PROCESSING
      ▲                                                  │
      │     (STT → LLM streaming → TTS streaming)        │
      │                                                  ▼
      └──(all audio sent back to iOS)──────────────  SPEAKING

iOS triggers utterance send (no VAD on server). The server:
  1. Accumulates PCM f32 audio chunks during LISTENING.
  2. Transcribes via faster-whisper (small model).
  3. Streams transcription to Ollama/Kimi (OpenAI-compatible).
  4. On sentence boundaries, feeds text into Piper TTS.
  5. Yields int16 PCM audio chunks back over the WebSocket.
"""

import asyncio
import json
import logging
import os
import struct
import io
import time
from enum import Enum
from typing import AsyncGenerator, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# Configuration via environment variables
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "gemma4:31b-cloud"
    whisper_model_size: str = "small"  # fallback to "base" if small unavailable
    piper_model_path: str = "/Users/nikitarat/.piper/models/ru_RU-dmitri-medium.onnx"
    piper_model_config: str = "/Users/nikitarat/.piper/models/ru_RU-dmitri-medium.onnx.json"
    host: str = "0.0.0.0"
    port: int = 8765

    class Config:
        env_prefix = "VOICE_BRIDGE_"


settings = Settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("voice-bridge")

# ---------------------------------------------------------------------------
# Session state machine
# ---------------------------------------------------------------------------

class SessionState(str, Enum):
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"


class VoiceSession:
    """
    Per-WebSocket session. Holds accumulated audio, state, and references
    to STT/LLM/TTS pipeline resources.
    """

    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.state = SessionState.LISTENING
        self.audio_buffer: list[np.ndarray] = []  # list of PCM f32 arrays
        self.conversation_history: list[dict] = [
            {"role": "system", "content": "You are a helpful voice assistant. Answer concisely."}
        ]
        self.last_utterance_text: str = ""

    def add_audio(self, chunk: bytes):
        """Accumulate raw PCM f32 chunk (little-endian float32)."""
        # Each chunk is a flat buffer of 32-bit floats
        arr = np.frombuffer(chunk, dtype=np.float32)
        self.audio_buffer.append(arr)

    def get_audio_np(self) -> np.ndarray:
        """Concatenate all accumulated audio into a single float32 array."""
        if not self.audio_buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.audio_buffer)

    def reset_audio(self):
        self.audio_buffer.clear()


# ---------------------------------------------------------------------------
# STT: faster-whisper
# ---------------------------------------------------------------------------

class SpeechToText:
    """
    Wraps faster-whisper for transcribing accumulated PCM f32 audio.
    Loads the model lazily on first use.
    """

    def __init__(self):
        self._model = None
        self._model_size = settings.whisper_model_size

    def _load_model(self):
        if self._model is not None:
            return
        from faster_whisper import WhisperModel
        try:
            self._model = WhisperModel(self._model_size, device="cpu", compute_type="float32")
            log.info("faster-whisper model loaded: %s", self._model_size)
        except Exception as exc:
            log.warning("Failed to load '%s' model: %s. Falling back to 'base'.", self._model_size, exc)
            self._model = WhisperModel("base", device="cpu", compute_type="float32")
            log.info("faster-whisper model loaded: base")

    async def transcribe(self, audio_np: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe float32 audio array (normalised -1..1) at 16 kHz.
        faster-whisper expects float32 in [-1, 1] range.
        Returns the recognised text.
        """
        self._load_model()
        # faster-whisper runs the model synchronously; run in executor to avoid
        # blocking the async event loop.
        loop = asyncio.get_running_loop()

        def _run():
            segments, info = self._model.transcribe(audio_np, beam_size=5, language="ru")
            text_parts = [seg.text for seg in segments]
            return " ".join(text_parts)

        text = await loop.run_in_executor(None, _run)
        log.info("STT result: %r", text)
        return text.strip()


# ---------------------------------------------------------------------------
# LLM: OpenAI-compatible streaming (Ollama / Kimi)
# ---------------------------------------------------------------------------

class StreamingLLM:
    """
    Stream tokens from an OpenAI-compatible chat completion endpoint.
    Yields text chunks; detects sentence boundaries for TTS handoff.
    """

    def __init__(self):
        self.base_url = settings.llm_base_url.rstrip("/")
        self.model = settings.llm_model

    async def stream_response(
        self, conversation: list[dict]
    ) -> AsyncGenerator[str, None]:
        """
        Stream response from LLM. Yields text fragments.
        The caller is responsible for sentence-boundary detection.
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=f"{self.base_url}", api_key="ollama")
        stream = await client.chat.completions.create(
            model=self.model,
            messages=conversation,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content


# ---------------------------------------------------------------------------
# TTS: Piper
# ---------------------------------------------------------------------------

class PiperTTS:
    """
    Loads a Piper voice model and streams int16 PCM audio for each text chunk.
    """

    def __init__(self):
        self._voice = None
        self._sample_rate = 22050

    def _load_voice(self):
        if self._voice is not None:
            return
        from piper import PiperVoice, SynthesisConfig

        # Load voice from ONNX model file
        self._voice = PiperVoice.load(
            settings.piper_model_path,
            config_path=settings.piper_model_config,
        )
        self._config = SynthesisConfig()
        log.info("Piper voice loaded from %s", settings.piper_model_path)

    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Synthesize text with Piper and yield int16 PCM chunks (22050 Hz, mono).
        Each AudioChunk from Piper contains audio_int16_bytes.
        """
        self._load_voice()
        loop = asyncio.get_running_loop()

        def _synth():
            return list(self._voice.synthesize(text, self._config))

        audio_chunks = await loop.run_in_executor(None, _synth)

        for chunk in audio_chunks:
            # chunk is piper.voice.AudioChunk with audio_int16_bytes attribute
            yield chunk.audio_int16_bytes


# ---------------------------------------------------------------------------
# Pipeline orchestrator per session
# ---------------------------------------------------------------------------

class VoicePipeline:
    """
    Runs the full STT→LLM→TTS pipeline for one utterance.
    """

    def __init__(self, session: VoiceSession, stt: SpeechToText, llm: StreamingLLM, tts: PiperTTS):
        self.session = session
        self.stt = stt
        self.llm = llm
        self.tts = tts

    async def run(self):
        """
        Execute the pipeline:
          1. STT: transcribe accumulated audio.
          2. LLM: stream response from LLM; detect sentence boundaries.
          3. TTS: synthesise each sentence and stream int16 PCM back.
        """
        self.session.state = SessionState.PROCESSING

        # --- STEP 1: STT ---
        audio_np = self.session.get_audio_np()
        if audio_np.size == 0:
            log.warning("Empty audio buffer — skipping pipeline.")
            self.session.state = SessionState.LISTENING
            return

        # Normalise and resample to 16 kHz for whisper
        # Audio is already PCM f32 [-1, 1]; whisper expects 16 kHz
        # If iOS sends at a different rate, resample here. We assume 16 kHz or
        # whisper handles it via internal resampling.
        utterance = await self.stt.transcribe(audio_np, sample_rate=16000)
        if not utterance:
            log.info("No speech detected, returning to LISTENING.")
            self.session.state = SessionState.LISTENING
            return

        self.session.last_utterance_text = utterance
        self.session.conversation_history.append({"role": "user", "content": utterance})

        # --- STEP 2: LLM streaming with sentence boundary detection ---
        acc_text = ""
        sentence_buffer = ""

        # Notify iOS that processing started (optional text message)
        await self.session.ws.send_text(json.dumps({"type": "processing", "stt": utterance}))

        log.info("LLM stream starting for utterance: %r", utterance)
        self.session.state = SessionState.SPEAKING  # we'll stream audio back

        async for token in self.llm.stream_response(self.session.conversation_history):
            sentence_buffer += token
            # Check for sentence boundary: . ! ? followed by space or end
            if sentence_buffer and sentence_buffer[-1] in ".!?":
                # Found a sentence boundary — synthesise and send
                sentence_text = sentence_buffer.strip()
                if sentence_text:
                    log.info("TTS sentence: %r", sentence_text)
                    # Accumulate full text for history
                    acc_text += sentence_text + " "
                    # Synthesise this sentence and send audio chunks
                    async for audio_chunk in self.tts.synthesize(sentence_text):
                        await self.session.ws.send_bytes(audio_chunk)
                sentence_buffer = ""

        # Flush remaining text (last sentence without trailing punctuation)
        if sentence_buffer.strip():
            sentence_text = sentence_buffer.strip()
            log.info("TTS final fragment: %r", sentence_text)
            acc_text += sentence_text
            async for audio_chunk in self.tts.synthesize(sentence_text):
                await self.session.ws.send_bytes(audio_chunk)

        # Store assistant response in conversation history
        if acc_text:
            self.session.conversation_history.append({"role": "assistant", "content": acc_text.strip()})

        # Signal end of audio stream (iOS can stop playback)
        await self.session.ws.send_text(json.dumps({"type": "done_speaking"}))

        # Cleanup for next round
        self.session.reset_audio()
        self.session.state = SessionState.LISTENING


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Voice Bridge Server")

# Global pipeline components (shared across connections; whisper/piper are read-only)
stt = SpeechToText()
llm = StreamingLLM()
tts = PiperTTS()


@app.get("/health")
async def health():
    """Health-check endpoint."""
    return {"status": "ok", "state": "running"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket audio bridge endpoint.

    Protocol:
      - iOS sends binary PCM f32 chunks (little-endian float32, 16 kHz, mono).
      - iOS sends a text message '{"type":"utterance_end"}' to signal utterance complete.
      - Server processes the utterance through STT→LLM→TTS.
      - Server sends binary int16 PCM chunks back (22050 Hz, mono).
      - Server sends text '{"type":"done_speaking"}' when TTS output is complete.
      - Cycle repeats.
    """
    await websocket.accept()
    session = VoiceSession(websocket)
    pipeline = VoicePipeline(session, stt, llm, tts)

    log.info("WebSocket connected — session started.")

    try:
        while True:
            # Wait for a message (binary audio or text command from iOS)
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message:
                # Binary PCM f32 audio chunk — accumulate
                if session.state != SessionState.LISTENING:
                    # In theory iOS should not send during processing/speaking;
                    # ignore or log warning.
                    log.warning("Audio chunk received in state %s — ignoring.", session.state)
                    continue
                session.add_audio(message["bytes"])

            elif "text" in message:
                # Text command from iOS
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    log.warning("Invalid JSON text message: %r", message["text"])
                    continue

                cmd = data.get("type", "")

                if cmd == "utterance_end":
                    # iOS finished sending audio for this utterance
                    if session.state != SessionState.LISTENING:
                        log.warning("utterance_end received in state %s — ignoring.", session.state)
                        continue
                    # Launch pipeline (non-blocking — we stay in the receive loop
                    # but pipeline sends audio back over the same websocket).
                    # We run the pipeline synchronously here since the WebSocket
                    # is single-threaded per connection; this blocks receiving
                    # until pipeline is done (desired behaviour).
                    await pipeline.run()

                elif cmd == "reset":
                    # iOS requests session reset
                    session.reset_audio()
                    session.state = SessionState.LISTENING
                    session.conversation_history = session.conversation_history[:1]  # keep system prompt
                    log.info("Session reset by iOS.")

                else:
                    log.info("Unknown text command: %s", cmd)

    except WebSocketDisconnect:
        log.info("WebSocket disconnected.")
    except Exception as exc:
        log.error("WebSocket error: %s", exc, exc_info=True)
    finally:
        log.info("Session ended.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
    )
