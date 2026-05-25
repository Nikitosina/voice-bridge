import sys
sys.path.insert(0, '/Users/nikitarat/.hermes/hermes-agent/venv/lib/python3.11/site-packages')
from piper import PiperVoice, SynthesisConfig, AudioChunk

voice = PiperVoice.load(
    '/Users/nikitarat/.piper/models/ru_RU-denis-medium.onnx',
    config_path='/Users/nikitarat/.piper/models/ru_RU-denis-medium.onnx.json',
)

config = SynthesisConfig()
result = voice.synthesize("Привет, как дела?", config)
for i, chunk in enumerate(result):
    print(f"  chunk {i}: type={type(chunk)}")
    print(f"    AudioChunk attrs: {[x for x in dir(chunk) if not x.startswith('_')]}")
    # Try accessing it
    print(f"    audio attribute: {type(chunk.audio) if hasattr(chunk, 'audio') else 'N/A'}")
    print(f"    sample_rate: {chunk.sample_rate if hasattr(chunk, 'sample_rate') else 'N/A'}")
    print(f"    len audio: {len(chunk.audio) if hasattr(chunk, 'audio') else 'N/A'}")
    if hasattr(chunk, 'audio'):
        print(f"    audio dtype: {chunk.audio.dtype}")
    break
print("Done")
