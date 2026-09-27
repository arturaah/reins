"""Independent STT and TTS adapters. All generated audio is mono PCM16 at 24 kHz."""
import base64
import io
import wave
import httpx
from .audio import MAX_SECONDS, RATE

STT_MODEL = 'gpt-transcribe'
GEMINI_TTS_MODEL = 'gemini-3.8-flash-lite-tts'
CARTESIA_TTS_MODEL = 'sonic-3.6'
MAX_OUTPUT = MAX_SECONDS * RATE * 2


def wav_bytes(pcm, rate=16000):
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(rate)
        wav.writeframes(pcm)
    return output.getvalue()


def decode_wav(raw):
    if len(raw) > MAX_OUTPUT + 65536:
        raise ValueError('TTS response too long')
    with wave.open(io.BytesIO(raw), 'rb') as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, RATE):
            raise ValueError('Unexpected TTS WAV format')
        if not 0 < wav.getnframes() <= MAX_SECONDS * RATE:
            raise ValueError('Invalid TTS duration')
        pcm = wav.readframes(wav.getnframes())
        if len(pcm) != wav.getnframes() * 2:
            raise ValueError('Truncated TTS WAV')
        return pcm


class OpenAISTT:
    def __init__(self, key, model=STT_MODEL):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=key, timeout=30, max_retries=0)
        self.model = model

    async def transcribe(self, pcm):
        response = await self.client.audio.transcriptions.create(
            model=self.model, file=('speech.wav', wav_bytes(pcm), 'audio/wav'))
        text = response.text.strip()
        if not 1 <= len(text) <= 1000:
            raise ValueError('Empty or oversized transcription')
        return text

    async def close(self):
        await self.client.close()


class GeminiTTS:
    voice = 'Charon'

    def __init__(self, key, model=GEMINI_TTS_MODEL):
        from google import genai
        self.client = genai.Client(api_key=key)
        self.model = model

    async def speak(self, text):
        result = await self.client.aio.interactions.create(
            model=self.model,
            input=[{'type': 'user_input', 'content': [{'type': 'text', 'text': text,
                    'annotations': [{'type': 'speech_metadata',
                                     'style': 'Low masculine voice, measured clipped delivery, dry warmth.'}]}]}],
            response_format={'type': 'audio'},
            generation_config={'speech_config': [{'voice': self.voice}]}, timeout=30)
        encoded = result.output_audio.data
        if len(encoded) > (MAX_OUTPUT + 65536) * 4 // 3 + 4:
            raise ValueError('TTS response too long')
        return decode_wav(base64.b64decode(encoded, validate=True))

    async def close(self):
        await self.client.aio.aclose()


class CartesiaTTS:
    def __init__(self, key, voice, model=CARTESIA_TTS_MODEL):
        self.model, self.voice = model, voice
        self.client = httpx.AsyncClient(timeout=30, headers={
            'Authorization': f'Bearer {key}', 'Cartesia-Version': '2026-08-14'})

    async def speak(self, text):
        async with self.client.stream('POST', 'https://api.cartesia.ai/tts/bytes', json={
                'model_id': self.model, 'transcript': text, 'voice': self.voice,
                'output_format': {'container': 'raw', 'encoding': 'pcm_s16le', 'sample_rate': RATE}}) as response:
            response.raise_for_status()
            pcm = bytearray()
            async for chunk in response.aiter_bytes():
                pcm.extend(chunk)
                if len(pcm) > MAX_OUTPUT:
                    raise ValueError('TTS response too long')
        if not pcm or len(pcm) % 2:
            raise ValueError('Invalid Cartesia PCM')
        return bytes(pcm)

    async def close(self):
        await self.client.aclose()
