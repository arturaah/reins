"""Run local speech-to-text and text-to-speech, with ai-coustics input processing."""
import argparse
import os
from pathlib import Path
from urllib.parse import urlsplit
from dotenv import dotenv_values
import uvicorn
from .server import create_app


def load_keys(path):
    values = dotenv_values(path) if path else {}
    for name in ('GEMINI_API_KEY', 'OPENAI_API_KEY', 'AIC_SDK_LICENSE', 'CARTESIA_API_KEY', 'CARTESIA_VOICE_ID'):
        if not os.environ.get(name) and values.get(name):
            os.environ[name] = values[name]


def local_origin(value):
    url = urlsplit(value)
    if (url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost')
            or url.username or url.password or url.query or url.fragment or url.path not in ('', '/')):
        raise argparse.ArgumentTypeError('Use a loopback HTTP origin')
    return value.rstrip('/')


def main():
    from .speech import STT_MODEL, GEMINI_TTS_MODEL, CARTESIA_TTS_MODEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['sim'], default='sim')
    parser.add_argument('--demo', action='store_true', help='Offline transport test tone; no providers')
    parser.add_argument('--stt-model', default=STT_MODEL)
    parser.add_argument('--tts', choices=['gemini', 'cartesia'], default='gemini')
    parser.add_argument('--tts-model', help='Exact compatible model ID for the chosen TTS provider')
    parser.add_argument('--tts-voice', help='Cartesia voice ID (or CARTESIA_VOICE_ID)')
    parser.add_argument('--dashboard-origin', type=local_origin, default='http://127.0.0.1:8091')
    parser.add_argument('--voice-focus', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--tyto', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--vad', choices=['webrtc', 'aic'], default='aic')
    parser.add_argument('--interference-threshold', type=float, default=.6)
    parser.add_argument('--noise-threshold', type=float, default=.8)
    parser.add_argument('--model-cache', type=Path, default=Path('.voice-cache/models'))
    parser.add_argument('--key-file', type=Path, help='Local .env; keys stay in this process')
    parser.add_argument('--port', type=int, default=8770)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or not all(0 < v <= 1 for v in (args.noise_threshold, args.interference_threshold)):
        parser.error('Invalid port or audio score threshold')
    load_keys(args.key_file)
    options = {'public_config': {'dashboard_origin': args.dashboard_origin}}
    if not args.demo:
        from .acoustics import Acoustics
        from .cascade import session_factory
        from .speech import OpenAISTT, GeminiTTS, CartesiaTTS
        required = ['OPENAI_API_KEY', 'GEMINI_API_KEY' if args.tts == 'gemini' else 'CARTESIA_API_KEY']
        if args.voice_focus or args.tyto or args.vad == 'aic': required.append('AIC_SDK_LICENSE')
        for name in required:
            if not os.environ.get(name): parser.error(f'Set {name}, provide --key-file, or use --demo')
        voice = 'Charon' if args.tts == 'gemini' else (args.tts_voice or os.environ.get('CARTESIA_VOICE_ID'))
        if not voice: parser.error('Cartesia needs --tts-voice or CARTESIA_VOICE_ID')
        model = args.tts_model or (GEMINI_TTS_MODEL if args.tts == 'gemini' else CARTESIA_TTS_MODEL)
        try:
            acoustics = Acoustics(focus=args.voice_focus, tyto=args.tyto, vad=args.vad,
                                  license_key=os.environ.get('AIC_SDK_LICENSE', ''), cache=args.model_cache)
        except Exception as error:
            parser.error(f'Acoustics setup failed ({type(error).__name__}). Check SDK installation, license and connection.')
        make_tts = (lambda: GeminiTTS(os.environ['GEMINI_API_KEY'], model)) if args.tts == 'gemini' else (
                    lambda: CartesiaTTS(os.environ['CARTESIA_API_KEY'], voice, model))
        options['session_factory'] = session_factory(
            make_stt=lambda: OpenAISTT(os.environ['OPENAI_API_KEY'], args.stt_model), make_tts=make_tts,
            acoustics=acoustics, noise_threshold=args.interference_threshold, background_threshold=args.noise_threshold)
        options['public_config'].update(voice=voice, stt_model=args.stt_model, tts_model=model,
                                        voice_focus=args.voice_focus, tyto=args.tyto, vad=args.vad)
    provider = 'demo' if args.demo else 'audio'
    print(f'Reins voice → http://127.0.0.1:{args.port} · simulation · {provider}', flush=True)
    uvicorn.run(create_app(port=args.port, provider=provider, **options),
                host='127.0.0.1', port=args.port, log_level='warning', ws_max_size=32768)


if __name__ == '__main__':
    main()
