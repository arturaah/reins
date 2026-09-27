"""Public, bounded speech diagnostics. Never expose provider payloads or credentials."""
MESSAGES = {
    'speaker_unavailable': 'The R1 speaker is unavailable. Check its interface, SDK environment and connection.',
    'speaker_backlog': 'R1 playback fell behind. The session stopped to avoid delayed speech.',
    'empty_transcript': 'STT returned no speech. Check the microphone input and speak closer, then click Talk again.',
    'transcript_too_long': 'The transcript exceeded 1,000 characters. Try a shorter phrase.',
    'invalid_response': 'STT returned an unexpected response format.',
    'authentication': 'The speech provider rejected the API key. Check the local key file.',
    'model_access': 'The speech model is unavailable for this API key. Check model access.',
    'rate_limit': 'The speech provider is rate-limiting requests or has no available quota. Try again later.',
    'timeout': 'The speech provider timed out. Try again; no automatic retry was made.',
    'connection': 'The speech provider connection failed. Check the internet connection.',
    'provider_error': 'The speech provider could not complete the request. Try again.',
}


class SpeechError(Exception):
    def __init__(self, code):
        self.code = code if code in MESSAGES else 'provider_error'
        self.public_text = MESSAGES[self.code]
        self.recoverable = self.code in ('empty_transcript', 'transcript_too_long')
        super().__init__(self.public_text)


def validate_transcript(text):
    if not isinstance(text, str):
        raise SpeechError('invalid_response')
    text = text.strip()
    if not text:
        raise SpeechError('empty_transcript')
    if len(text) > 1000:
        raise SpeechError('transcript_too_long')
    return text


def provider_error(error):
    import openai
    if isinstance(error, SpeechError): return error
    if isinstance(error, openai.APITimeoutError) or isinstance(error, TimeoutError): return SpeechError('timeout')
    if isinstance(error, openai.APIConnectionError): return SpeechError('connection')
    status = getattr(error, 'status_code', None)
    code = {401: 'authentication', 403: 'model_access', 404: 'model_access', 429: 'rate_limit'}.get(status, 'provider_error')
    return SpeechError(code)
