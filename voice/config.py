"""Server-only credentials. Never pass this configuration to an audio client."""
import os
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {
    'OPENAI_API_KEY': ('OPENAI_KEY',),
    'AIC_SDK_LICENSE': ('AIC_KEY',),
    'GEMINI_API_KEY': (),
    'CARTESIA_API_KEY': (),
    'CARTESIA_VOICE_ID': (),
}


def load_keys(path=None):
    """Environment (including aliases) wins over the chosen file, then normalize."""
    source = Path(path) if path is not None else ROOT / '.env'
    if path is not None and not source.is_file():
        raise ValueError('The requested key file does not exist')
    values = dotenv_values(source) if source.is_file() else {}
    for canonical, aliases in ALIASES.items():
        names = (canonical, *aliases)
        value = next((os.environ[n] for n in names if os.environ.get(n)), None)
        value = value or next((values[n] for n in names if values.get(n)), None)
        if value:
            os.environ[canonical] = value
