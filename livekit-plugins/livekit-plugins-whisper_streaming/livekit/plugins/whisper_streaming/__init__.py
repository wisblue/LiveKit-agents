from .stt import STT, SpeechStream
from .version import __version__

__all__ = [
    "STT",
    "SpeechStream",
    "__version__",
]


from livekit.agents import Plugin


class WhisperStreamingPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__)

    def download_files(self):
        pass


Plugin.register_plugin(WhisperStreamingPlugin())
