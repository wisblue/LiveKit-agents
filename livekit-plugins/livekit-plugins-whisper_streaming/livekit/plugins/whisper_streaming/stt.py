import sys
sys.path.insert(1, '/home/dennis/github/dnb/Eleanor/whisper_streaming')
from whisper_online_ex import *

import asyncio
import dataclasses
import io
import json
import logging
import wave
from dataclasses import dataclass
from typing import Optional, Union, List

from livekit import rtc
from livekit.agents import stt
from livekit.agents.utils import AudioBuffer, merge_frames

from . models import WhisperStreamingLanguages, WhisperStreamingModels

from faster_whisper import WhisperModel

import math
import numpy as np
import ctypes

#STREAM_KEEPALIVE_MSG: str = json.dumps({"type": "KeepAlive"})
STREAM_CLOSE_MSG: str = json.dumps({"type": "CloseStream"})
END_OF_FRAME = rtc.AudioFrame(data=b"", sample_rate=0, samples_per_channel=0, num_channels=0)

# internal
@dataclass
class STTOptions:
    language: Optional[Union[WhisperStreamingLanguages, str]]
    detect_language: bool
    interim_results: bool
    punctuate: bool
    model: WhisperStreamingModels
    endpointing: Optional[str]
    vad: bool
    min_chunk_size: float
    device: str # "cuda" or "cpu"

class AudioFrameEx(rtc.AudioFrame):
    def __init__(
        self,
        audio_frame: rtc.AudioFrame = None,
        data: Union[bytes, bytearray, memoryview]=b"",
        sample_rate: int=16000, # 16000 Hz (used in some audio codecs);44100 Hz (CD audio quality)
        num_channels: int=1,
        samples_per_channel: int=0,
        stream_close: bool=False, # last frame in queue
    ) -> None:
        """
        initialize with rtc.AudioFrame or parameter of AudioFrame.
        added additional parameter stream_close to indicate the last frame in queue.
        """
        if audio_frame is None:
            data = audio_frame.data, 
            sample_rate = audio_frame.sample_rate, 
            num_channels = audio_frame.num_channels, 
            samples_per_channel = audio_frame.samples_per_channel
        elif samples_per_channel == 0: 
            samples_per_channel = len(data) // (num_channels * ctypes.sizeof(ctypes.c_int16))
        super(AudioFrameEx, self).__init__(data, sample_rate, num_channels, samples_per_channel)
        self.stream_close = stream_close
        
    @property
    def duration(self) -> float:
        """
        to audio duration in seconds
        """
        return self.samples_per_channel / self.sample_rate

    @staticmethod
    def to_numpy(frame: rtc.AudioFrame) -> np.ndarray:
        """
        This function is to convert PCM16 audio frame to numpy array.
        """
        # Reshape the bytes array into a NumPy array
        data_np = np.frombuffer(frame.data, dtype=np.int16).reshape((-1, frame.num_channels))
        # Normalize the data to float32 in the range [-1, 1]
        data_np = data_np.astype(np.float32) / np.iinfo(np.int16).max
        return data_np

class STT(stt.STT):
    def __init__(
        self,
        *,
        language: WhisperStreamingLanguages = "en",
        detect_language: bool = True,
        interim_results: bool = True,
        model: WhisperStreamingModels = "large-v2",
        min_silence_duration: int = 10,
        vad: bool = True,
        min_chunk_size:float=1.0,
        device="cuda"
    ) -> None:
        super().__init__(streaming_supported=True)

        self._config = STTOptions(
            language=language,
            detect_language=detect_language,
            interim_results=interim_results,
            punctuate=False,
            model=model,
            endpointing=str(min_silence_duration),
            vad=vad,
            min_chunk_size=min_chunk_size,
            device="cuda",
        )

        # Run on GPU with FP16
        print("Loading model ...", end="")
        self.model = WhisperModel(self._config.model, 
                                  device=self._config.device, 
                                  compute_type="float16",
                                  local_files_only=True)
        print("done")

    def _sanitize_options(
        self,
        *,
        language: Optional[str] = None,
    ) -> STTOptions:
        config = dataclasses.replace(self._config)

        if config.detect_language:
            config.language = None

        elif isinstance(language, list):
            logging.warning("whisper_streaming only supports one language at a time")
            config.language = config.language[0]  # type: ignore
        else:
            config.language = language or config.language

        return config

    async def recognize(
        self,
        *,
        buffer: AudioBuffer,
        language: Optional[Union[WhisperStreamingLanguages, str]] = None,
    ) -> stt.SpeechEvent:
        # whisper_streaming requires WAV, so we write our PCM into a wav buffer
        buffer = merge_frames(buffer)
        io_buffer = io.BytesIO()
        # open as write and read mode
        with wave.open(io_buffer, "wb") as wav:
            wav.setnchannels(buffer.num_channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(buffer.sample_rate)
            wav.writeframes(buffer.data)

        # do not forget to move the cursor to begin of file.
        io_buffer.seek(0)
        segments, info = self.model.transcribe(
            io_buffer,
            vad_filter=self._config.vad,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        return stt.SpeechEvent(
                is_final=True,
                end_of_speech=True,
                alternatives=[
                    stt.SpeechData(
                        language=info.language,
                        start_time=seg.start,
                        end_time=seg.end,
                        confidence=math.exp(seg.avg_logprob),
                        text=seg.text.strip() or "",
                    )
                    for seg in segments
                ],
    )

    def stream(
        self,
        *,
        language: Optional[Union[WhisperStreamingLanguages, str]] = None,
    ) -> "SpeechStream":
        config = self._sanitize_options(language=language)
        config.vad = False
        return SpeechStream(
            config,
            self.model
        )


class SpeechStream(stt.SpeechStream):
    """
    A stream of speech events. 
    This is an asynchronous iterable that yields SpeechEvent objects.
    """

    def __init__(
        self,
        config: STTOptions,
        model: WhisperModel=None,
        sample_rate: int = 16000,
        num_channels: int = 1,
    ) -> None:
        super().__init__()
        self._config = config
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        if model is None:
            self.asr = FasterWhisperASREx(self._config.language, self._config.model)  # loads and wraps Whisper model
        else:
            self.asr = FasterWhisperASREx(self._config.language, model=model)
        if self._config.vad:
            self.asr.use_vad()
        self.online = OnlineASRProcessorEx(self.asr)
        self._queue = asyncio.Queue()
        self._event_queue = asyncio.Queue[stt.SpeechEvent]()
        self._closed = False
        self._main_task = asyncio.create_task(self._run())

        def log_exception(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logging.error(f"whisper_streaming task failed: {task.exception()}")

        self._main_task.add_done_callback(log_exception)

        self._frameBuffer = []

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """
        Pushes an audio frame to the stream. 
        This method is called by the StreamAdapter.
        The frame is 10ms per second, so 100 frames is 1 second.
        """
        if self._closed:
            raise ValueError("cannot push frame to closed stream")

        self._queue.put_nowait(
            frame.remix_and_resample(self._sample_rate, self._num_channels)
        )

    async def flush(self) -> None:
        """
        Flushes the stream, indicating that no more audio frames 
        will be pushed to the stream.
        When the count of unfinished tasks drops to zero, _queue.join() unblocks.
        """
        await self._queue.put(STREAM_CLOSE_MSG)
        await self._queue.join()

    async def aclose(self) -> None:
        """
        handles asynchronous stream closure.
        """
        #await self._queue.put(STREAM_CLOSE_MSG)
        await self._main_task

    async def _run(self) -> None:
        """
        Internal function.
        Try to connect to whisper_streaming with exponential backoff and forward frames
        """
        # this 
        while True:
            try:
                # create a task to listen to the websocket and parse the results.
                listen_task = asyncio.create_task(self._listen_loop(self.online))
                # break out of the retry loop if we are done
                if await self._send_loop(self.online):
                    await asyncio.wait_for(listen_task, timeout=5)
                    break
            except Exception as e:
                print(e)
                break
                pass
 
        self._closed = True

    async def _send_loop(self, online: OnlineASRProcessor) -> bool:
        """
        International function.
        this is to send audio frames to the queue for further processing.
        Outputs:
        True: when is received STREAM_CLOSE_MSG, returns after call online.insert_audio_chunk(audio_chunk) 
        False: when is not received STREAM_CLOSE_MSG
        """
        while not self._closed:
            # get AudioFrame from queue
            frame = await self._queue.get()
            # fire and forget, we don't care if we miss frames in the error case
            # we will just keep sending the next frame
            self._queue.task_done()
            
            if isinstance(frame, rtc.AudioFrame):
                self._frameBuffer.append(frame)
                #print(f'len(self._frameBuffer={len(self._frameBuffer)}')
                return False
            else:
                if frame == STREAM_CLOSE_MSG:
                    self._frameBuffer.append(END_OF_FRAME)
                    return True
        return False
            
    async def _transcribe(self, online: OnlineASRProcessor) -> tuple:
        """
        International function.
        This is to transcribe the audio frame.
        """
        # test if buffer accumalated over 1 second
        print(f'len(self._frameBuffer={len(self._frameBuffer)}')
        stream_close = False
        if len(self._frameBuffer[-1].data) == 0:
            merged = merge_frames(self._frameBuffer[:-1])
            stream_close = True
        else:
            merged = merge_frames(self._frameBuffer)
        # convert to numpy array
        audio_chunk = AudioFrameEx.to_numpy(merged)
        self._frameBuffer=[]

        online.insert_audio_chunk(audio_chunk)
        b,e,t,c = online.process_iter(stream_close)
        return (b,e,t,c)


    async def _listen_loop(self, online: OnlineASRProcessor) -> None:
        """
        International function.
        This is to listen to the websocket and parse the results.
        1. get data from ws message
        2. convert data to stt_event
        3. put stt_event to event_queue
        """
        while not self._closed:
            await asyncio.sleep(0.01)
            if len(self._frameBuffer) >= 100 or \
                (len(self._frameBuffer)>0 and len(self._frameBuffer[-1].data)==0):
                # begin, end, text, complete
                b,e,t,c = await self._transcribe(self.online)
                if c:
                    online.finish()
                if not b is None:
                    stt_event = live_transcription_to_speech_event(
                        self._config.language, (b,e,t,c)
                    )
                    await self._event_queue.put(stt_event)
                    continue
            # except Exception as ecpt:
            #     logging.error("Error handling message %s: %s", (b,e,t,c), ecpt)
            #     continue

    async def __anext__(self) -> stt.SpeechEvent:
        """
        When you use an async for loop on an object of this class, 
        the __aiter__ method is called first to get the iterator. 
        Then, for each iteration of the loop, 
        the __anext__ method is called on the returned iterator object.
        """

        if self._closed and self._event_queue.empty():
            raise StopAsyncIteration

        return await self._event_queue.get()
    
    def __aiter__(self) -> "SpeechStream":
        """
        The __aiter__ method is a special method in Python 
        that defines an iterator for asynchronous iterables.
        """
        return self



def live_transcription_to_speech_event(
    language: Optional[str],
    event: tuple,
) -> stt.SpeechEvent:
    """
    This function is to convert live transcription to speech event.
    """
    return stt.SpeechEvent(
        is_final=event[3],  # could be None?
        end_of_speech=event[3],
        alternatives=[
            stt.SpeechData(
                language=language or "",
                start_time=event[0] or 0,
                end_time=event[1] or 0,
                confidence=None,
                text=event[2],
            )
        ],
    )


# def prerecorded_transcription_to_speech_event(
#     language: Optional[str],
#     event: dict,
# ) -> stt.SpeechEvent:
#     try:
#         dg_alts = event["results"]["channels"][0]["alternatives"]
#     except KeyError:
#         raise ValueError("no alternatives in response")

#     return stt.SpeechEvent(
#         is_final=True,
#         end_of_speech=True,
#         alternatives=[
#             stt.SpeechData(
#                 language=language or "",
#                 start_time=(alt["words"][0]["start"] if alt["words"] else 0) or 0,
#                 end_time=(alt["words"][-1]["end"] if alt["words"] else 0) or 0,
#                 confidence=alt["confidence"] or 0,
#                 # not sure why transcript is Optional inside DG SDK ...
#                 text=alt["transcript"] or "",
#             )
#             for alt in dg_alts
#         ],
#     )
