import asyncio
import wave
import os
from livekit import rtc, agents
#from livekit.plugins import deepgram, google, openai, silero, whisper_streaming
from livekit.plugins import whisper_streaming, silero
from difflib import SequenceMatcher

import sys # for python version

TEST_AUDIO_FILEPATH = os.path.join(os.path.dirname(__file__), "change-sophie.wav")
TEST_AUDIO_TRANSCRIPT = "the people who are crazy enough to think they can change the world are the ones who do"


def read_wav_file(filename: str) -> rtc.AudioFrame:
    with wave.open(filename, "rb") as file:
        frames = file.getnframes()

        if file.getsampwidth() != 2:
            raise ValueError("Require 16-bit WAV files")

        return rtc.AudioFrame(
            data=file.readframes(frames),
            num_channels=file.getnchannels(),
            samples_per_channel=frames // file.getnchannels(),
            sample_rate=file.getframerate(),
        )


async def test_recognize():
    stts = [
        # deepgram.STT(), google.STT(), openai.STT(),
        whisper_streaming.STT()]
    frame = read_wav_file(TEST_AUDIO_FILEPATH)

    async def recognize(stt: agents.stt.STT):
        event = await stt.recognize(buffer=frame)
        text = event.alternatives[0].text
        print(text)
        print(event)
        assert SequenceMatcher(None, text, TEST_AUDIO_TRANSCRIPT).ratio() > 0.9
        assert event.is_final
        assert event.end_of_speech

    if sys.version_info >= (3,11):
        # for python version >=11
        async with asyncio.TaskGroup() as group:
            for stt in stts:
                group.create_task(recognize(stt))
    else: # for python version 10
        tasks = []
        for stt in stts:
            task = asyncio.create_task(recognize(stt))
            tasks.append(task)
        await asyncio.gather(*tasks)
    
async def test_stream():
    #silero_vad = silero.VAD()
    stts = [
        # deepgram.STT(),
        # google.STT(),
        # agents.stt.StreamAdapter(openai.STT(), 
        # silero_vad.stream()),
        whisper_streaming.STT(),
    ]
    frame = read_wav_file(TEST_AUDIO_FILEPATH)

    # divide data into chunks of 10ms
    chunk_size = frame.sample_rate // 100
    frames = []
    for i in range(0, len(frame.data), chunk_size):
        data = frame.data[i : i + chunk_size]
        frames.append(
            rtc.AudioFrame(
                data=data.tobytes() + b"\0\0" * (chunk_size - len(data)),
                num_channels=frame.num_channels,
                samples_per_channel=chunk_size,
                sample_rate=frame.sample_rate,
            )
        )

    async def stream(stt: agents.stt.STT):
        stream = stt.stream()
        stream._config.vad = False
        for frame in frames:
            stream.push_frame(frame)
            await asyncio.sleep(0.001)

        await stream.flush()
        async for event in stream:
            print(event)
            if event.is_final:
                text = event.alternatives[0].text
                print(text)
                assert SequenceMatcher(None, text, TEST_AUDIO_TRANSCRIPT).ratio() > 0.8
                assert event.end_of_speech
                break

        await stream.aclose()

    if sys.version_info >= (3,11):
        # for python version >=11
        async with asyncio.TaskGroup() as group:
            for stt in stts:
                group.create_task(stream(stt))
    else: # for python version 10
        tasks = []
        for stt in stts:
            task = asyncio.create_task(stream(stt))
            tasks.append(task)
        await asyncio.gather(*tasks)


print("Testing STT:")
asyncio.run(test_recognize())
print()

print("Testing stream:")
asyncio.run(test_stream())