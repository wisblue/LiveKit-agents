# ref: https://github.com/ufal/whisper_streaming?tab=readme-ov-file
import sys
sys.path.insert(1, '/home/dennis/github/dnb/Eleanor/stt/whisper_streaming')

demo_audio_path= 'change-sophie.wav'
# reset to store stderr to different file stream, e.g. open(os.devnull,"w")
logfile = sys.stderr
SAMPLING_RATE = 16000
model = "large-v2"
language = "en"  # source language
# 'Minimum audio chunk size in seconds. 
# It waits up to this time to do processing. 
# If the processing takes shorter time, it waits, 
# otherwise it processes the whole segment that 
# was received by this time.
min_chunk = 1.0 

from whisper_online import *
from functools import lru_cache

@lru_cache
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a

def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg*16000)
    end_s = int(end*16000)
    return audio[beg_s:end_s]

def output_transcript(o, now=None):
    # output format in stdout is like:
    # 4186.3606 0 1720 Takhle to je
    # - the first three words are:
    #    - emission time from beginning of processing, in milliseconds
    #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
    # - the next words: segment transcript
    if now is None:
        now = time.time()-start
    if o[0] is not None:
        print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000,o[1]*1000,o[2]),file=logfile,flush=True)
        print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000,o[1]*1000,o[2]),flush=True)
        print("start=%1.2f end=%1.2f %s" % (o[0],o[1],o[2]),flush=True)
    else:
        print(o,file=logfile,flush=True)

asr = FasterWhisperASR(language, model)  # loads and wraps Whisper model
# set options:
# asr.set_translate_task()  # it will translate from lan into English
asr.use_vad()  # set using VAD

online = OnlineASRProcessor(asr)  # create processing object with default buffer trimming option

# warm-up
# load the audio into the LRU cache before we start the timer
a = load_audio_chunk(demo_audio_path,0,1)
# warm up the ASR, because the very first transcribe takes much more time than the other
asr.transcribe(a)

a = load_audio(demo_audio_path)
duration = len(a)/SAMPLING_RATE
print("Audio duration is: %2.2f seconds" % duration, file=logfile)

# online.insert_audio_chunk(a)
# try:
#     o = online.process_iter()
# except AssertionError:
#     print("assertion error",file=logfile)
#     pass
# else:
#     if o[0] is not None:
#         print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000,o[1]*1000,o[2]),file=logfile,flush=True)
#         print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000,o[1]*1000,o[2]),flush=True)
# now = None
# # at the end of this audio processing
# o = online.finish()
# print(o)  # do something with the last output


# online.init()  # refresh if you're going to re-use the object for the next audio

beg = 0
end = beg + min_chunk

while True:
    a = load_audio_chunk(demo_audio_path, beg,end)
    online.insert_audio_chunk(a)
    try:
        o = online.process_iter()
    except AssertionError:
        print("assertion error",file=logfile)
        pass
    else:
        output_transcript(o, now=end)

    print(f"## last processed {end:.2f}s",file=logfile,flush=True)

    if end >= duration:
        break
    
    beg = end
    
    if end + min_chunk > duration:
        end = duration
    else:
        end += min_chunk
now = duration

o = online.finish()
output_transcript(o, now=end)