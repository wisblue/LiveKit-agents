## Test for whisper_streaming
- Starting whisper_streaming stt service
    - install faster-whiper and download models
    ref: https://github.com/SYSTRAN/faster-whisper
    ```bash
    HTTPS_PROXY=http://192.168.1.11:8123 python whisper_online_server.py --model large-v2
    ```

    - start whisper_streaming stt server:
        - ref: https://github.com/ufal/whisper_streaming
    ```bash
    $ python whisper_online_server.py --model large-v3
    # or using large-v2 bu default:
    $ python whisper_online_server.py
    # Outputs:
    # Loading Whisper large-v3 model for auto... done. It took 5.85 seconds.
    # Whisper is not warmed up
    # whisper-server-INFO: INFO: Listening on('localhost', 43007)
    ```

- install LiveKit Python SDK
    - ref: https://github.com/livekit/python-sdks
        ```bash
        pip install livekit
        ```
    
- install LiveKit agent
    ```bash
    cd LiveKit-agents/

    pip install .
    ```

- install silero
    ```bash
    cd LiveKit-agents/livekit-plugins/livekit-plugins-silero
    pip install .
    ```

- install silero-vad
    ```bash
    git clone https://github.com/wisblue/silero-vad.git
    cd silero-vad
    pip install torchaudio -U
    pip install onnxruntime
    ```

## test STT.recognize():
```bash
cd LiveKit-agents/livekit-plugins/livekit-plugins-whisper_streaming
pip install . -U

cd LiveKit-agents/tests
python test_stt.py
```
