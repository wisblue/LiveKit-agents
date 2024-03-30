## Installation
To install LiveKit plugin abstract classes:
```bash
pip install livekit-agents
```


## Running
- Starting whisper_streaming stt service
    - install faster-whiper and download models
    ref: https://github.com/SYSTRAN/faster-whisper
    ```bash
    HTTPS_PROXY=http://192.168.1.11:8123 python whisper_online_server.py --model large-v2
    ```

    - start whisper_streaming stt server:
    ```bash
    $ python whisper_online_server.py --model large-v3
    # or using large-v2 bu default:
    $ python whisper_online_server.py
    # Outputs:
    # Loading Whisper large-v3 model for auto... done. It took 5.85 seconds.
    # Whisper is not warmed up
    # whisper-server-INFO: INFO: Listening on('localhost', 43007)
    ```


- install LiveKit agent
```bash
cd /home/dennis/github/dnb/Eleanor/LiveKit-agents/livekit-agents

pip install .
```

