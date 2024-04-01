import os
import pathlib

import setuptools
import setuptools.command.build_py


here = pathlib.Path(__file__).parent.resolve()
about = {}
with open(os.path.join(here, "livekit", "plugins", "whisper_streaming", "version.py"), "r") as f:
    exec(f.read(), about)


setuptools.setup(
    name="livekit-plugins-whisper_streaming",
    version=about["__version__"],
    description="Agent Framework plugin for services using whisper_streaming.",
    long_description=(here / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/livekit/agents",
    cmdclass={},
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords=["webrtc", "realtime", "audio", "video", "livekit"],
    license="Apache-2.0",
    packages=setuptools.find_namespace_packages(include=["livekit.*"]),
    python_requires=">=3.10.0",  # deepgram-sdk requires 3.10
    install_requires=[
        "livekit >= 0.9.0",
        "livekit-agents >= 0.3.0",
        "aiohttp >= 3.7.4",
    ],
    package_data={},
    project_urls={
        "Documentation": "https://docs.livekit.io",
        "Website": "https://livekit.io/",
        "Source": "https://github.com/livekit/agents",
    },
)
