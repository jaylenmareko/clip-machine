# Setup (one-time)

Run on a machine with **open network egress** and **~3 GB+ free RAM** (not a
blocked institutional network, not a memory-starved box).

1. `git clone https://github.com/jaylenmareko/clip-machine.git`
2. `pip install -r requirements.txt`; put `ffmpeg` and `yt-dlp` on PATH.
3. **Whisper model** (avoids a HuggingFace download): reassemble the 3 parts
   you were sent — `cat faster-whisper-tiny.en.zip.part0{0,1,2} > m.zip`, unzip,
   and put the `models--Systran--faster-whisper-tiny.en` folder in
   `~/.cache/huggingface/hub/`.
4. **Credentials** — placed in `~/.config/clip-machine/` by the channel owner,
   out of band. Four things, which the owner supplies (never in this repo):
   - `.env` with `ANTHROPIC_API_KEY` (scoring + titling) and `YOUTUBE_API_KEY`
     (discovery)
   - `youtube_client_secret.json` and `youtube_token.json` (upload OAuth)

Verify: run one VOD end to end (discover → runner → review → post one clip
scheduled private). If transcription runs with no download and a clip posts,
setup is good.
