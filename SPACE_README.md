---
title: MoodSync
emoji: 🎵
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
app_file: app_gradio.py
pinned: false
---

# MoodSync

Describe how you're feeling; an LLM reads your message into a valence/arousal
coordinate, that coordinate is mapped onto ten hand-authored emotion regions,
and the resulting descriptor prompts MusicGen to render an instrumental clip.

**Speed:** runs on ZeroGPU, so a 30-second clip renders in seconds. The first
request after a restart also downloads the model (~1.5GB) and is slower.

**Persistence:** clips are written to a temp directory and are not retained
across restarts. The local Flask app persists chats in SQLite.
