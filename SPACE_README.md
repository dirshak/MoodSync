---
title: MoodSync
emoji: 🎵
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app_gradio.py
pinned: false
---

# MoodSync

Describe how you're feeling; an LLM reads your message into a valence/arousal
coordinate, that coordinate is mapped onto ten hand-authored emotion regions,
and the resulting descriptor prompts MusicGen to render an instrumental clip.

**Speed:** free CPU hardware, so generation takes a few minutes per clip.
`CLIP_DURATION_SECONDS` defaults to 10 here; the local Flask app uses 30.

**Persistence:** clips are written to a temp directory and are not retained
across restarts. The local Flask app persists chats in SQLite.
