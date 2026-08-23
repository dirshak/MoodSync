# MoodSync

A local chat app that turns how you're feeling into a short (30s) generated piece of
music. You describe your mood in a message; an LLM extracts a valence/arousal (V/A)
reading and writes a one-line reply; the V/A point is mapped to a music descriptor
and used to prompt MusicGen, which renders a clip that's embedded right in the chat.

Chats and messages persist in SQLite, and generated clips persist on disk — reopening
an old chat replays the same audio, it's never regenerated.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open the URL it prints (`http://127.0.0.1:5000`).

Copy `.env.example` to `.env` and fill in at least one LLM provider:

```
GROQ_API_KEY=your-key-here
```

Get a free key at https://console.groq.com/keys. If `GROQ_API_KEY` is unset (or a
Groq call fails), the app automatically falls back to a local [Ollama](https://ollama.com)
server (`OLLAMA_BASE_URL`, default `http://localhost:11434`) — install Ollama and run
`ollama pull llama3.1` if you'd rather run the chat/emotion step fully locally. If
neither provider is reachable, the app doesn't crash — it falls back to a neutral
mood reading and a generic reply so music still generates.

## A note on the music backend

The original plan for this project called for `facebookresearch/audiocraft`
(`MusicGen.get_pretrained(...)`). That package hard-pins `torch==2.1.0`, which has no
wheels for Python 3.12 — the only Python available on the machine this was built on —
so it cannot be installed here. Instead this app uses Hugging Face `transformers`'
`MusicgenForConditionalGeneration`, which is the same `facebook/musicgen-small`
pretrained weights, officially maintained, and installs cleanly on modern Python/torch.
Output quality and behavior are equivalent; only the wrapper library differs.

If you're on Python 3.9–3.11 and want to use the original `audiocraft` package instead,
swap the `torch`/`transformers` lines in `requirements.txt` for:

```
torch==2.1.0
audiocraft
```

and replace `music/generator.py`'s model loading with
`audiocraft.models.MusicGen.get_pretrained('facebook/musicgen-small')`.

## First run

The first time you send a message, `facebook/musicgen-small` (~1.5GB) downloads via
Hugging Face and is cached (default `~/.cache/huggingface`, override with `HF_HOME` in
`.env`). Loading the model and generating a 30-second clip on CPU can take a couple of
minutes — the UI shows an animated "Composing your track…" indicator while it works,
and you can keep browsing/starting other chats in the meantime (generation runs in a
background thread and is serialized so it doesn't fight itself for CPU).

`ffmpeg` is **not required** for this app — audio is written directly as WAV via
`scipy.io.wavfile`, no container/codec transcoding involved.

## Where things live

- `emotionground.db` — SQLite database (chats + messages), created on first run.
- `static/audio/<chat_id>/<message_id>.wav` — generated clips, served directly by
  Flask so `<audio src="...">` just works on reload. Never regenerated once written.
- `config/va_regions.yaml` — the named valence/arousal regions (relaxed, happy, sad,
  angry, anxious, hopeful, calm, excited, bored, exhausted) and their music
  descriptors (tempo, instruments, dynamics, mood words) that V/A points are mapped
  (and interpolated between the two nearest) onto.

## Architecture

```
user message
  -> LLM (Groq, falls back to Ollama, falls back to neutral) -> {valence, arousal, reply}
  -> nearest-2-region interpolation (config/va_regions.yaml) -> music descriptor
  -> prompt_builder.py -> natural-language MusicGen prompt (lightly randomized)
  -> MusicGen (facebook/musicgen-small, pretrained, frozen) -> 30s wav
  -> saved to static/audio/<chat_id>/<message_id>.wav, path stored in SQLite
```

The reply text and V/A are returned to the browser immediately (LLM call is fast);
music generation happens in a background thread, and the frontend polls
`GET /api/messages/<id>` every ~2.5s until the clip is ready.

## API

- `GET /api/chats` — list chats (id, title, created_at, preview)
- `POST /api/chats` — create a new chat
- `GET /api/chats/<id>` — full chat with all messages
- `DELETE /api/chats/<id>` — delete a chat and its audio files
- `POST /api/chats/<id>/messages` — send a user message, returns the user message
  plus an assistant message (status `generating` until the clip is done)
- `GET /api/messages/<id>` — poll a single message's current status/audio_path
