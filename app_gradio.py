"""Gradio front-end for MoodSync, for deployment on a free Hugging Face Space.

The Flask app (app.py) remains the reference implementation with the custom UI;
this module is a thin presentation swap for hosting. Every stage of the actual
pipeline -- LLM emotion extraction, V/A region interpolation, prompt building,
and MusicGen inference -- is imported unchanged from the same modules the Flask
app uses, so the hosted demo exercises identical code paths.

Gradio's queue handles the multi-minute CPU inference directly, which removes
the need for app.py's background-thread-plus-polling arrangement.
"""

import os
import tempfile
import traceback
from pathlib import Path

import gradio as gr

# --- gradio_client schema shim -------------------------------------------
# gradio_client's JSON-schema walker assumes every schema node is a dict, but
# JSON Schema allows `additionalProperties: true` (a bare boolean). Gradio's
# own Chatbot schema contains one, so building the /info endpoint raises
#   TypeError: argument of type 'bool' is not iterable
# at gradio_client/utils.py:get_type. HF Spaces probes /info at startup, so the
# repeated ASGI error takes the Space down. show_api=False does not help --
# the route stays registered. Teach both helpers to treat a boolean node as
# "anything", which is what `true` means in JSON Schema.
import gradio_client.utils as _gcu

_orig_get_type = _gcu.get_type
_orig_to_python_type = _gcu._json_schema_to_python_type


def _get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)


def _json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_to_python_type(schema, defs)


_gcu.get_type = _get_type
_gcu._json_schema_to_python_type = _json_schema_to_python_type
# -------------------------------------------------------------------------
from dotenv import load_dotenv

load_dotenv()

from llm_client import get_emotion_and_reply, NEUTRAL_FALLBACK
from music.va_mapper import map_va_to_descriptor
from music.prompt_builder import build_prompt

# Free Space CPU is ~2 vCPU, so default to a shorter clip than the local app.
CLIP_DURATION_SECONDS = float(os.environ.get("CLIP_DURATION_SECONDS", 10))
AUDIO_DIR = Path(tempfile.gettempdir()) / "moodsync_audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def respond(message, history):
    """Streams: user echo -> reply + V/A reading -> generated clip."""
    message = (message or "").strip()
    if not message:
        yield history, None, ""
        return

    history = history + [{"role": "user", "content": message}]
    yield history, None, "Reading how you're feeling…"

    try:
        result = get_emotion_and_reply(message)
    except Exception:
        traceback.print_exc()
        result = dict(NEUTRAL_FALLBACK)

    valence, arousal, reply = result["valence"], result["arousal"], result["reply"]
    descriptor = map_va_to_descriptor(valence, arousal)
    prompt = build_prompt(descriptor)

    region = f"{descriptor['primary_region']}/{descriptor['secondary_region']}"
    status = (f"valence {valence:+.2f} · arousal {arousal:+.2f} · region {region}\n\n"
              f"Prompt: {prompt}\n\nGenerating {CLIP_DURATION_SECONDS:.0f}s of audio "
              f"(a few minutes on free CPU)…")
    history = history + [{"role": "assistant", "content": reply}]
    yield history, None, status

    try:
        from music.generator import generate_music
        out = AUDIO_DIR / f"clip_{abs(hash((message, prompt)))}.wav"
        generate_music(prompt, CLIP_DURATION_SECONDS, out)
        yield history, str(out), status.replace("Generating", "Generated").replace(
            "(a few minutes on free CPU)…", "— done.")
    except Exception:
        traceback.print_exc()
        yield history, None, status + "\n\nGeneration failed; see logs."


with gr.Blocks(title="MoodSync", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎵 MoodSync\n"
        "Describe how you're feeling. An LLM reads your message into a "
        "**valence/arousal** coordinate (Russell's Circumplex Model), that point is "
        "mapped onto a bank of ten hand-authored emotion regions, and the resulting "
        "descriptor prompts **MusicGen** to render an original instrumental clip.\n\n"
        "*Free CPU hardware — generation takes a few minutes per clip.*"
    )
    # Gradio 5 defaults to the legacy tuples format; respond() yields dicts.
    chat = gr.Chatbot(type="messages", height=340, label="Chat")
    status = gr.Markdown("")
    audio = gr.Audio(label="Generated clip", type="filepath", interactive=False)
    box = gr.Textbox(placeholder="e.g. I've had a long, draining day…",
                     label="How are you feeling?", lines=2)
    with gr.Row():
        send = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear")

    gr.Examples(
        ["I've had a terrible day, I'm exhausted",
         "I just got the job!! I can't believe it",
         "nothing really happening, just a normal tuesday",
         "my flight got cancelled and I'm stuck at the airport"],
        inputs=box,
    )

    for trigger in (send.click, box.submit):
        trigger(respond, [box, chat], [chat, audio, status]).then(
            lambda: "", None, box)
    clear.click(lambda: ([], None, ""), None, [chat, audio, status])

if __name__ == "__main__":
    # ssr_mode=False: Gradio 5 serves an experimental SSR frontend through a
    # Node proxy. On this Space that proxy exits at startup ("Stopping
    # Node.js server..."), taking the process with it. Serving the Python
    # app directly is stable. show_api is left at its default now that the
    # /info schema builds correctly on Gradio 5.50.
    demo.queue(max_size=8).launch(ssr_mode=False)
