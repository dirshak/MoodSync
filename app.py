import os
import threading
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

import db
from llm_client import get_emotion_and_reply, NEUTRAL_FALLBACK
from music.va_mapper import map_va_to_descriptor
from music.prompt_builder import build_prompt

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "static" / "audio"
CLIP_DURATION_SECONDS = 30

app = Flask(__name__)
db.init_db()
# Any message still marked "generating" is orphaned from a previous process
# that died mid-generation (crash, restart) -- it can never complete, so
# surface it as an error instead of spinning forever in the UI.
db.execute("UPDATE messages SET status = 'error' WHERE status = 'generating'")


def make_title(text: str) -> str:
    text = " ".join(text.strip().split())
    return (text[:42] + "…") if len(text) > 42 else (text or "New chat")


def run_generation(chat_id: int, message_id: int, valence: float, arousal: float):
    """Runs in a background thread: build the MusicGen prompt from V/A,
    generate the clip, save it to disk, and update the message row."""
    try:
        descriptor = map_va_to_descriptor(valence, arousal)
        prompt = build_prompt(descriptor)
        print(f"[generation] chat={chat_id} msg={message_id} va=({valence:.2f},{arousal:.2f}) "
              f"region={descriptor['primary_region']}/{descriptor['secondary_region']} prompt=\"{prompt}\"")

        out_path = AUDIO_DIR / str(chat_id) / f"{message_id}.wav"

        from music.generator import generate_music
        generate_music(prompt, CLIP_DURATION_SECONDS, out_path)

        rel_path = f"/static/audio/{chat_id}/{message_id}.wav"
        db.update_message(message_id, audio_path=rel_path, status="done")
        print(f"[generation] chat={chat_id} msg={message_id} done -> {rel_path}")
    except Exception:
        traceback.print_exc()
        db.update_message(message_id, status="error")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    # Some browsers request this path directly regardless of <link rel="icon">.
    return app.send_static_file("favicon.ico")


@app.route("/api/chats", methods=["GET"])
def api_list_chats():
    return jsonify(db.list_chats())


@app.route("/api/chats", methods=["POST"])
def api_create_chat():
    chat_id = db.create_chat("New chat")
    return jsonify(db.get_chat(chat_id)), 201


@app.route("/api/chats/<int:chat_id>", methods=["GET"])
def api_get_chat(chat_id):
    chat = db.get_chat(chat_id)
    if not chat:
        return jsonify({"error": "Chat not found"}), 404
    chat["messages"] = db.list_messages(chat_id)
    return jsonify(chat)


@app.route("/api/chats/<int:chat_id>", methods=["DELETE"])
def api_delete_chat(chat_id):
    chat = db.get_chat(chat_id)
    if not chat:
        return jsonify({"error": "Chat not found"}), 404
    db.delete_chat(chat_id)
    chat_audio_dir = AUDIO_DIR / str(chat_id)
    if chat_audio_dir.exists():
        for f in chat_audio_dir.glob("*.wav"):
            f.unlink(missing_ok=True)
        try:
            chat_audio_dir.rmdir()
        except OSError:
            pass
    return jsonify({"ok": True})


@app.route("/api/chats/<int:chat_id>/messages", methods=["POST"])
def api_post_message(chat_id):
    chat = db.get_chat(chat_id)
    if not chat:
        return jsonify({"error": "Chat not found"}), 404

    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Message text is required"}), 400

    existing = db.list_messages(chat_id)
    is_first_message = len(existing) == 0

    user_msg_id = db.create_message(chat_id, "user", text, status="done")

    try:
        result = get_emotion_and_reply(text)
    except Exception:
        traceback.print_exc()
        result = dict(NEUTRAL_FALLBACK)

    valence, arousal, reply = result["valence"], result["arousal"], result["reply"]

    assistant_msg_id = db.create_message(
        chat_id, "assistant", reply, valence=valence, arousal=arousal, status="generating"
    )

    if is_first_message:
        db.update_chat_title(chat_id, make_title(text))

    thread = threading.Thread(
        target=run_generation,
        args=(chat_id, assistant_msg_id, valence, arousal),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "user_message": db.get_message(user_msg_id),
        "assistant_message": db.get_message(assistant_msg_id),
        "chat_title": db.get_chat(chat_id)["title"],
    }), 201


@app.route("/api/messages/<int:message_id>", methods=["GET"])
def api_get_message(message_id):
    msg = db.get_message(message_id)
    if not msg:
        return jsonify({"error": "Message not found"}), 404
    return jsonify(msg)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"MoodSync running at http://127.0.0.1:{port}")
    # use_reloader=False: the reloader restarts the whole process on file changes,
    # which would kill any in-flight background music-generation thread.
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=port, threaded=True)
