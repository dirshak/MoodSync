FROM python:3.11-slim

# HF Spaces runs containers as UID 1000; create a matching user so the
# HF cache and SQLite/audio writes land somewhere writable.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PORT=7860 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

USER user

# Bake musicgen-small (~1.5GB) into the image at build time so the first
# end-user request doesn't pay a multi-minute download on top of inference.
RUN python -c "from transformers import AutoProcessor, MusicgenForConditionalGeneration; \
AutoProcessor.from_pretrained('facebook/musicgen-small'); \
MusicgenForConditionalGeneration.from_pretrained('facebook/musicgen-small')"

COPY --chown=user . .

EXPOSE 7860

# 1 worker: run_generation() uses an in-process background thread and a
# module-level model singleton, so extra workers would each load their own
# copy of the model and lose track of each other's jobs. Threads handle the
# concurrent polling requests. Timeout is generous for CPU inference.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "8", \
     "--timeout", "600", "app:app"]
