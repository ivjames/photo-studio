FROM python:3.12-slim

WORKDIR /opt/photostudio

# Runtime deps for opencv-python-headless
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY photo_studio.py scan_splitter.py photo_tagger.py ./

# Default data dir — override via PHOTOSTUDIO_HOME; mount a volume here in prod
RUN mkdir -p /data
ENV PHOTOSTUDIO_HOME=/data

# DO App Platform injects PORT; default to 8080 for local docker run
ENV PORT=8080

CMD gunicorn -w 1 --threads 8 --timeout 600 -b "0.0.0.0:${PORT}" photo_studio:app
