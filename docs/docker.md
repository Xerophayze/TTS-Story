# Docker deployment

There is a docker image that can be used for faster/easier deployment. Currently the Docker image has support only for Nvidia GPUs or using CPU/external interfaces

## Building locally

From the repository root:

```sh
docker build --platform linux/amd64 --target cpu -t tts-story:local .
TTS_STORY_IMAGE=tts-story:local docker compose up -d
docker compose ps
```

Open <http://127.0.0.1:5000>. Change `TTS_STORY_PORT` if the local port
is occupied. Use the correct IP/Host and port if you have changed this

## Published images

Each image has 2 variants. A CUDA and a non-CUDA variant. You can also select specific variants for pinned versions

| Target | Floating tag | Explicit version tag |
| --- | --- | --- |
| CPU / cloud / remote | `latest` | `0.2.0` |
| NVIDIA CUDA | `latest-cuda` | `0.2.0-cuda` |

E.g `ghcr.io/xerophayze/tts-story:latest-cuda`.

The CUDA target installs CUDA 12.8 PyTorch wheels in the application
environment. A CUDA deployment also requires a working NVIDIA driver and
NVIDIA Container Toolkit on the host, plus uncommenting the GPU reservation
in `compose.yaml`. The CPU image does not require GPU passthrough.

Pull by a specific image version rather than `latest`:

```sh
TTS_STORY_IMAGE=ghcr.io/xerophayze/tts-story:0.2.0 docker compose pull
TTS_STORY_IMAGE=ghcr.io/xerophayze/tts-story:0.2.0 docker compose up -d
```

## Persistence and ownership

Compose defines five named volumes:

| Volume | Container path | Purpose |
| --- | --- | --- |
| `tts-data` | `/app/data` | Configuration, saved projects, jobs DB and metadata, voice prompts |
| `tts-audio` | `/app/static/audio` | Generated audio and chapter exports |
| `tts-samples` | `/app/static/samples` | Voice sample files |
| `tts-models` | `/app/models` | Application-managed models, where supported |
| `tts-cache` | `/home/tts/.cache` | Hugging Face / XDG model caches |

The container uses UID/GID **1000:1000**. Docker named volumes are
recommended. For host bind mounts, create directories owned by that UID/GID
before starting, and do not make the application directories world-writable.

This can usually be done in Linux via running the command `chown -R 1000:1000 <bind/mount/on/host>`

The container sets `TTS_STORY_CONFIG_FILE=/app/data/config.json`.

## Security and operations
- No externally reachable HTTP port is exposed by default. For remote use,
  put an authenticated HTTPS reverse proxy in front, restrict ingress and
  review your forwarding-header policy.
- A health check requests the local web home page. It detects HTTP startup
  failures, **not** model readiness, queue health or provider availability.
- The service uses a production WSGI server with a single process and eight
  threads. Never run additional WSGI workers or replicas against its
  in-process dispatcher.
