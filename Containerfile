# Container image for running budok-ai end-to-end on Linux without installing
# Xvfb/Mesa/FFmpeg on the host.
#
# Build:
#   podman build -t budok-ai -f Containerfile .
#
# Run (see scripts/run_match_podman.sh for the full wrapper):
#   podman run --rm --network=host \
#     -v /path/to/yomi-game:/games/yomi:ro \
#     -v "$PWD":/budok-ai:Z \
#     -e ANTHROPIC_API_KEY=... \
#     budok-ai \
#     scripts/run_match_linux.sh --daemon-config daemon/config/default_config.json

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_PYTHON=3.12
ENV PATH="/root/.local/bin:${PATH}"

# Install runtime dependencies for YOMI Hustle, Xvfb, and ffmpeg.
RUN apt-get update && apt-get install -y \
    curl ca-certificates \
    xvfb libgl1 mesa-utils unzip ffmpeg zip x11-utils \
    libx11-6 libxcursor1 libxinerama1 libxrandr2 libxi6 \
    libgles2-mesa libegl1-mesa libgl1-mesa-dri \
    libpulse0 libasound2 \
    fonts-dejavu-core fonts-liberation2 \
    python3-minimal \
    && rm -rf /var/lib/apt/lists/*

# Install uv.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /budok-ai

# The repository and game files are mounted at runtime.  We only copy enough
# metadata here so the image can be built standalone; runtime mounts override.
COPY README.md .
COPY daemon/ ./daemon/
COPY scripts/run_match_linux.sh scripts/package_mod.sh scripts/install_mod.sh ./scripts/
COPY prompts/ ./prompts/
COPY schemas/ ./schemas/
COPY mod/ ./mod/
COPY tests/ ./tests/

# Pre-sync dependencies so the first container start is fast.
RUN uv sync --project daemon

CMD ["bash"]
