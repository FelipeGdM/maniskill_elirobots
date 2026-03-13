FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN apt update && apt install -y git fish

SHELL ["/usr/bin/fish", "-c"]

RUN pip config set global.break-system-packages true

RUN pip install poetry

ADD pyproject.toml .

RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR

RUN apt install -y mesa-vulkan-drivers libvulkan1 vulkan-tools ffmpeg libsm6 libxext6
# RUN poetry install
