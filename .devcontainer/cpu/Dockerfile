FROM python:3.12-trixie

ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y \
    git \
    fish \
    libwayland-client0 \
    libwayland-server0 \
    libwayland-egl1 \
    mesa-utils \
    wayland-protocols \
    vulkan-tools \
    libvulkan1 \
    mesa-vulkan-drivers

SHELL ["/usr/bin/fish", "-c"]

RUN pip install poetry

ADD pyproject.toml .

ADD poetry.lock .

RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR

