FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# RUN apk update && apk add --no-cache git fish
RUN apt update && apt install -y git fish

# apk update && apk add --no-cache git fish

SHELL ["/usr/bin/fish", "-c"]

RUN pip install poetry

ADD pyproject.toml .

RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR

RUN sudo apt-get install -y mesa-vulkan-drivers libvulkan1 vulkan-tools ffmpeg libsm6 libxext6
# RUN poetry install
