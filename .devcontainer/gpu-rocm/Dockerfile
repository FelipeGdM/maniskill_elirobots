FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN apt update && apt install -y git fish

SHELL ["/usr/bin/fish", "-c"]

RUN pip install poetry

ADD pyproject.toml .

ADD poetry.lock .

RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR
