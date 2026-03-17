FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN apt update && apt install -y git fish

SHELL ["/usr/bin/fish", "-c"]

RUN chsh -s /usr/bin/fish root

RUN pip config set global.break-system-packages true

RUN pip install poetry 

ADD pyproject.toml .

ADD poetry.lock .

RUN source /opt/venv/bin/activate.fish && poetry install --no-root && rm -rf $POETRY_CACHE_DIR

# download physx GPU binary via sapien
# RUN python3 -c "exec('import sapien.physx as physx;\ntry:\n  physx.enable_gpu()\nexcept:\n  pass;')"

