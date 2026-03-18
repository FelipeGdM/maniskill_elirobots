# FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.13_pytorch_release_2.10.0
# FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.13_pytorch_release_2.10.0
# FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.10.0
FROM rocm/dev-ubuntu-24.04:7.1.1

ENV UV_PYTHON=/usr/bin/python3 \
    UV_PROJECT_ENVIRONMENT=/usr \
    UV_BREAK_SYSTEM_PACKAGES=true \
    UV_LINK_MODE=symlink

RUN apt update && apt install -y git fish python-is-python3

SHELL ["/usr/bin/fish", "-c"]

RUN chsh -s /usr/bin/fish root

RUN pip config set global.break-system-packages true

RUN pip install uv 

ADD pyproject.toml .

ADD uv.lock .

RUN uv sync --no-install-project --extra rocm71 --inexact

# RUN uv cache prune --ci

# download physx GPU binary via sapien
RUN python3 -c "exec('import sapien.physx as physx;\ntry:\n  physx.enable_gpu()\nexcept:\n  pass;')"
