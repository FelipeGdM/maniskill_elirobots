FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN apt update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    htop \
    libegl1 \
    libxext6 \
    libjpeg-dev \
    libpng-dev  \
    libvulkan1 \
    rsync \
    tmux \
    unzip \
    vim \
    vulkan-tools \
    wget \
    xvfb \
    fish \
    mesa-vulkan-drivers \
    ffmpeg \
    libsm6 \
    libxext6 \
    libvulkan-dev \
    wget \
    libglvnd-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/usr/bin/fish", "-c"]

RUN chsh -s /usr/bin/fish root

RUN pip config set global.break-system-packages true

RUN pip install poetry

ADD pyproject.toml .

ADD poetry.lock .

RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR

# download physx GPU binary via sapien
RUN python -c "exec('import sapien.physx as physx;\ntry:\n  physx.enable_gpu()\nexcept:\n  pass;')"

RUN mkdir -p /usr/share/vulkan/icd.d

RUN wget -q https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/nvidia_icd.json
RUN mv nvidia_icd.json /usr/share/vulkan/icd.d

RUN wget -q https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/10_nvidia.json
RUN mv 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN wget -q https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/nvidia_layers.json
RUN mv nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json
