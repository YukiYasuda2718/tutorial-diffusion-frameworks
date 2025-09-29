FROM --platform=linux/amd64 nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
# the platform specification (linux/amd64) is required for PyTorch

ENV LD_LIBRARY_PATH=/usr/local/lib/ \
    POETRY_VERSION=1.8.5 \
    POETRY_HOME=/opt/poetry

ARG DEBIAN_FRONTEND=noninteractive

RUN set -eux \
    && apt update -y \
    && apt install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt update -y \
    && apt install -y --no-install-recommends \
    screen \
    openssh-client \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    python3.11 \
    python3.11-dev \
    python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && apt autoremove -y \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY poetry.lock pyproject.toml /workspace/docker/

RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=${POETRY_VERSION} python3 - \
    && cd /usr/local/bin \
    && ln -s /opt/poetry/bin/poetry \
    && poetry config virtualenvs.create false \
    && cd /workspace/docker \
    && poetry install --no-root \
    && rm -rf /root/.cache/pip /root/.cache/pypoetry
