ARG PYTHON_VERSION_HOST

FROM python:${PYTHON_VERSION_HOST}

# 他のARGはFROMより後でないと受け取ってくれないらしい
ARG USER_HOST
ARG UID_HOST
ARG GID_HOST
ARG APPNAME
ARG SHARED_WORKSPACE_HOST
ARG SHARED_WORKSPACE_CONTAINER
ARG DOCKER_TOOLS_PYTHON_HOST
ARG DOCKER_TOOLS_ROOT="/opt/docker_tools"
ARG PYTHON_VERSION_HOST

## ${PYTHON_ROOT}にinstall_requirements.bash(とuser_requirements.txt)をコピー
COPY ./${DOCKER_TOOLS_PYTHON_HOST} ${DOCKER_TOOLS_ROOT}/python

RUN apt-get update \
    && groupadd -g ${GID_HOST} ${USER_HOST} \
    && useradd -m -s /bin/bash -u ${UID_HOST} -g ${GID_HOST} ${USER_HOST} \
    # ユーザ作成後にユーザ環境変数を登録
    && echo "export PIPENV_VENV_IN_PROJECT=1" >> /home/$(id -un ${UID_HOST})/.bashrc

# USER ${UID_HOST} # これやらないでrootでインストールとかしても，root以外でそのインストールした諸々のCLIとか使えるっぽいので良いか…？

ENV PIPENV_VENV_IN_PROJECT 1

RUN apt-get update \
    && apt-get install -y \
        vim \
        wget \
        curl \
        git \
    && python${PYTHON_VERSION_HOST} -V \
    && bash ${DOCKER_TOOLS_ROOT}/python/install_requirements.bash -v ${PYTHON_VERSION_HOST} -r ${DOCKER_TOOLS_ROOT}/python/user_requirements.txt \
    # 以下，デフォルトで入っているべきライブラリをインストール
    && python${PYTHON_VERSION_HOST} -m pip install \
        pipdeptree \
        pipenv \
    # 以下，Pythonライブラリに必要なDebian系パッケージをインストール
    && apt-get install -y \
        ## opencv-python
        libgl1-mesa-dev

WORKDIR ${SHARED_WORKSPACE_CONTAINER}
