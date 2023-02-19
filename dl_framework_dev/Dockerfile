# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ubuntu:18.04

ARG NB_USER="luban"
ARG NB_UID="1000"
ARG NB_PREFIX="/"
ARG NB_PORT=8888
ARG MLFLOW_TRACKING_URI="http://submarine-mlflow-service:5000"

USER root

SHELL ["/bin/bash", "-c"]

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    apt-transport-https \
    build-essential \
    curl \
    wget \
    git \
    bzip2 \
    ca-certificates \
    sudo \
    locales \
    fonts-liberation \
    vim \
    inetutils-ping \
    run-one && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo "$LOG_TAG Set locale" && \
    echo "LC_ALL=en_US.UTF-8" >> /etc/environment && \
    echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && \
    echo "LANG=en_US.UTF-8" > /etc/locale.conf && \
    locale-gen en_US.UTF-8

ENV NB_USER=$NB_USER \
    NB_UID=$NB_UID \
    NB_PREFIX=$NB_PREFIX \
    NB_PORT=$NB_PORT \
    CONDA_DIR=/opt/conda \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8
ENV PATH=$CONDA_DIR/bin:$PATH \
    HOME=/home/$NB_USER

ENV MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI

# Create NB_USER user with UID=1000 and in the 'users' group
RUN useradd -M -s /bin/bash -N -u $NB_UID $NB_USER && \
    chown -R ${NB_USER}:users /usr/local/bin && \
    mkdir -p $HOME && \
    chown -R ${NB_USER}:users ${HOME} && \
    mkdir -p ${CONDA_DIR} && \
    chown -R ${NB_USER}:users ${CONDA_DIR} && \
    chmod g+w /etc/passwd && \
    adduser ${NB_USER} sudo && \
    echo "${NB_USER}:luban" | chpasswd

# Add Tini
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN mv /tini /usr/local/bin/tini && chmod +x /usr/local/bin/tini

# Install conda
USER $NB_UID
ARG PYTHON_VERSION=default
ENV MINICONDA_VERSION=4.8.3 \
    MINICONDA_MD5=751786b92c00b1aeae3f017b781018df \
    CONDA_VERSION=4.8.3

WORKDIR /tmp
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-py37_${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "${MINICONDA_MD5} *Miniconda3-py37_${MINICONDA_VERSION}-Linux-x86_64.sh" | md5sum -c - && \
    /bin/bash Miniconda3-py37_${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-py37_${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "conda ${CONDA_VERSION}" >> $CONDA_DIR/conda-meta/pinned && \
    conda config --system --prepend channels conda-forge && \
    conda config --system --set auto_update_conda false && \
    conda config --system --set show_channel_urls true && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/ && \
    pip config set global.index-url http://mirrors.aliyun.com/pypi/simple && \
    pip config set global.trusted-host mirrors.aliyun.com && \
    if [ ! $PYTHON_VERSION = 'default' ]; then conda install --yes python=$PYTHON_VERSION; fi && \
    conda list python | grep '^python ' | tr -s ' ' | cut -d '.' -f 1,2 | sed 's/$/.*/' >> $CONDA_DIR/conda-meta/pinned && \
    conda clean --all -f -y && \
    rm -rf /home/$NB_USER/.cache/yarn

RUN conda init bash
    
# install FLAML
ENV PIP_SOURCE_TSINGHUA https://pypi.tuna.tsinghua.edu.cn/simple/

# install flaml, ray, ray[air], torch, blendsearch
RUN pip install flaml -i $PIP_SOURCE_TSINGHUA && \
    pip install "flaml[ray,nni,blendsearch]" -i $PIP_SOURCE_TSINGHUA && \
    pip install pyarrow==7.0.0 -i $PIP_SOURCE_TSINGHUA && \
    pip install torch, torchvision -i $PIP_SOURCE_TSINGHUA && \
    pip install -U "ray==2.0" -i $PIP_SOURCE_TSINGHUA && \
    pip install -U "ray[air]" -i $PIP_SOURCE_TSINGHUA && \

# packages for Autoformer, FEDformer
RUN pip install sympy -i $PIP_SOURCE_TSINGHUA

# install fastai, tsai for time series tasks
# RUN pip install tsai, tsai[extras] -i $PIP_SOURCE_TSINGHUA
# upgrade protobuf to match ray version
RUN pip install --upgrade protobuf==3.20.0 -i $PIP_SOURCE_TSINGHUA

# env for pyspark, should install didi-spark-nmg first
RUN pip install py4j && \
    pip pip install findspark

# copy automl_flaml_interface, hadoop_tools
COPY --chown=${NB_USER}:users . ${HOME}

USER root

ENV JAVA_HOME=${HOME}/tools/hadoop_tools/jdk1.8.0_77 \
    HADOOP_HOME=${HOME}/tools/hadoop_tools/didi-hadoop-nmg-2.7.2 \
    PATH=${HOME}/tools/hadoop_tools/didi-hadoop-nmg-2.7.2/bin:$PATH \
    SPARK_HOME=${HOME}/tools/hadoop_tools/didi-spark-nmg-2.4.3 \
    PATH=${HOME}/tools/hadoop_tools/didi-spark-nmg-2.4.3/bin:$PATH \
    ARROW_LIBHDFS_DIR=${HOME}/tools/hadoop_tools/didi-hadoop-nmg-2.7.2/lib/native

RUN echo "export JAVA_HOME=${HOME}/hadoop_tools/jdk1.8.0_77" >> ~/.bashrc && \
    echo "export HADOOP_HOME=${HOME}/hadoop_tools/didi-hadoop-nmg-2.7.2" >> ~/.bashrc && \
    echo "export SPARK_HOME=${HOME}/tools/hadoop_tools/didi-spark-nmg-2.4.3" >> ~/.bashrc && \
    echo "export PATH=${HOME}/hadoop_tools/didi-hadoop-nmg-2.7.2/bin:$PATH" >> ~/.bashrc && \
    echo "export ARROW_LIBHDFS_DIR=${HOME}/hadoop_tools/didi-hadoop-nmg-2.7.2/lib/native" ~/.bashrc && \
    echo "export CLASSPATH=`${HOME}/hadoop_tools/didi-hadoop-nmg-2.7.2/bin/hdfs classpath --glob`" ~/.bashrc && \
    source ~/.bashrc

USER $NB_UID

ENTRYPOINT ["tini", "-g", "--"]
#CMD ["echo","Welcome to automl-flaml env"]

WORKDIR ${HOME}
