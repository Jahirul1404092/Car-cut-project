# Use a docker image as base image
FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu18.04
# Declare some ARGuments
ARG PYTHON_VERSION=3.7
ARG CONDA_VERSION=3
ARG CONDA_PY_VERSION=py37_4.12.0
# Installation of some libraries / RUN some commands on the base image
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# RUN apt-key del 7fa2af80
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-pip python3-dev wget \
    bzip2 libopenblas-dev pbzip2 libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN gcc --version
# INSTALLATION OF CONDA
ENV PATH /opt/conda/bin:$PATH
RUN wget https://repo.anaconda.com/miniconda/Miniconda$CONDA_VERSION-$CONDA_PY_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo “. /opt/conda/etc/profile.d/conda.sh” >> ~/.bashrc && \
    echo “conda activate base” >> ~/.bashrc
ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True
RUN apt-get update
# RUN apt install -y libgl1-mesa-glx

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./
# Create the environment:
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
RUN echo "conda activate carCut" >> ~/.bashrc

# The code to run when container is started:
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["/bin/bash", "--login", "-c","./entrypoint.sh"]

