# # install correct version of python
# FROM python:3.8 AS packagepython

# ARG GRB_VERSION=9.5.2

# LABEL vendor="Gurobi"
# LABEL version=${GRB_VERSION}

# # Ubuntu Packages
# RUN apt-get update -y && \
#     apt-get install software-properties-common -y && \
#     add-apt-repository -y multiverse && \
#     apt-get update -y && \
#     apt-get upgrade -y && \
#     apt-get install --no-install-recommends -y\
#        ca-certificates \
#        p7zip-full \
#        zip && \
#     apt-get install -y apt-utils nano vim man build-essential wget sudo && \
#     python -m pip install gurobipy==${GRB_VERSION} && \
#     rm -rf /var/lib/apt/lists/*

FROM python:3.8 AS packagepython

ARG GRB_VERSION=9.5.2

LABEL vendor="Gurobi"
LABEL version=${GRB_VERSION}

# update system and certificates
RUN apt-get update \
    && apt-get install --no-install-recommends -y\
       ca-certificates  \
       p7zip-full \
       zip \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git

# Install python3 pip3
RUN apt-get update
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip

# Python packages we use
RUN pip3 install gurobipy==${GRB_VERSION}
RUN pip3 install scipy==1.5.4
RUN pip3 install matplotlib==3.5.3
RUN pip3 install pyyaml==3.13
RUN pip3 install networkx==2.8.8
RUN pip3 install numpy==1.19.5

RUN mkdir /install
WORKDIR /install

WORKDIR /marl-opt
