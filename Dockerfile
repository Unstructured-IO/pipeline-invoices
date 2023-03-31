# syntax=docker/dockerfile:experimental

FROM centos:centos7.9.2009

# NOTE(crag): NB_USER ARG for mybinder.org compat:
#             https://mybinder.readthedocs.io/en/latest/tutorials/dockerfile.html
ARG NB_USER=notebook-user
ARG NB_UID=1000
ARG PIP_VERSION
ARG PIPELINE_PACKAGE

RUN yum -y update && \
  yum -y install gcc openssl-devel bzip2-devel libffi-devel make git sqlite-devel ffmpeg libsm6 libxext6 python3-opencv mesa-libGL mesa-libGL-devel xz-devel && \
  curl -O https://www.python.org/ftp/python/3.8.15/Python-3.8.15.tgz && tar -xzf Python-3.8.15.tgz && \
  cd Python-3.8.15/ && ./configure --enable-optimizations && make altinstall && \
  cd .. && rm -rf Python-3.8.15* && \
  ln -s /usr/local/bin/python3.8 /usr/local/bin/python3


RUN yum -y install wget libstdc++ autoconf automake libtool autoconf-archive gcc gcc-c++ make libjpeg-devel libpng-devel libtiff-devel zlib-devel
RUN yum group install -y "Development Tools"

# Build leptonica
WORKDIR /opt
RUN curl -O http://www.leptonica.org/source/leptonica-1.82.0.tar.gz
RUN ls -la
RUN tar -zxvf leptonica-1.82.0.tar.gz
WORKDIR ./leptonica-1.82.0
RUN ./configure
RUN make -j2
RUN make install
RUN cd .. && rm leptonica-1.82.0.tar.gz

# Build tesseract
RUN wget https://github.com/tesseract-ocr/tesseract/archive/refs/tags/5.2.0.tar.gz
RUN tar -zxvf 5.2.0.tar.gz
WORKDIR ./tesseract-5.2.0
RUN ./autogen.sh
RUN yum install -y centos-release-scl
RUN yum install -y devtoolset-11
RUN . /opt/rh/devtoolset-11/enable && \
  PKG_CONFIG_PATH=/usr/local/lib/pkgconfig LIBLEPT_HEADERSDIR=/usr/local/include ./configure --with-extra-includes=/usr/local/include --with-extra-libraries=/usr/local/lib && \
  LDFLAGS="-L/usr/local/lib" CFLAGS="-I/usr/local/include" make -j2 && \
  make install && \
  /sbin/ldconfig && \
  cd .. && rm 5.2.0.tar.gz

# Optional: install language packs
RUN wget https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata
RUN mv *.traineddata /usr/local/share/tessdata

# create user with a home directory
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

RUN groupadd --gid ${NB_UID} ${NB_USER}
RUN useradd --uid ${NB_UID}  --gid ${NB_UID} ${NB_USER}

USER ${NB_USER}
WORKDIR ${HOME}
ENV PYTHONPATH="${PYTHONPATH}:${HOME}"
ENV PATH="/home/${NB_USER}/.local/bin:${PATH}"

COPY logger_config.yaml logger_config.yaml
COPY requirements/dev.txt requirements-dev.txt
COPY requirements/base.txt requirements-base.txt
COPY prepline_${PIPELINE_PACKAGE}/ prepline_${PIPELINE_PACKAGE}/
COPY exploration-notebooks exploration-notebooks
COPY pipeline-notebooks pipeline-notebooks

# NOTE(robinson) - Can remove the secret mount once the unstructured repo is public
# NOTE(crag) - Cannot use an ARG in the dst= path (so it seems), hence no ${NB_USER}, ${NB_UID}
RUN python3.8 -m pip install --no-cache -r requirements-base.txt \
  && python3.8 -m pip install --no-cache -r requirements-dev.txt

