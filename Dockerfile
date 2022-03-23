FROM dhealth/pylibs-toolkit:1.2.0-1-cudnn

# install cassandra c++ driver
RUN \
    export DEBIAN_FRONTEND=noninteractive \
    && apt-get update -y -q \
    && apt-get install -y libuv1-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/* 

ARG CD_VERS=2.16.0
RUN \
    wget "https://github.com/datastax/cpp-driver/archive/$CD_VERS.tar.gz" \
    && tar xfz $CD_VERS.tar.gz \
    && cd cpp-driver-$CD_VERS \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j \
    && make install


# Install a few build tools
RUN \
    export DEBIAN_FRONTEND=noninteractive \
    && apt-get update -y -q \
    && apt-get install -y ssh m4 autoconf automake libtool flex pandoc \
    && rm -rf /var/lib/apt/lists/*

# download and compile ucx + gdrcopy
RUN \
    export DEBIAN_FRONTEND=noninteractive \
    && apt-get update -y -q \
    && apt-get install -y libnuma-dev ibverbs-providers perftest strace \
       libibverbs-dev librdmacm-dev binutils-dev gettext rdma-core \
    && rm -rf /var/lib/apt/lists/*

ENV LIBRARY_PATH $LIBRARY_PATH:/usr/local/cuda/lib64/stubs
RUN ldconfig -n /usr/local/cuda/lib64/stub

ARG GDR_VERS=v2.3
RUN \
    git clone https://github.com/NVIDIA/gdrcopy.git \
    && cd gdrcopy \
    && git checkout $GDR_VERS \
    && make prefix=/usr/local/ lib_install \
    && ldconfig

ARG UCX_VERS=1.12.1
RUN \
    wget -nv "https://github.com/openucx/ucx/releases/download/v$UCX_VERS/ucx-$UCX_VERS.tar.gz" \
    && tar xf ucx-$UCX_VERS.tar.gz \
    && cd ucx-$UCX_VERS \
    && mkdir build \
    && cd build \
    && ../configure --prefix=/usr/local --with-cuda=/usr/local/cuda/ --with-gdrcopy=/usr/local \
    && make -j4 \
    && make install \
    && ldconfig


# download and compile openmpi
ARG MPI_VERS=4.1.2
RUN \
    wget -nv "https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-$MPI_VERS.tar.bz2" \
    && tar xf openmpi-$MPI_VERS.tar.bz2 \
    && cd openmpi-$MPI_VERS \
    && mkdir build \
    && cd build \
    && ../configure --prefix=/usr/local --with-ucx=/usr/local --with-cuda=/usr/local/cuda/ \
    && make all install

RUN \
    ldconfig

# setup ssh and set bash as default shell
RUN \
    bash -c 'echo "PermitRootLogin yes" >> /etc/ssh/sshd_config'

RUN \
    bash -c 'echo -e "* soft memlock unlimited\n* hard memlock unlimited\n" >> /etc/security/limits.conf'

# add some tools
RUN \
    export DEBIAN_FRONTEND=noninteractive \
    && apt-get update -y -q \
    && apt-get install -y \
       aptitude \
       bash-completion \
       dnsutils \
       elinks \
       emacs25-nox emacs-goodies-el \
       fish \
       git \
       graphicsmagick \
       htop \
       iproute2 \
       iputils-ping \
       ipython3 \
       less \
       libatlas-base-dev \
       libblas-dev \
       libgeos-dev \
       libopenslide0 \
       mc \
       nload \
       nmon \
       psutils \
       python3-tk \
       python-h5py \
       python-tifffile \
       source-highlight \
       sudo \
       tmux \
       vim \
       wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade --no-cache pip \
    && pip3 install --upgrade --no-cache tqdm \ 
    && pip3 install --upgrade --no-cache pyyaml \ 
    && pip3 install --upgrade --no-cache numpy scipy matplotlib \
    && pip3 install --upgrade --no-cache sklearn \
    && pip3 install --upgrade --no-cache cassandra-driver \
    && pip3 install --upgrade --no-cache opencv-python \
    && pip3 install --upgrade --no-cache psutil  

## install cassandradl

ARG CDL_VERS=v0.1
RUN \
    git clone https://github.com/deephealthproject/CassandraDL.git \
    && cd CassandraDL \
    && git checkout $CDL_VERS \
    && pip3 install .

# configure user

ENV USER=sgd_mpi

RUN useradd -m $USER \
    && echo "$USER ALL=(ALL:ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/$USER

WORKDIR /home/$USER
COPY examples /home/$USER/examples
COPY varia/utils /home/$USER/utils
COPY varia/bin /home/$USER/bin
COPY opt_mpi /home/$USER/opt_mpi
RUN chown -R $USER:$USER /home/$USER
RUN chmod 755 /home/$USER/bin/*

# install OPT_MPI
RUN pip3 install /home/$USER/opt_mpi


USER $USER
WORKDIR /home/$USER

ENTRYPOINT \
    sudo service ssh restart \
    && ssh-keygen -t rsa -N '' -f ~/.ssh/id_rsa \
    && cat /home/$USER/.ssh/id_rsa.pub >> /home/$USER/.ssh/authorized_keys \
    && ssh-keyscan -t rsa $HOSTNAME >> /home/$USER/.ssh/known_hosts \
    && echo "$HOSTNAME slots=2" > /home/$USER/examples/hostfile\	
    && sleep infinity
