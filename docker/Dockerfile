FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/miniconda3/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    USE_CUDA=1 \
    CUDA_VERSION=11.6.1 \
    CUDNN_VERSION=8 \
    LINUX_DISTRO=ubuntu \
    DISTRO_VERSION=20.04 \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6" \
    PYTHON_VERSION=3.9

# Prepare APT and install basic dependencies
RUN rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ccache \
    curl \
    git \
    wget \
    cmake \
    vim && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-py39_24.1.2-0-Linux-x86_64.sh
RUN curl -fsSL -v -o ~/miniconda.sh -O ${CONDA_URL} && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/miniconda3

# Set up the workspace
RUN mkdir /app
WORKDIR /app

# Create and activate a Conda environment
RUN conda create -y -n csrf python=3.9
SHELL ["conda", "run", "-n", "csrf", "/bin/bash", "-c"]

# Install PyTorch and related packages with automatic 'yes' to prompts
RUN conda install -y pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
RUN conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
RUN conda install -y -c bottler nvidiacub
RUN conda install -y pytorch3d -c pytorch3d

# Install other Python packages with pip
RUN pip install scipy==1.6.2 scikit-image==0.18.1 numba==0.53.1 nibabel==3.2.1 trimesh==3.9.15 tqdm==4.59.0 torchdiffeq==0.2.1
RUN pip install --no-index pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
RUN pip install monai==1.2.0

# Clean up APT, pip, and conda caches to reduce the image size
RUN apt-get clean && \
    pip cache purge && \
    conda clean -a

# Set the default command for the container
ENTRYPOINT ["/bin/bash"]
