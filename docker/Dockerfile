FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

# Fix nvidia GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update
RUN apt-get install -y zsh git curl
RUN apt-get install -y build-essential libtool clang autoconf pkg-config cmake
RUN apt-get install -y ninja-build
RUN apt-get install -y python3-dev
RUN apt-get install -y \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libgl1 \
    libglx0 \
    libgles2 \
    libegl1 \
    libglvnd0 \
    libglvnd-dev
RUN apt-get install -y libglfw3-dev libglfw3
RUN apt-get install -y libjpeg-dev zlib1g-dev libopenexr-dev openexr
RUN apt-get install -y libeigen3-dev
RUN apt-get install -y libassimp-dev
RUN apt-get install -y libnuma-dev
RUN apt-get install -y libgoogle-glog-dev
RUN apt-get install -y libsuitesparse-dev
RUN apt-get install -y libatlas-base-dev

RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
 
RUN pip install loguru
RUN pip install setproctitle
RUN pip install opencv-python
RUN pip install mmcv-full
RUN pip install pytorch-lightning==1.6.5
RUN pip install thop
RUN pip install scipy
RUN pip install transforms3d
RUN pip install imageio imageio-ffmpeg
RUN pip install pypng
RUN pip install numba
RUN pip install scikit-image
RUN pip install einops
RUN pip install PyOpenGL PyOpenGL_accelerate
RUN pip install pyassimp==4.1.3
RUN pip install cython

RUN apt-get install -y vim

# RUN git clone https://github.com/curioushi/gdrnpp_bop2022
# RUN bash /workspace/gdrnpp_bop2022/scripts/compile_all.sh 
