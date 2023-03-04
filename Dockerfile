# Use an official Python runtime as the base image

FROM nvidia/cuda:10.1-runtime-ubuntu18.04

# Set the working directory
WORKDIR /app
COPY . /app


RUN apt update || true
RUN apt install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt -y install python3.8 python3-pip
RUN apt install -y unzip git qt5-default vim build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
RUN apt-get -y install build-essential cmake
RUN apt-get -y install git libjpeg-dev libtiff5-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev \
libv4l-dev v4l-utils libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgtk2.0-dev mesa-utils libgl1-mesa-dri libgtkgl2.0-dev \ 
libgtkglext1-dev libatlas-base-dev gfortran libeigen3-dev python3-dev python3-numpy 


RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install -r requirements.txt

WORKDIR /app
RUN git clone https://github.com/opencv/opencv.git
RUN git clone https://github.com/opencv/opencv_contrib.git

WORKDIR /app/opencv_contrib
RUN git checkout 4.2.0

WORKDIR /app/opencv
RUN git checkout 4.2.0 && mkdir build

WORKDIR /app/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
       -D CMAKE_INSTALL_PREFIX=/usr/local \
       -D INSTALL_C_EXAMPLES=ON \
       -D INSTALL_PYTHON_EXAMPLES=ON \
       -D WITH_TBB=ON \
       -D WITH_V4L=ON \
       -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
       -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
       -D BUILD_EXAMPLES=ON ..

RUN make -j16 && make install

WORKDIR /app
RUN wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip && unzip models.zip

RUN rm -rf opencv opencv_contrib models.zip

CMD sh -c 'python3.8 evaluate.py --model=$MODEL_PATH --dataset-path=$DATASET_PATH --output-path=$OUTPUT_PATH'

