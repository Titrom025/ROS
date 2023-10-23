FROM ros:noetic

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    mesa-utils \
    libgl1-mesa-glx \
    python-dev \
    build-essential wget git

ENV CONDA_DIR /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV PIP_ROOT_USER_ACTION=ignore
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

RUN conda install python=3.8.10

RUN pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117  --index-url https://download.pytorch.org/whl/cu117
ADD requirements.txt ./

RUN pip install Cython
RUN pip install -r requirements.txt

ADD test_data.bag /sources/

# RUN git clone https://github.com/andrey1908/kas_utils.git /sources/
ADD kas_utils/ /sources/kas_utils/

# RUN git clone https://github.com/andrey1908/BoT-SORT /sources/catkin_ws/src/
ADD husky_tidy_bot_cv/ /sources/catkin_ws/src/husky_tidy_bot_cv/

# RUN git clone https://github.com/andrey1908/BoT-SORT /sources/BoT-SORT/ 
ADD BoT-SORT/ /sources/BoT-SORT/ 

WORKDIR /sources/kas_utils/python
RUN pip install .

WORKDIR /sources/BoT-SORT/ 
RUN pip install .

ADD rviz_conf.rviz /sources/rviz_conf.rviz

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
CMD ["/bin/bash", "/entrypoint.sh"]
