FROM ubuntu:20.04

RUN chmod 1777 /tmp
ENV TZ=Etc/UTC
ARG DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && apt-get install -y lsb-core curl
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

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
RUN pip install git+https://github.com/openai/CLIP.git

ADD test_data.bag /sources/

ADD init_clip_model.py /sources/init_clip_model.py
RUN python /sources/init_clip_model.py

# git clone https://github.com/andrey1908/kas_utils.git
ADD kas_utils/ /sources/kas_utils/

# git clone https://github.com/andrey1908/BoT-SORT
ADD BoT-SORT/ /sources/BoT-SORT/ 

# git clone https://github.com/andrey1908/husky_tidy_bot_cv
ADD husky_tidy_bot_cv/ /sources/catkin_ws/src/husky_tidy_bot_cv/

ADD rviz_conf.rviz /sources/rviz_conf.rviz

WORKDIR /sources/kas_utils/python
RUN pip install .

ADD modified_files/bot_sort_node.py /sources/catkin_ws/src/husky_tidy_bot_cv/scripts/bot_sort_node.py
ADD modified_files/bot_sort.py /sources/BoT-SORT/tracker/bot_sort.py
ADD modified_files/fast_reid_interfece.py /sources/BoT-SORT/fast_reid/fast_reid_interfece.py

WORKDIR /sources/BoT-SORT/ 
RUN pip install .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
CMD ["/bin/bash", "/entrypoint.sh"]
