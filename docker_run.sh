docker build -t noetic_full . ; \ 
docker run -it --name ros --rm -e DISPLAY=:1 \
   -v /tmp/.X11-unix:/tmp/.X11-unix --gpus all \
   --mount type=bind,source=/home/roman.titkov/ROS/docker_data/,target=/docker_data \
   --mount type=bind,source=/home/roman.titkov/ROS/cache_clip,target=/root/.cache/clip \
   --mount type=bind,source=/home/roman.titkov/ROS/cache_torch,target=/root/.torch/iopath_cache/ \
   noetic_full
