# ROS Tracking docker  

## Сборка docker образа  

1) Склонировать репозитории:  
- ``git clone https://github.com/andrey1908/kas_utils.git``  
- ``git clone https://github.com/andrey1908/husky_tidy_bot_cv``  
- ``git clone https://github.com/andrey1908/BoT-SORT``  

2) Загрузить файл test_data.bag (ссылка в отчете)

3) Установить библиотеку для работы с nvidia в docker:  
``sudo apt-get install -y nvidia-container-toolkit``

и модифицировать "/etc/docker/daemon.json":

```
{  
    "runtimes": {  
        "nvidia": {  
            "path": "/usr/bin/nvidia-container-runtime",  
            "runtimeArgs": []  
        }  
    },  
    "default-runtime": "nvidia",  
    "data-root": "/mnt/archive/docker"  
}  
```

4) В файле BoT-SORT/fast_reid/fast_reid_interfece.py закомментировать строку 11 (строка не используется, но на данный момент вызывает конфликт версий пакетов)

5) Запустить сборку образа  
``docker build -t noetic_ros .``

6) Запустить контейнер 
``docker run -it --rm -e DISPLAY=:1 -v /tmp/.X11-unix:/tmp/.X11-unix --gpus all noetic_ros``