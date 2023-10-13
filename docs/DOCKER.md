# Using Docker

The docker images can run the tool, but still need access to a GPU and needs to have a visual output (DISPLAY) when running. On windows running the tool in a docker container results in a significant performance loss. The docker image can be found [here](https://hub.docker.com/repository/docker/julrog/nn_vis).

## Run the tool

### Windows

Using WSL2 for docker:

```Shell
docker run -e DISPLAY=:0 -v /run/desktop/mnt/host/wslg/.X11-unix:/tmp/.X11-unix --gpus=all -it julrog/nn_vis:v1
```

### Linux

Not tested, but it should be the same just without the volume mount:

```Shell
docker run -e DISPLAY=:0 --gpus=all -it julrog/nn_vis:v1
```

## Development - Windows

Added `.devcontainer` folder to develop in a linux container for VSCode (only tested on Windows).
