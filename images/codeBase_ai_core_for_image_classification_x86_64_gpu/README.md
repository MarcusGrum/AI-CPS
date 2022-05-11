
# Dealing with this local image

## Build local docker image manually with `Dockerfile` for Over-The-Air-Deployment of relevant data.

1. Build docker image from Dockerfile specified.

    ```
    docker build --tag codebase_ai_core_for_image_classification_x86_64_gpu .
    ```

1. Have a look on the image created.    
    
    ```
    docker run -it --rm codebase_ai_core_for_image_classification_x86_64_gpu sh
    ```
    
## Build local docker image manually with `yml` file for Over-The-Air-Deployment of relevant data.

1. If not available, yet, create independent volume for being bound to image.

    ```
    docker volume create ai_system
    ```
    
1. Build image with `docker-compose`.
    
    ```
    docker-compose build
    ```

## Test local docker image.

1. Start image with `docker-compose`.
    
    ```
    docker-compose up
    ```
    
1. Shut down image with `docker-compose`.
    
    ```
    docker-compose down
    ```

## Deploy local docker image to dockerhub.
 
1. Push image to `https://hub.docker.com/` of account called `marcusgrum`.
    
    ```
    docker image push marcusgrum/codebase_ai_core_for_image_classification_x86_64_gpu:latest
    ```
