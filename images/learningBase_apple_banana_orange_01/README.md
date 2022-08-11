
# Dealing with this local image

This image contains all data from 'fruits-fresh-and-rotten-fruits' dataset.

## Build local docker image manually with `Dockerfile` for Over-The-Air-Deployment of relevant data.

### Build local docker image manually with `Dockerfile`.

1. Build docker image from Dockerfile specified.

    ```
    docker build --tag marcusgrum/learningbase_apple_banana_orange_01:latest .
    ```

1. Have a look on the image created.    
    
    ```
    docker run -it --rm marcusgrum/learningbase_apple_banana_orange_01:latest sh
    ```

### Alternatively, build local docker image manually with `yml` file.

1. If not available, yet, create independent volume for being bound to image.

    ```
    docker volume create ai_system
    ```
    
1. Build image with `docker-compose`.
    
    ```
    docker-compose build
    ```

### Test local docker image.

1. Start image with `docker-compose`.
    
    ```
    docker-compose up
    ```

1. Test your image, e.g. by executing a shell.

    ```
    docker exec -it marcusgrum/learningbase_apple_banana_orange_01:latest sh
    ```
    
1. Shut down image with `docker-compose`.
    
    ```
    docker-compose down
    ```

### Deploy local docker image to dockerhub.
 
1. Push image to `https://hub.docker.com/` of account called `marcusgrum`.
    
    ```
    docker image push marcusgrum/learningbase_apple_banana_orange_01:latest
    ```
    
## Build and deploy your image for multiple architectures, such as `amd64`, `arm32v7` and `arm64v8 `.

1. Create a new builder which gives access to the new multi-architecutre features.

    ```
    docker buildx create --name mybuilder
    ```

1. Switch to this builder.

    ```
    docker buildx use mybuilder
    ```

1. Create images for corresponding architectures and push them to ´dockerhub´.

    ```
    docker buildx build --platform linux/arm/v7,linux/arm64/v8,linux/amd64 --tag marcusgrum/learningbase_apple_banana_orange_01:latest --push  .
    
    ```
    
    Since `buildex` creates platform-specific manifest files, any platform considers the corresponding image automatically.

# Credits

Picture material is coming from the following repositories:
Original fruit data is coming from kaggle repository [Sriram Reddy Kalluri](https://www.kaggle.com/sriramr/fruits-fresh-and-rotten-for-classification) under `unknown` license.