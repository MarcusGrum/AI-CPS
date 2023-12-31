
# Dealing with this local image

## Build local docker image manually with `Dockerfile` for Over-The-Air-Deployment of relevant data.

1. Build docker image from Dockerfile specified.

    If you base this image on content of the current directory, execute the following:

    ```
    docker build --tag marcusgrum/codebase_ai_core_for_transport_classification_x86_64 .
    ```
	
	Since you base this image on content outside the current dockerfile context,	
	switch to parent repository path to execute the following instead:
	
	```
	docker build --tag marcusgrum/codebase_ai_core_for_transport_classification_x86_64:latest -f ./images/codeBase_ai_core_for_transport_classification_x86_64/Dockerfile .
	```

1. Have a look on the image created.    
    
    ```
    docker run -it --rm codebase_ai_core_for_transport_classification_x86_64 sh
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
    docker image push marcusgrum/codebase_ai_core_for_transport_classification_x86_64:latest
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
    docker buildx build --platform linux/arm/v7,linux/arm64/v8,linux/amd64 --tag marcusgrum/codebase_ai_core_for_transport_classification_x86_64:latest --push  .
    
    ```
    
    Since `buildex` creates platform-specific manifest files, any platform considers the corresponding image automatically.
