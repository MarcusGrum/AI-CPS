
# Dealing with this local image

## Build local docker image manually with `Dockerfile` for Over-The-Air-Deployment of relevant data.

1. Build docker image from Dockerfile specified.

	If you base this image on content of the current directory, execute the following:

    ```
    docker build --tag marcusgrum/codebase_ai_core_for_image_classification_aarch64 .
    ```
	
	Since you base this image on content outside the current dockerfile context,	
	switch to parent repository path to execute the following instead:
	
	```
	docker build --tag marcusgrum/codebase_ai_core_for_image_classification_aarch64:latest -f ./images/codebase_ai_core_for_image_classification_aarch64/Dockerfile .
	```
	
1. Have a look on the image created.    
    
    ```
    docker run -it --rm codebase_ai_core_for_image_classification_aarch64 sh
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
    docker image push marcusgrum/codebase_ai_core_for_image_classification_aarch64:latest
    ```