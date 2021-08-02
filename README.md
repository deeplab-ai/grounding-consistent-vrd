# Deeplab: VRD

Visual Relationship Detection

## Repo Overview
This repo has three major components:

1. Prerequisites

2. Training (research)

3. Serving (api)


The prerequisites part downloads all the relevant datasets in order to proceed with the training experiments.

The training one is for training new models that can be passed to the serving one and go live.

The api is a backend implementation using Flask and Gunicorn. This api is exposed through on-premise k8s-cluster (plato).
  
## How to run it
The repo has a handy Makefile for these three components, taking care of:

1. docker tag: the tag of the image is the git_commit.
 
2. model output path and logs.

First of all:

- git commit your code

- docker login

- export KUBECONFIG=...

## Prerequisites

```
make prereqs
```

output: /nfs-data/vrd-prerequisites/{all the datasets} 

## Research (train)

```
make train
```

output: /nfs-data/vrd-prerequisites/{GIT_COMMIT}

where this directory contains:

- model.pt
- model logs  


## Serve (api)

Edit Makefile to specify OBJ_MODEL_PATH and REL_MODEL_PATH. Then run:

```
make serve
```

By specifying the models that you want to serve, the new image goes live substituting the old model version. 

## Under the hood

- Each and every of these three components is dockerized.
- The produced images are pushed to the relevant dockerhub registry.
- All the docker containers run on deeplab.ai k8s-cluster.

## Under the serving hood
 
The api is being served on k8s-cluster as a "Deployment" and is exposed as a k8s "Service" on nodePort: 32304 on node: Plato

!! The nodePort in order to be accessible outside node, needs to be exposed via NGINX.

NGINX instance has been configured on Plato node with the following configs:

/etc/nginx/nginx.conf
 
```
include /etc/nginx/conf.d/*.conf;
        include /etc/nginx/sites-enabled/*;
        server {
        listen 8888;
        listen [::]:8888;
        server_name 172.16.2.9
        access_log /var/log/nginx/reverse-access.log;
        error_log /var/log/nginx/reverse-error.log;

        location / {
                    proxy_pass http://172.16.2.9:32304/new_image;
                    }
                }
```
 
So this port: 8888 has been forwarded via UniFi Network:
 
```
172.16.2.9:8888 --> athens-deeplab.selfip.com:1195
```

in order to be publicly available.

## Monitoring

Since all these components run on k8s-cluster, the k8s-dashboard is very helpful for prereqs, training and serving pods.

k8s-dashboard:

```
https://172.16.2.20:32155/
```

## API Monitoring (requests)

In order to monitor the incoming requests of the API, a lightweight solution has been used: flask-monitoringdashboard.
This is a simple package that binds on the existing flask app, creating a new /dashboard route.
Under the hood, sqlite is being used which is securely stored on the nfs volume.

```
https://172.16.2.9:32304/dashboard
```

## Important config details

 To successfully set and run a VRD experiment, mind the following:

- For prerequisites:

 1. prerequisites_config.yaml: Sets up the path where various VRD-related items (annotations, models, logs etc.) will be stored.

 2. main_prerequisites.py: Reads from 1.

 3. prerequisites/data_config.py: Reads from 2, configures various paths (annotations, GloVe, images)

- For research:

 1. prerequisites_config.yaml: Sets up the path where various VRD-related items (annotations, models, logs etc.) will be stored.

 2. main_research.py Reads from 1 and argparse.

 3. common/config.py: Basic config class, reads from 1, 2.

 4. research/config.py: Inherits 3, reads from 2. Configurates parmeters related to research. Also, paths for logs, stored models and results are further deployed here.

- For production:

 1. prerequisites_config.yaml: Sets up the path where various VRD-related items (annotations, models, logs etc.) will be stored.

 2. production/api_properties.yaml: Params such as valid image-file-types, image sizes etc.

 3. production/flask_dashboard.cfg: Configures params for logging.

 4. production/api/main_production.py: Reads from 2, 3.

 5. common/config.py: Basic config class, reads from 1, 4.

 6. production/graph_inference/main.py: Reads from 2, 4, 5.

 7. production/graph_inference/object_inference.py: Reads from 2, 6.

 8. production/graph_inference/relation_inference.py: Reads from 2, 6.

