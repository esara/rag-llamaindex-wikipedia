SHELL := /bin/bash

DOCKER_REGISTRY = docker.io/esara
DOCKER_IMAGE = rag
DOCKER_TAG = traceloop

.PHONY: build

image: build Dockerfile
	docker buildx build --platform linux/amd64,linux/arm64 --push -t $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG) .
