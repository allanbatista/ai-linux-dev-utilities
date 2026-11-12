#!/usr/bin/env bash

docker system prune -a --volumes && \
docker volume prune -a && \
docker image prune -a
