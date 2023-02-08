#!/usr/bin/env bash

docker build --tag=application .

docker run -p 8081:8081 application