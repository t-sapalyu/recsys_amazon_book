#! /usr/bin/env bash

# docker-compose's healthcheck already gates startup on Postgres being
# ready; this is just a small extra buffer for the app-level connection.
sleep 3
python /app/prestart.py
