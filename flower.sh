#!/bin/bash

export FLOWER_UNAUTHENTICATED_API=true

celery -A maniskill_elirobots.jobs.celery_app flower --port=5555
