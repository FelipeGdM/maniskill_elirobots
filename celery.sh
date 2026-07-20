#!/bin/bash

celery -A maniskill_elirobots.jobs.celery_app worker -E --concurrency=1 --loglevel=info
