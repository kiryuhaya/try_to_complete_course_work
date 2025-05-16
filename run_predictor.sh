#!/bin/bash
cd predictor_service
uvicorn predictor:app --host 127.0.0.1 --port 8000
