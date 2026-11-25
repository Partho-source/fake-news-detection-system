# Fake News Detection System - production image (gunicorn)
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FNDS_DB=/app/var/fnds.db FNDS_MODEL=/app/var/model.pkl PORT=8000
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# download the real dataset and train (compares the 4 algorithms) while building the image
RUN mkdir -p var && python get_dataset.py && python train.py --dataset real
RUN useradd --create-home fnds && chown -R fnds /app/var
USER fnds
EXPOSE 8000
# one worker keeps a single copy of the model in memory, so a retrained model is used everywhere
CMD gunicorn --workers 1 --threads 8 --timeout 120 --bind 0.0.0.0:${PORT} app:app
