FROM python:3.12-slim

WORKDIR /app

COPY ./requirements.txt .

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && pip install --timeout 3000 --retries 10 -r requirements.txt

COPY ./backend /app/backend
COPY ./frontend /app/frontend

# Copy entrypoint script to run both apps
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
