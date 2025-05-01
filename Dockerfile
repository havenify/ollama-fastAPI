FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve_llm.py .

EXPOSE 8288

CMD ["python", "serve_llm.py"]
