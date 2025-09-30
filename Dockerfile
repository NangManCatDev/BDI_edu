FROM python:3.12.3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 로컬 pylo 포함
COPY vendor/pylo /app/pylo
COPY . .

ENV PYTHONPATH=/app

CMD ["uvicorn", "interface.chat_ui:app", "--host", "0.0.0.0", "--port", "8000"]
