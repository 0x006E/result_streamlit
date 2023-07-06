FROM python:slim
RUN apt update && apt install git libgl-dev libglib2.0-0 -y
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV PORT 5000
EXPOSE $PORT
CMD python main.py
