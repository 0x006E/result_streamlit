FROM python:slim
RUN apt update && apt install git -y
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV PORT 5000
EXPOSE $PORT
RUN python main.py