FROM python:slim
RUN apt update && apt install git python3-numpy python3-pandas python3-opencv -y
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV PORT 5000
EXPOSE $PORT
RUN python main.py