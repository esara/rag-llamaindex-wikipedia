FROM python:3.12

EXPOSE 8501
USER root

WORKDIR /app

COPY requirements.txt /app/

RUN pip install -r requirements.txt
RUN opentelemetry-bootstrap -a install

COPY *.py /app/

#CMD ["streamlit", "run", "wiki_chat.py"]
CMD ["opentelemetry-instrument", "streamlit", "run", "wiki_chat.py"]
