FROM python:3.13

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt && \
    rm requirements.txt

COPY ./ /measles-dashboard/

EXPOSE 8050
WORKDIR /measles-dashboard/

CMD ["gunicorn", "-c", "./gunicorn_conf.py", "app:server" ]

