FROM python:3.7-slim
RUN pip install numpy
RUN pip install pandas
RUN pip install matplotlib
RUN mkdir /src
WORKDIR /src
COPY lsr.py ./
ADD train_data ./
ENTRYPOINT ["python", "lsr.py"] 
