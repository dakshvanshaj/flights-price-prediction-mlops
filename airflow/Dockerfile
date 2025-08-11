# Base image airflow 3.0.4
FROM apache/airflow:3.0.4

# Set the working directory for your project
WORKDIR /opt/airflow/Flights_Mlops

# Copy dependency definition files
COPY airflow/requirements.txt ./airflow/requirements.txt

RUN pip install -r ./airflow/requirements.txt 

