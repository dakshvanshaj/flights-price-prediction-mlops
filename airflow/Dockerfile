# Base image airflow 3.0.4
FROM apache/airflow:3.0.4

# Set the working directory for your project
WORKDIR /opt/airflow/Flights_Mlops

# Switch to root user to install packages
USER root

# Install system-level dependencies required for ML libraries like LightGBM
RUN apt-get update && apt-get install -y libgomp1

# Switch back to the airflow user
USER airflow

# Copy dependency definition files
COPY airflow/requirements.txt ./airflow/requirements.txt

RUN pip install -r ./airflow/requirements.txt 

