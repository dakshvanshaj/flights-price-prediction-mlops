# Base image airflow 3.0.4
FROM apache/airflow:3.0.4

# Set a working directory for all subsequent commands
WORKDIR /opt/airflow/Flights_Mlops

# This command copies everything from the build context (our project)
# into the current WORKDIR (/opt/airflow/Flights_Mlops).
# and changes owner ship to airflow:root
COPY --chown=airflow:root . .

# install the project Pip will find 'pyproject.toml' in pwd.
RUN pip install --no-deps .

# Finally, install the dependencies from your requirements.txt file.
# The path is now relative to our WORKDIR.
RUN pip install --no-cache-dir -r ./airflow/requirements.txt