# Base image airflow 3.0.4
FROM apache/airflow:3.0.4

# Set the AIRFLOW_HOME and working directory
ENV AIRFLOW_HOME=/opt/airflow
WORKDIR /opt/airflow

# Switch to root user to install packages
USER root

# Install system-level dependencies required for ML libraries like LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Switch back to the airflow user
USER airflow

# Copy project source code into the image
# This makes the image self-contained and removes the need for a large volume mount.
COPY --chown=airflow:root . /opt/airflow

# Install Airflow-specific requirements first
RUN pip install --no-cache-dir -r /opt/airflow/airflow/requirements.airflow.txt

# Install the project's own dependencies in editable mode.
# This makes 'src' importable and installs packages from pyproject.toml.
RUN pip install --no-cache-dir -e .

# The dvc executable will be at /home/airflow/.local/bin/dvc
# We add this to the PATH so it can be called directly.
ENV PATH="/home/airflow/.local/bin:${PATH}"

# Initialize DVC within the image build process
# RUN dvc init --no-scm
