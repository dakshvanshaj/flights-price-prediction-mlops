
# Start from the official Airflow image
FROM apache/airflow:3.0.4

# Copy your project's requirements file into the image
COPY ./requirements.txt /requirements.txt

# Install the Python packages from your requirements file
RUN pip install --no-cache-dir -r /requirements.txt