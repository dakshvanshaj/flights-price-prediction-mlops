# Use a slim Python image for a smaller footprint
FROM python:3.12.9-slim

# Set the working directory
WORKDIR /app

# Install uv, a faster package installer
RUN pip install uv

# Copy the production requirements file to the container
COPY src/prediction_server/requirements.prod.txt .

# Install dependencies using uv
RUN uv pip install -r requirements.prod.txt --system

# Install dependencies using pip(takes around 7 minutes using pip)
# RUN pip install --no-cache-dir -r requirements.prod.txt

# Copy the application source code
COPY src/ ./src

# Expose the port the app runs on
EXPOSE 8000

# Command to run the Uvicorn server
CMD ["uvicorn", "src.prediction_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
