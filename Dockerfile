# Use a slim Python image for a smaller footprint
FROM python:3.12.9-slim

WORKDIR /app

# 1. Copy only the requirements file
COPY src/prediction_server/requirements.prod.txt .

# 2. Install dependencies. This layer is now cached.
RUN pip install uv && uv pip install -r requirements.prod.txt --system

# 3. Now, copy the rest of the project files.
COPY . .

# 4. Set the PYTHONPATH to include the 'src' directory
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# 5. Make the entrypoint script executable
RUN chmod +x /app/docker-entrypoint.sh

# 6. Set the entrypoint script to handle runtime setup
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Expose the port the app runs on
EXPOSE 8000

# 7. Set the default command to be executed by the entrypoint
CMD ["uvicorn", "prediction_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
