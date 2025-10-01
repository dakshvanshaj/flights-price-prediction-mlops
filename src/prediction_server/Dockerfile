# ==============================================================================
# Stage 1: The "Builder" Stage
# ==============================================================================
# This stage installs uv and uses it to install all Python dependencies.
# The result is a clean site-packages directory that we can copy later.
FROM python:3.12.3-slim AS builder

# 1. Install uv in its own layer
RUN pip install uv

# 2. Create a virtual environment for clean dependency management
WORKDIR /app
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# 3. Copy only the requirements file and install dependencies
# This layer is heavily cached and will only rerun if requirements change.
COPY src/prediction_server/requirements.prod.txt ./
RUN uv pip install -r requirements.prod.txt

# ==============================================================================
# Stage 2: The "Final" Runtime Stage
# ==============================================================================
# This is the final, slim image that will be used in production.
FROM python:3.12.3-slim

WORKDIR /app

# 1. Install *only* the necessary runtime system dependencies.
# lightgbm (and other libraries) needs this shared library to run.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 2. Copy the installed packages from the builder stage.
# This is the core of the multi-stage build. We get all the packages
# without bringing the installer (uv) or build cache with them.
COPY --from=builder /app/venv /app/venv

# 3. Copy the application code and configuration.
# This is done *after* dependencies so code changes don't invalidate the cache.
ENV PYTHONPATH="/app/src"
COPY src/ /app/src/
COPY params.yaml /app/params.yaml
COPY dvc.yaml .
COPY dvc.lock .

# 4. Activate the virtual environment and set up the runtime.
ENV PATH="/app/venv/bin:$PATH"
RUN dvc init --no-scm
RUN chmod +x /app/src/prediction_server/docker-entrypoint.sh

# 5. Define the entrypoint, port, and default command.
ENTRYPOINT ["/app/src/prediction_server/docker-entrypoint.sh"]
EXPOSE 9000
CMD ["uvicorn", "prediction_server.main:app", "--host", "0.0.0.0", "--port", "9000"]