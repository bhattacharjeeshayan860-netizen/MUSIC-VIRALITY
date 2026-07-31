# ──────────────────────────────────────────────────────────────────────────────
# Music Virality System — Production-ish Runtime Image
# ──────────────────────────────────────────────────────────────────────────────
# Goals:
#   • Minimal runtime image (python-slim, no dev tooling, pip cache cleaned).
#   • Non-root user for security.
#   • Selective COPY instead of COPY . to avoid dragging .venv / .git / data / etc.
#   • Trained model artifacts are baked in so dashboards/API work out of the box.
#
# Build:  docker build -t music-virality .
# Sizes:  Python base ~120 MB; ML deps ~500-700 MB; app code/models ~5 MB.
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app

WORKDIR ${APP_HOME}

# Create non-root user/group early
RUN groupadd --system appuser && \
    useradd --system --gid appuser --create-home appuser

# Install runtime dependencies first so the layer is cached until requirements change.
# We copy only requirements.txt first = minimal rebuild when source code changes.
COPY --chown=appuser:appuser requirements.txt ${APP_HOME}/requirements.txt

# Most deps ship wheels. If a wheel is missing and compilation is needed,
# build-essential can be installed temporarily and removed in the same RUN.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    apt-get purge -y --auto-remove gcc g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy only what is needed to run the system.
# Code / config / assets
COPY --chown=appuser:appuser src/ ${APP_HOME}/src/
COPY --chown=appuser:appuser config/ ${APP_HOME}/config/
COPY --chown=appuser:appuser .streamlit/ ${APP_HOME}/.streamlit/
COPY --chown=appuser:appuser dashboard.py dashboard_prediction.py main.py ${APP_HOME}/

# Trained artifacts — required for serving. Keep model sizes small in git.
RUN mkdir -p ${APP_HOME}/models/trained ${APP_HOME}/data/raw ${APP_HOME}/data/processed
COPY --chown=appuser:appuser models/trained/ ${APP_HOME}/models/trained/

# Ensure the non-root user owns the working tree
RUN chown -R appuser:appuser ${APP_HOME}

# Switch to non-root user for all runtime operations
USER appuser

# Expose the ports we serve on
EXPOSE 8501 8000

# Default to the detection dashboard; override via docker run command.
CMD python -m streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
