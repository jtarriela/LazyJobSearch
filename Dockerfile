# Dockerfile - LazyJobSearch Production Container
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    # PostgreSQL client
    postgresql-client \
    libpq-dev \
    # Chrome dependencies for Selenium
    wget \
    gnupg \
    unzip \
    curl \
    # Image processing
    libmagic1 \
    # PDF processing
    poppler-utils \
    # Common utilities
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install Chrome for Selenium (if running locally)
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install ChromeDriver
RUN CHROMEDRIVER_VERSION=$(curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE) \
    && wget -q "https://chromedriver.storage.googleapis.com/${CHROMEDRIVER_VERSION}/chromedriver_linux64.zip" \
    && unzip chromedriver_linux64.zip \
    && mv chromedriver /usr/local/bin/ \
    && chmod +x /usr/local/bin/chromedriver \
    && rm chromedriver_linux64.zip

# Create app directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p \
    /app/logs \
    /app/data \
    /app/debug/scraper_failures \
    /app/cache \
    && chmod -R 755 /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Install additional tools
RUN pip install \
    supervisor \
    gunicorn \
    ipython

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -s /bin/bash ljs \
    && chown -R ljs:ljs /app

# Switch to non-root user
USER ljs

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command (can be overridden)
CMD ["python", "-m", "cli.ljs", "--help"]