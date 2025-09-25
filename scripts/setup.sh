#!/bin/bash
# setup.sh - LazyJobSearch Automated Setup Script
# This script will set up everything you need to run LazyJobSearch

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   log_error "This script should not be run as root!"
   exit 1
fi

# Banner
cat << "EOF"
╔══════════════════════════════════════════════════════════╗
║                  LazyJobSearch Setup                      ║
║         Production-Ready Job Scraping System              ║
╚══════════════════════════════════════════════════════════╝
EOF

echo ""
log_info "Starting LazyJobSearch setup..."
echo ""

# Check system requirements
log_info "Checking system requirements..."

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [[ $(echo "$PYTHON_VERSION >= 3.11" | bc -l) -eq 1 ]]; then
        log_success "Python $PYTHON_VERSION found"
    else
        log_error "Python 3.11+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    log_error "Python 3 not found"
    exit 1
fi

# Check Docker
if command -v docker &> /dev/null; then
    log_success "Docker found"
else
    log_warning "Docker not found. Install Docker for containerized deployment"
fi

# Check PostgreSQL client
if command -v psql &> /dev/null; then
    log_success "PostgreSQL client found"
else
    log_warning "PostgreSQL client not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install postgresql
    else
        sudo apt-get update && sudo apt-get install -y postgresql-client
    fi
fi

# Create project structure
log_info "Creating project structure..."
mkdir -p {logs,data,cache,debug/scraper_failures,monitoring/{grafana/{dashboards,datasources},prometheus}}
mkdir -p {libs/{db,scraper,tasks,nlp,companies},cli,config,alembic/versions,tests}

# Create virtual environment
log_info "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
log_info "Installing Python dependencies..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    log_warning "requirements.txt not found, installing core dependencies..."
    pip install \
        SQLAlchemy==2.0.29 \
        psycopg2-binary==2.9.9 \
        selenium==4.18.1 \
        undetected-chromedriver==3.5.5 \
        beautifulsoup4==4.12.3 \
        httpx==0.27.0 \
        typer[all]==0.12.3 \
        rich==13.7.1 \
        python-dotenv==1.0.1 \
        tenacity==8.2.3 \
        alembic==1.13.1
fi

# Create .env file if not exists
if [ ! -f .env ]; then
    log_info "Creating .env file..."
    cat > .env << 'EOL'
# Database
DATABASE_URL=postgresql://lazyjob:changeme@localhost:5432/lazyjobsearch

# API Keys (Get these from the providers)
INDEED_PUBLISHER_ID=
ADZUNA_APP_ID=
ADZUNA_API_KEY=

# Redis
REDIS_URL=redis://localhost:6379/0

# Application
APP_ENV=development
LOG_LEVEL=INFO
DEBUG=false

# Proxies (Optional)
PROXY_LIST=

# OpenAI (for embeddings)
OPENAI_API_KEY=
EOL
    log_success ".env file created. Please add your API keys!"
else
    log_info ".env file already exists"
fi

# Setup database
log_info "Setting up database..."
read -p "Do you want to use Docker for PostgreSQL? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Starting PostgreSQL with Docker..."
    docker run -d \
        --name ljs-postgres \
        -e POSTGRES_USER=lazyjob \
        -e POSTGRES_PASSWORD=changeme \
        -e POSTGRES_DB=lazyjobsearch \
        -p 5432:5432 \
        pgvector/pgvector:pg16
    
    log_info "Waiting for PostgreSQL to start..."
    sleep 10
    
    # Install pgvector extension
    docker exec ljs-postgres psql -U lazyjob -d lazyjobsearch -c "CREATE EXTENSION IF NOT EXISTS vector;"
    docker exec ljs-postgres psql -U lazyjob -d lazyjobsearch -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"
    
    log_success "PostgreSQL started and configured"
else
    log_info "Please ensure PostgreSQL is installed and running"
    log_info "Create a database named 'lazyjobsearch' and install pgvector extension"
fi

# Initialize Alembic
log_info "Initializing database migrations..."
if [ ! -f alembic.ini ]; then
    alembic init alembic
    # Update alembic.ini with correct database URL
    sed -i.bak 's|sqlalchemy.url = .*|sqlalchemy.url = postgresql://lazyjob:changeme@localhost:5432/lazyjobsearch|' alembic.ini
fi

# Run migrations
log_info "Running database migrations..."
alembic upgrade head

# Install Chrome/Chromium for Selenium
log_info "Checking Chrome installation..."
if ! command -v google-chrome &> /dev/null && ! command -v chromium &> /dev/null; then
    log_warning "Chrome not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install --cask google-chrome
    else
        wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
        sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list'
        sudo apt-get update
        sudo apt-get install -y google-chrome-stable
    fi
else
    log_success "Chrome found"
fi

# Download ChromeDriver
log_info "Setting up ChromeDriver..."
if [ ! -f /usr/local/bin/chromedriver ]; then
    CHROME_VERSION=$(google-chrome --version | grep -oP '\d+\.\d+\.\d+' | head -1)
    CHROMEDRIVER_VERSION=$(curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE_${CHROME_VERSION%%.*})
    
    wget -q "https://chromedriver.storage.googleapis.com/${CHROMEDRIVER_VERSION}/chromedriver_linux64.zip"
    unzip -o chromedriver_linux64.zip
    sudo mv chromedriver /usr/local/bin/
    sudo chmod +x /usr/local/bin/chromedriver
    rm chromedriver_linux64.zip
    
    log_success "ChromeDriver installed"
else
    log_success "ChromeDriver already installed"
fi

# Create CLI shortcut
log_info "Creating CLI shortcut..."
cat > ljs << 'EOL'
#!/bin/bash
source venv/bin/activate
python -m cli.ljs "$@"
EOL
chmod +x ljs

# Test installation
log_info "Testing installation..."
source venv/bin/activate
python -c "
import sys
try:
    import sqlalchemy
    import selenium
    import undetected_chromedriver
    import typer
    print('✅ All core packages installed successfully!')
except ImportError as e:
    print(f'❌ Missing package: {e}')
    sys.exit(1)
"

# Create systemd service (optional)
read -p "Do you want to create a systemd service for automatic crawling? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Creating systemd service..."
    
    SERVICE_FILE="lazyjobsearch.service"
    cat > $SERVICE_FILE << EOL
[Unit]
Description=LazyJobSearch Crawler Service
After=network.target postgresql.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$(pwd)/venv/bin/python -m cli.ljs crawl --mode hybrid
Restart=always
RestartSec=3600

[Install]
WantedBy=multi-user.target
EOL
    
    sudo mv $SERVICE_FILE /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable lazyjobsearch.service
    
    log_success "Systemd service created. Start with: sudo systemctl start lazyjobsearch"
fi

# Final setup steps
log_info "Running final setup..."

# Initialize the CLI configuration
source venv/bin/activate
python -m cli.ljs init --force

# Add sample companies
log_info "Adding sample companies..."
python -m cli.ljs add-company "Databricks" "databricks.com"
python -m cli.ljs add-company "Anthropic" "anthropic.com"
python -m cli.ljs add-company "OpenAI" "openai.com"

# Display summary
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║            Setup Complete! 🎉                             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
log_success "LazyJobSearch is ready to use!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your API keys:"
echo "   - Indeed Publisher ID (free): https://www.indeed.com/publisher"
echo "   - Adzuna API (free): https://developer.adzuna.com/"
echo ""
echo "2. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "3. Start crawling:"
echo "   ./ljs crawl --mode hybrid"
echo ""
echo "4. Monitor progress:"
echo "   ./ljs monitor --follow"
echo ""
echo "5. Check status:"
echo "   ./ljs status"
echo ""
echo "For Docker deployment, run:"
echo "   docker-compose up -d"
echo ""
echo "Documentation: https://github.com/yourusername/lazyjobsearch"
echo ""
log_info "Happy job hunting! 🚀"