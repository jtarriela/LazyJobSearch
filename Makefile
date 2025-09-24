PROJECT ?= lazyjobsearch
COMPOSE ?= docker compose
PYTHON  ?= python

.PHONY: help up down clean-volume rebuild rebuild-app logs shell db-shell db-init install-deps seed fresh

help:
	@echo "Targets:"
	@echo "  up              - Start services"
	@echo "  down            - Stop services (keep volumes)"
	@echo "  clean-volume    - Stop and remove volumes (DB wiped)"
	@echo "  rebuild         - Rebuild all images (no cache)"
	@echo "  rebuild-app     - Rebuild only app image"
	@echo "  logs            - Follow logs"
	@echo "  shell           - Enter app container"
	@echo "  db-shell        - psql into postgres"
	@echo "  db-init         - Create tables (dev convenience)"
	@echo "  install-deps    - Rebuild app after dep changes"
	@echo "  seed            - Create companies seed + run seeding"
	@echo "  fresh           - Full rebuild & table init"

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

clean-volume:
	$(COMPOSE) down -v

rebuild:
	$(COMPOSE) build --no-cache

rebuild-app:
	$(COMPOSE) build app

logs:
	$(COMPOSE) logs -f

shell:
	$(COMPOSE) exec app bash

db-shell:
	$(COMPOSE) exec postgres psql -U postgres -d lazyjobsearch

db-init:
	$(COMPOSE) exec app $(PYTHON) -c "from libs.db.session import engine; from libs.db import models; models.Base.metadata.create_all(bind=engine); print('Tables created.')"

install-deps: rebuild-app up

seed:
	@mkdir -p seeds
	printf "Anduril\nPalantir\nOpenAI\n" > seeds/companies.txt
	$(COMPOSE) exec app $(PYTHON) cli/ljs.py companies seed seeds/companies.txt

fresh:
	$(COMPOSE) down -v
	$(COMPOSE) up -d --build
	$(COMPOSE) exec app $(PYTHON) -c "from libs.db.session import engine; from libs.db import models; models.Base.metadata.create_all(bind=engine); print('Tables created.')"