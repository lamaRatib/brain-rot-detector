VENV = .venv
PYTHON = python3
SRC = *.py

venv:
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)

compile: venv
	$(VENV)/bin/pip install --upgrade pip-tools
	$(VENV)/bin/pip-compile requirements.in

install: venv
	$(VENV)/bin/pip install -r requirements.txt

install-dev: venv
	$(VENV)/bin/pip install --upgrade pip-tools isort black mypy ruff

lint: venv
	$(VENV)/bin/isort --check $(SRC)
	$(VENV)/bin/black --line-length 120 --check $(SRC)
	$(VENV)/bin/ruff check $(SRC)
	mkdir -p .mypy_cache
	$(VENV)/bin/mypy --install-types --non-interactive $(SRC)

fmt format: venv
	$(VENV)/bin/isort $(SRC)
	$(VENV)/bin/black --line-length 120 $(SRC)

run: venv
	$(VENV)/bin/streamlit run app.py
