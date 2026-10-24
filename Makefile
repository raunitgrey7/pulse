.PHONY: setup lint format test bench serve fleet docker-build clean

UV ?= uv

setup:
	$(UV) sync --all-extras --dev

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

test:
	$(UV) run pytest -q

bench:
	$(UV) run python -m pulse.bench

serve:
	$(UV) run pulse serve --host 0.0.0.0 --port 8000

fleet:
	$(UV) run pulse fleet --api http://127.0.0.1:8000 --devices 12 --rounds 60 --interval 1.0

docker-build:
	docker build -t pulse:local .

clean:
	rm -rf artifacts .pytest_cache .ruff_cache .hypothesis
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
