UV_RUN	= uv run python3
MYPY	= $(UV_RUN) -m mypy
FLAKE8	= $(UV_RUN) -m flake8

ARGS	?=

run: install
	@$(UV_RUN) -B -m src $(ARGS)

install:
	uv sync

debug: install
	@$(UV_RUN) -B -m pdb -m src $(ARGS)

lint: install
	$(FLAKE8) src
	$(MYPY) src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: install
	$(FLAKE8) src
	$(MYPY) src --strict

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

fclean: clean
	rm -rf .venv

.PHONY: run clean fclean install lint lint-strict debug