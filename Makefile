.PHONY: all format lint test tests test_watch integration_tests docker_tests help extended_tests

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

test:
	python -m pytest $(TEST_FILE)

test_watch:
	python -m ptw --snapshot-update --now . -- -vv tests/unit_tests

test_profile:
	python -m pytest -vv tests/unit_tests/ --profile-svg

extended_tests:
	python -m pytest --only-extended $(TEST_FILE)

######################
# LANGSMITH MONITORING
######################

monitor:
	python monitor_with_langsmith.py

dashboard:
	python check_langsmith_dashboard.py

test_langsmith:
	python -m pytest tests/integration_tests/test_graph.py -v

# Comprehensive LangSmith Testing
evaluate_comprehensive:
	python langsmith_lease_evaluator.py

ab_test:
	python langsmith_ab_testing.py

performance_benchmark:
	python langsmith_performance_benchmark.py

# Combined testing workflow
test_full_suite: test evaluate_comprehensive performance_benchmark
	@echo "✅ Complete testing suite finished"

# Comprehensive testing orchestrator
test_orchestrated:
	python test_orchestrator.py

# Selective pipeline testing (e.g., make test_selective PIPELINES=unit_tests,integration_tests)
test_selective:
	python test_orchestrator.py $(PIPELINES)

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=src/
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=src
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	python -m ruff check .
	[ "$(PYTHON_FILES)" = "" ] || python -m ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || python -m ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || python -m mypy --strict $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && python -m mypy --strict $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	ruff format $(PYTHON_FILES)
	ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	codespell --toml pyproject.toml

spell_fix:
	codespell --toml pyproject.toml -w

######################
# HELP
######################

help:
	@echo '----'
	@echo 'TESTING COMMANDS:'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'test_watch                   - run unit tests in watch mode'
	@echo 'test_langsmith               - run LangSmith integration tests'
	@echo 'test_full_suite              - run complete testing suite'
	@echo 'test_orchestrated            - run comprehensive testing orchestrator'
	@echo 'test_selective PIPELINES=... - run selective pipeline testing'
	@echo '----'
	@echo 'LANGSMITH EVALUATION:'
	@echo 'evaluate_comprehensive       - run comprehensive lease analysis evaluation'
	@echo 'ab_test                      - run A/B testing for configuration optimization'
	@echo 'performance_benchmark        - run performance benchmarking and monitoring'
	@echo '----'
	@echo 'MONITORING:'
	@echo 'monitor                      - run comprehensive LangSmith monitoring'
	@echo 'dashboard                    - quick LangSmith dashboard check'
	@echo '----'
	@echo 'CODE QUALITY:'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'

