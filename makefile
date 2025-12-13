# ---------- Config ----------
APP_IMAGE ?= kpf-drp
CI_IMAGE  ?= kpf-drp-ci
TAG       ?= latest
MASTERS_IMAGE ?= kpfmastersdrp

ifndef KPFCRONJOB_DOCKER_IMAGE
    MASTERS_IMAGE_WITH_TAG ?= $(MASTERS_IMAGE):$(TAG)
    $(info KPFCRONJOB_DOCKER_IMAGE is not defined, so defaulting to kpfmastersdrp:latest)
else
    MASTERS_IMAGE_WITH_TAG = $(KPFCRONJOB_DOCKER_IMAGE)
    $(info KPFCRONJOB_DOCKER_IMAGE is defined, and is set to $(KPFCRONJOB_DOCKER_IMAGE))
endif

# Cache-busting when requirements.txt changes (Dockerfile must consume REQS_SHA)
REQS_SHA  := $(shell sha256sum requirements.txt | cut -d ' ' -f1)

# test_env behavior:
# - Local default: interactive shell in the CI image
# - Jenkins: set DOCKER_RUN_TTY= (empty) and RUN="make init regression_tests"
DOCKER_RUN_TTY ?= -it
RUN ?= bash

CCF_C = modules/CLib/CCF

init:
	pip3 install -e . --quiet --no-warn-script-location
	$(MAKE) C -C ${CCF_C}

update:
	pip3 install -r requirements.txt --upgrade --quiet --no-warn-script-location

clear:
	rm -f -r *.log
	rm -f -r cores
	rm -f    core.*
	rm -f    temp_kpf_ts_*.db
	rm -f -r temp_kpf_ts_plots*
	rm -f -r temp_QLP_plots*

clean: clear
	rm -f -r build/
	rm -f -r dist/
	rm -f -r *.egg-info
	rm -f -r .pytest_cache
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +

notebook:
	pip3 install jupyter --quiet --no-warn-script-location
	jupyter notebook --port ${KPFPIPE_PORT} --allow-root --ip=0.0.0.0 ""

docker:
	@echo "Building Docker image for KPF DRP $(APP_IMAGE):$(TAG)…"
	@DOCKER_BUILDKIT=1 docker build \
		--cache-from $(APP_IMAGE):$(TAG) \
		--build-arg REQS_SHA=$(REQS_SHA) \
		--tag $(APP_IMAGE):$(TAG) . --quiet
	$(if $(KPFPIPE_DATA),,$(error Must set KPFPIPE_DATA))
	$(if $(KPFPIPE_PORT),, @echo "Starting Docker container (no port specified)..." && ./docker-run.sh)
	$(if $(KPFPIPE_PORT), @echo "Starting Docker container on port ${KPFPIPE_PORT}..." && KPFPIPE_PORT=${KPFPIPE_PORT} ./docker-run.sh)

docker_masters:
	@echo "Building Docker image for KPF masters pipeline $(MASTERS_IMAGE_WITH_TAG)…"
	@DOCKER_BUILDKIT=1 docker build \
		--no-cache \
		--build-arg REQS_SHA=$(REQS_SHA) \
		--tag $(MASTERS_IMAGE_WITH_TAG) . --quiet
	$(if $(KPFPIPE_DATA),,$(error Must set KPFPIPE_DATA))
	$(if $(KPFPIPE_PORT),, @echo "Starting Docker container (no port specified)..." && ./docker-masters-run.sh)
	$(if $(KPFPIPE_PORT), @echo "Starting Docker container on port ${KPFPIPE_PORT}..." && KPFPIPE_PORT=${KPFPIPE_PORT} ./docker-masters-run.sh)

# Build the CI image and run inside it (interactive by default; Jenkins overrides RUN/tty)
test_env:
	DOCKER_BUILDKIT=1 docker build --cache-from $(CI_IMAGE):$(TAG) \
		--build-arg REQS_SHA=$(REQS_SHA) \
		--tag $(CI_IMAGE):$(TAG) .
	docker run $(DOCKER_RUN_TTY) --rm \
		--network=host \
			   \
		-v "$${PWD}:/code/KPF-Pipeline" \
		-v "$${CI_DATA_DIR}:/data" \
		-v "$${CI_DATA_DIR}/masters:/masters" \
		-v "$${KPFPIPE_TEST_DATA}:/testdata" \
		-e COVERALLS_REPO_TOKEN=VQhy1molIcAo0rTz2geFOhucmvkBiEPFc \
		-e CI_PULL_REQUEST=$$ghprbPullId \
		-e DBPORT=6125 \
		-e DBNAME=kpfopsdb \
		-e DBUSER=$${KPFPIPE_DB_USER} \
		-e DBPASS="$${KPFPIPE_DB_PASS}" \
		-e DBSERVER=127.0.0.1 \
		-e TSDBSERVER=127.0.0.1 \
		-e TSDBPORT=6127 \
		-e TSDBNAME=timeseriesopsdb \
		-e TSDBUSER=$${KPFPIPE_TSDB_USER} \
		-e TSDBPASS="$${KPFPIPE_TSDB_PASS}" \
		$(CI_IMAGE):$(TAG) \
		$(RUN)

# Dependency smoke test inside the current Python env (not Docker)
sanity:
	@python -c "import sys; \
try: \
    import pvlib, pandas, numpy; \
    print('pvlib:', pvlib.__version__); \
    print('pandas:', pandas.__version__); \
    print('numpy:', numpy.__version__); \
except Exception as e: \
    print(f'Dependency check failed: {e}', file=sys.stderr); \
    sys.exit(1)"

regression_tests:
	pytest -x --cov=kpfpipe --cov=modules --pyargs tests.regression
	coveralls

performance_tests:
	pytest -x --pyargs tests.performance

validation_tests:
	pytest -x --pyargs tests.validation

.PHONY: init update clear clean notebook docker test_env sanity regression_tests performance_tests validation_tests
