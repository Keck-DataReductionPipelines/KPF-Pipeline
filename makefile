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
	@echo "Building Docker image..."
	@DOCKER_BUILDKIT=1 docker build --cache-from kpf-drp:latest --tag kpf-drp:latest . --quiet
	$(if $(KPFPIPE_DATA),,$(error Must set KPFPIPE_DATA))
	$(if $(KPFPIPE_PORT),, @echo "Starting Docker container (no port specified)..." && ./docker-run.sh)
	$(if $(KPFPIPE_PORT), @echo "Starting Docker container on port ${KPFPIPE_PORT}..." && KPFPIPE_PORT=${KPFPIPE_PORT} ./docker-run.sh)

test_env:
	docker build --cache-from kpf-drp-ci:latest --tag kpf-drp-ci:latest .
	docker run -it --rm \
		--network=host \
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
		-e TSDBPASS=$${KPFPIPE_TSDB_PASS} \
		kpf-drp:latest \
		bash

regression_tests:
	pytest -x --cov=kpfpipe --cov=modules --pyargs tests.regression
	coveralls

performance_tests:
	pytest -x --pyargs tests.performance

validation_tests:
	pytest -x --pyargs tests.validation

.PHONY: init update clear clean notebook docker regression_tests performance_tests validation_tests
