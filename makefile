CCF_C=modules/CLib/CCF
init: 
	pip3 install -e .
	$(MAKE) C  -C ${CCF_C}

update: 
	pip3 install -r requirements.txt --upgrade

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
	pip3 install jupyter
	jupyter notebook --port ${KPFPIPE_PORT} --allow-root --ip=0.0.0.0 ""

docker:
	docker build --cache-from kpf-drp:latest --tag kpf-drp:latest .

	$(if $(KPFPIPE_DATA),,$(error Must set KPFPIPE_DATA))
	$(if $(KPFPIPE_PORT),,docker run -it -v ${PWD}:/code/KPF-Pipeline -v ${KPFPIPE_TEST_DATA}:/testdata -v ${KPFPIPE_DATA}:/data -v ${KPFPIPE_DATA}/masters:/masters --network=host -e DBPORT=6125 -e DBNAME=kpfopsdb -e DBUSER=${KPFPIPE_DB_USER} -e DBPASS="${KPFPIPE_DB_PASS}" -e DBSERVER=127.0.0.1 kpf-drp:latest bash)
	docker run -it -p ${KPFPIPE_PORT}:${KPFPIPE_PORT} --network=host \
			   -e KPFPIPE_PORT=${KPFPIPE_PORT} -e DBPORT=6125 -e DBNAME=kpfopsdb -e DBUSER=${KPFPIPE_DB_USER} -e DBPASS="${KPFPIPE_DB_PASS}" -e DBSERVER=127.0.0.1 \
			   -v ${PWD}:/code/KPF-Pipeline -v ${KPFPIPE_TEST_DATA}:/testdata -v ${KPFPIPE_DATA}:/data -v ${KPFPIPE_DATA}/masters:/masters kpf-drp:latest bash

regression_tests:
	pytest -x --cov=kpfpipe --cov=modules --pyargs tests.regression.test_level0
	coveralls

performance_tests:
	pytest -x --pyargs tests.performance

validation_tests:
	pytest -x --pyargs tests.validation

.PHONY: init
