CCF_C=modules/CLib/CCF
init: 
	pip3 install -e .
	$(MAKE) C  -C ${CCF_C}

update: 
	pip3 install -r requirements.txt --upgrade

clear: 
	rm -f -r *.log

clean: clear
	rm -f -r build/
	rm -f -r dist/
	rm -f -r *.egg-info
	rm -f -r .pytest_cache
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +

notebook:
	pip3 install jupyter
	jupyter notebook --port 8888 --allow-root --ip=0.0.0.0 ""&

docker:
	docker build --cache-from kpf-drp:latest --tag kpf-drp:latest .
	docker run -it -v ${PWD}:/code/KPF-Pipeline -v ${KPFPIPE_TEST_DATA}:/testdata -v ${KPFPIPE_DATA}:/data kpf-drp:latest bash

regression_tests:
	pytest -s --cov=kpfpipe --cov=modules --pyargs tests.regression
	coveralls

performance_tests:
	pytest -s --pyargs tests.performance

validation_tests:
	pytest -s --pyargs tests.validation

.PHONY: init
