init: 
	mkdir -p logs
	pip3 install -r requirements.txt .

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

docker:
	docker build --cache-from kpf-drp:latest --tag kpf-drp:latest .
	docker run -it -v ${KPFPIPE_TEST_DATA}:/data kpf-drp:latest make init regression_tests

regression_tests:
	pytest -n 16 --cov=kpfpipe --cov=modules --pyargs tests.regression
	coveralls

performance_tests:
	pytest --pyargs tests.performance

validation_tests:
	pytest --pyargs tests.validation

.PHONY: init
