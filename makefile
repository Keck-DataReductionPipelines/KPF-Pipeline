init: 
	mkdir -p logs

	# turn off built-in Python multithreading
	export MKL_NUM_THREADS=1
	export NUMEXPR_NUM_THREADS=1
	export OMP_NUM_THREADS=1

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

regression_tests:
	pytest -n=8 --max-worker-restart 3 --cov=kpfpipe --cov=modules --pyargs tests.regression
	coveralls

performance_tests:
	pytest --pyargs tests.performance

validation_tests:
	pytest --pyargs tests.validation

.PHONY: init
