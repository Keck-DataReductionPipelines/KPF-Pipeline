init: 
	pip3 install -r requirements.txt
	pip install .
	mkdir -p logs

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

test:
	pytest --ignore=kpfpipe/tests/test_recipe.py --ignore=kpfpipe/tests/test_level1.py --cov=kpfpipe
	coveralls

.PHONY: init
