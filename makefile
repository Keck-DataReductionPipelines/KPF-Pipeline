init: 
	pip3 install -r requirements.txt
	python3 setup.py install
	mkdir -p logs

update: 
	pip3 install -r requirements.txt --upgrade

clear: 
	rm -f --recursive *.log

clean: clear
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info
	rm --force --recursive .pytest_cache
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +

test:
	pytest --ignore=kpfpipe/tests/test_recipe.py

.PHONY: init