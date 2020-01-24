init: 
	pip3 install -r requirements.txt
	python3 setup.py install

update: 
	pip3 install -r requirements.txt --upgrade

clear:
	find . -type f -name '*.png' -delete

clean:
	rm -rf *.egg
	find . -type f -name '*.xlsx' -delete
	find . -type f -name '*.log' -delete
	find . | grep -E "(__pycache__)" | xargs rm -rf

.PHONY: init