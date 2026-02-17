.PHONY: notebook

notebook:
	jupyter notebook --port ${KPFPIPE_PORT} --allow-root --ip=0.0.0.0 ""
