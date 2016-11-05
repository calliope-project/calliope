###
# Testing and linting
###

.PHONY : lint
lint :
	pylint calliope

.PHONY : test
test :
	py.test --cov calliope --cov-report term-missing

.PHONY : ci
ci :
ifeq ($(JOB_TYPE), lint)
	make lint
else
	make test
endif

.PHONY : all-ci
all-ci : test lint

###
# Build package and upload to PyPI
###

.PHONY : sdist
sdist :
	python setup.py sdist

.PHONY : upload
upload :
	twine upload dist/*

.PHONY : clean
clean :
	rm dist/*

.PHONY : all-dist
all-dist : sdist upload clean
