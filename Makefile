PROFILE_FILE := profile_$(shell /bin/date +'%Y-%m-%d-%H-%M-%S')

###
# Testing and linting
###

.PHONY : lint
lint :
	pylint calliope

.PHONY : test
test :
	py.test --cov calliope --cov-report term-missing -W ignore::FutureWarning
	# Run simple profiling job to spot major performance regressions
	calliope run calliope/example_models/national_scale/model.yaml --scenario=profiling --profile

.PHONY : profile
profile :
	mprof run -C -T 1.0 --python calliope run calliope/example_models/national_scale/model.yaml --scenario=profiling --profile --profile_filename=$(PROFILE_FILE).profile
	pyprof2calltree -i $(PROFILE_FILE).profile -o $(PROFILE_FILE).calltree
	gprof2dot -f callgrind $(PROFILE_FILE).calltree | dot -Tsvg -o $(PROFILE_FILE).callgraph.svg

.PHONY : profile-clean
profile-clean :
	rm profile_*
	rm mprofile_*

.PHONY : ci
ci :
ifeq ($(JOB_TYPE), lint)
	make lint
else
	make test
endif

.PHONY : all-ci
all-ci : test lint

.PHONY : doc-plots
doc-plots :
	python doc/helpers/generate_plots.py

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
