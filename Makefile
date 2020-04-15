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
