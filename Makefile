.PHONY : doc-math

doc-math :
	python docs/pre_build/generate_math.py

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
