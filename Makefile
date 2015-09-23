sdist:
	python setup.py sdist

upload:
	twine upload dist/*

clean:
	rm dist/*

all-dist: sdist upload clean
