sdist:
	python setup.py sdist

upload:
	twine upload dist/*
