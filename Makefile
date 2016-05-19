test:
	py.test --cov calliope --cov-report term-missing

sdist:
	python setup.py sdist

upload:
	twine upload dist/*

clean:
	rm dist/*

all-dist: sdist upload clean
