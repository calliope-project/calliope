info:
	@echo 'Look at Makefile to see valid targets.'

deploy-brutus:
	scp -r lisa brutus:~/lisa

deploy-chemeng:
	scp -r lisa chemeng_cluster:~/lisa-pyomo

deploy-data-brutus:
	rsync --progress -avhe ssh Input/* brutus:~/lisa/Input
