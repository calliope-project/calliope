deploy:
	scp -r lisa brutus:~/lisa

deploy-data:
	rsync --progress -avhe ssh Input/* brutus:~/lisa/Input
