# start interactive docker container for running code
docker_int:
	docker run -it --rm \
		--user root \
		-v `pwd`:/marl-opt \
		-t marl-opt:1.0 \
		bash

# docker_int_gpu:
# 	nvidia-docker run -it --rm \
# 		--user root \
# 		--gpus all \
# 		-v `pwd`:/marl-opt \
# 		-t marl-opt:1.0 \
# 		bash


####################
# docker
####################
# use rsync to sync files between this computer and the titan computer (that's the "central" computer I'm using for storing results)
containers:
	docker ps -a

images:
	docker image ls

nvidia:
	watch nvidia-smi

# change permissions of any created files so they can be deleted, moved, etc. without any issues (ownership of files created in docker container)
chown:
	sudo chown -R `whoami` .

