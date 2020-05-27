docker pull jenkins/jenkins

# docker-compose up --build

docker build . --tag jenkins-docker

docker run --privileged --rm -d --group-add 0 -v /var/run/docker.sock:/var/run/docker.sock -v jenkinsvol:/var/jenkins_home -v ${KPFPIPE_TEST_DATA}:/data/ -v ${KPFPIPE_TEST_OUTPUTS}:/outputs/ -p 5555:8080 -p 50000:50000 -P jenkins-docker
