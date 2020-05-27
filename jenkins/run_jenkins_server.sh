docker pull jenkins/jenkins

docker build . --tag jenkins-docker

docker run --privileged --rm -d --group-add 0 -v /var/run/docker.sock:/var/run/docker.sock -v jenkinsvol:/var/jenkins_home -p 5555:8080 -p 50000:50000 -P jenkins-docker
