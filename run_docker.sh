docker container ls -a | grep 'cigen' &> /dev/null
if [ $? == 0 ]; then
    docker start cigen
    docker exec -it --user root cigen bash
else
    docker run -it -v "$PWD":/root/cigen --name cigen ucdavisplse/cigen:latest
fi