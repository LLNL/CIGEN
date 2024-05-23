apt-get update -y
apt-get install -y apt-utils
DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
apt-get upgrade -y
apt-get install -y python3=3.11.2-1+b1 \
                    git=1:2.39.2-1.1 \
                    wget=1.21.3-1+b2 \
                    build-essential=12.9 \
                    gcc=4:12.2.0-3 \
                    make=4.3-4.1 \
                    cmake=3.25.1-1 \
                    lsb-release=12.0-1 \
                    software-properties-common=0.99.30-4 \
                    python3-pip=23.0.1+dfsg-1 \

pip3 install matplotlib==3.8.3 --break-system-packages
pip3 install -U scikit-learn==1.4.1.post1 --break-system-packages
pip3 install scikit-optimize==0.9.0 --break-system-packages