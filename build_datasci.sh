#!/bin/bash

# setup git stuff
sudo apt-get install -y git
git config --global user.name "MasonGallo"
git config --global user.email "MasonGallo@users.noreply.github.com"

# setup R
sudo sh -c 'echo "deb http://cran.rstudio.com/bin/linux/ubuntu trusty/" >> /etc/apt/sources.list'
gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
gpg -a --export E084DAB9 | sudo apt-key add -
sudo apt-get update
sudo apt-get -y install r-base r-base-dev
sudo apt-get -y install libcurl4-gnutls-dev libxml2-dev libssl-dev
sudo apt-get -y install openjdk-7-jdk
export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-i386
export LD_LIBRARY_PATH=$JAVA_HOME/jre/lib/i386:$JAVA_HOME/jre/lib/i386/client
sudo R CMD javareconf
echo -e "\nlocal({r <- getOption('repos')\nr['CRAN'] <- 'https://cran.rstudio.com'\noptions(repos=r)\n})" >> ~/.Rprofile

# get RStudio
sudo apt-get -y install gdebi-core
wget -O /tmp/rstudio-server-latest-amd64.deb http://www.rstudio.org/download/latest/stable/server/ubuntu64/rstudio-server-latest-amd64.deb
sudo gdebi -n /tmp/rstudio-server-latest-amd64.deb
