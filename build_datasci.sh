#!/bin/bash

# setup git stuff
git config --global user.name "MasonGallo"
git config --global user.email "MasonGallo@users.noreply.github.com"
git config --global core.editor vim

# setup R
sudo sh -c 'echo "deb http://cran.rstudio.com/bin/linux/ubuntu xenial/" >> /etc/apt/sources.list'
gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
gpg -a --export E084DAB9 | sudo apt-key add -
sudo apt-get update
sudo apt-get -y install r-base r-base-dev
sudo apt-get -y install libcurl4-gnutls-dev libxml2-dev libssl-dev
sudo apt-get -y install openjdk-8-jdk
sudo apt-get -y install r-cran-rgl libcairo-dev
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-i386
export LD_LIBRARY_PATH=$JAVA_HOME/jre/lib/i386:$JAVA_HOME/jre/lib/i386/client
sudo R CMD javareconf
echo -e "\nlocal({r <- getOption('repos')\nr['CRAN'] <- 'https://cran.rstudio.com'\noptions(repos=r)\n})" >> ~/.Rprofile

# get RStudio
sudo apt-get -y install gdebi-core
wget -O /tmp/rstudio-server-latest-amd64.deb http://www.rstudio.org/download/latest/stable/server/ubuntu64/rstudio-server-latest-amd64.deb
sudo gdebi -n /tmp/rstudio-server-latest-amd64.deb

# get Python 2.7 from conda
wget -O /tmp/conda-python.sh https://repo.continuum.io/archive/Anaconda2-4.2.0-MacOSX-x86_64.sh
bash conda-python.sh
