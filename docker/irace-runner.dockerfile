FROM ubuntu:20.04

# update indices
RUN apt update -qq
# install two helper packages we need
RUN apt install -y --no-install-recommends software-properties-common dirmngr gpg-agent
# import the signing key (by Michael Rutter) for these repo
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
# add the R 4.0 repo from CRAN -- adjust 'focal' to 'groovy' or 'bionic' as needed
RUN add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
RUN apt install -y --no-install-recommends r-base

RUN mkdir /var/lib/irace

ENV IRACE_HOME=/usr/local/lib/R/site-library/irace

COPY src /app

ENTRYPOINT ["/app/run-irace.sh"]
