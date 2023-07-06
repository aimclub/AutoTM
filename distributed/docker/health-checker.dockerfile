FROM boxboat/kubectl:1.21.3

WORKDIR /
USER root:root

RUN apk add perl

ENTRYPOINT [ "/usr/local/bin/kubectl"]
