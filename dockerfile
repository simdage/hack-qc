FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y \
    && apt-get install alsa -y \
    && apt-get install -y python3-pyqt5 \
    && apt-get install python3-pip -y
# RUN apt-get update && apt-get install libgl1-mesa-glx --yes
RUN apt-get update
RUN apt-get install libglib2.0-0 --yes
RUN apt-get install libnss3 --yes
# RUN  localedef --no-archive -i en_US -f UTF-8 en_US.UTF-8 && \
#      export LANG=en_US.UTF-8
# Installing Necessary packages including firefox
# RUN  yum install -y dbus-x11 PackageKit-gtk3-module libcanberra-gtk2 firefox# Generating a universally unique ID for the Container
# RUN  dbus-uuidgen > /etc/machine-id
ENV QT_DEBUG_PLUGINS=1
COPY . .
RUN pip3 install -r requirements.txt
# CMD  /usr/bin/firefox
# WORKDIR ui
CMD ["python3", "ui/interface.py"]