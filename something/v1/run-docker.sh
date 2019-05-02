#!/bin/bash
nvidia-docker run -it --shm-size 64G -v /rscratch:/rscratch pytorch/pytorch

apt-get update
apt-get install -y vim
apt-get install -y nano
apt-get install -y tmux
apt-get install -y psmisc
apt-get install -y lsof

pip install gpustat


cd ~
ln -s /rscratch/zhendong rscratch-zhen 
ln -s /rscratch/data Datasets

cd rscratch-zhen/video-acc
