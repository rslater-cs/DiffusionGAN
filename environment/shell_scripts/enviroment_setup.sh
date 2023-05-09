wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p miniconda3 -s -u
rm Miniconda3-latest-Linux-x86_64.sh
conda=miniconda3/bin/conda
$conda env create -f '/vol/research/COM3025Team5/DiffusionGAN/enviroment/ubuntu_environment.yml'