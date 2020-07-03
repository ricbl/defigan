source $(conda info --base)/etc/profile.d/conda.sh
conda create --name defigan -y python=3.7.3
conda activate
conda activate defigan
conda install -y numpy==1.17.0
conda install -y pandas==0.25.1
conda install -y pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
conda install -y scipy==1.3.1
conda install -y h5py==2.9.0
conda install -y nibabel==2.5.1
conda install -y scikit-image==0.15.0
conda install -y tensorboardX==1.8
conda install -y pillow==6.2.1
conda install -c anaconda -y cudnn
conda install -c conda-forge -y cudatoolkit
conda install -y tensorflow-gpu==1.15