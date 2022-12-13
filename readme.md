<!-- Dependencies: -->
pytorch with cuda 
pytorch geometric
SU2 solver:
    swig --> for python wrapper
    openmpi --> mpi
pyvista --> geometry manipulation
tqdm --> progress indicator in for loop 

<!-- install commands -->
###########install pytorch and pytorch geometric########################################
#create conda enviroment
conda create -n cfd-gcn python=3.7
conda activate cfd-gcn 
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
conda install pyg -c pyg
#Install torch_geometric
env PATH=/usr/local/cuda/bin:$PATH
env CPATH=/usr/local/cuda/include:$CPATH
env LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.5.0+cu101.html
pip install torch-sparse==0.6.6 -f https://pytorch-geometric.com/whl/torch-1.5.0+cu101.html
pip install torch-cluster==1.5.5 -f https://pytorch-geometric.com/whl/torch-1.5.0+cu101.html
pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.5.0+cu101.html
pip install torch-geometric==1.6.0 -f https://pytorch-geometric.com/whl/torch-1.5.0+cu101.html

#assuming your in the current path = your-own-path
apt-get update -y && apt-get install -y openmpi-bin libopenmpi-dev swig m4
env MPICC=/usr/bin/mpicc pip install mpi4py
export PATH=/usr/local/cuda/bin:$PATH
export CPATH=/usr/local/cuda/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CXXFLAGS="-O3"
git clone --branch feature_pytorch_communicator  https://github.com/su2code/SU2 
mkdir su2install
cd SU2 
./preconfigure.py --enable-mpi --with-cc=/usr/bin/mpicc --with-cxx=/usr/bin/mpicxx --prefix=your-own-path/su2install/SU2 --enable-autodiff --enable-PY_WRAPPER --disable-tecio --update
sudo make -j 16 Install
#add this to ~/.bashrc
export PATH=$PATH:your-own-path/su2install/SU2/bin
export PYTHONPATH=$PYTHONPATH:your-own-path/su2install/SU2/bin
export SU2_RUN=your-own-path/su2install/SU2/bin
export SU2_HOME=your-own-path/SU2


<!-- run result -->

#activate the conda enviroment cfd-gcn
git clone https://github.com/shanjierenyidp/CFD_GCN.git
cd CFD-GCN
conda activate cfd-gcn
#run this to convert simulation to graphs
cd project-fold-path
python data.py --root './data/cases' -d './data/dataset0' -fm './data/aorta3.su2' -ts 2000

# run this to run the experiment
## Data:
#you need to download the data from google drive via this link, please download it and unzip it as data folder. 
https://drive.google.com/file/d/1VTnuxaQXppp2mKpLFD9KJ86RL1zCWYQf/view?usp=sharing
## experiments 
#then start runing the experiments:
mpirun -np 5 python train.py > log
#note that the np = 5 = batch_size + 1 
