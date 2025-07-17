# On the frontend node:

## Setup the SSH key for accessing the nodes through SSH
```
ssh-keygen -t ed25519
cd .ssh
touch authorized_keys
cat id_ed25519.pub >> authorized_keys
```

## Clone the repository and setup the Python virtual environment
```
git clone https://github.com/Sera91/SMR4088.git
cd SMR4088/Federated_Learning/xffl/
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .
```

## Prepare the first example
```
cd examples/simulation/01_simple-MLP/
pip install -r requirements.txt
```

## Get computational resources
```
srun -A tra25_ictp_scd --partition boost_usr_prod --time 01:00:00 --gpus-per-node 2 --nodes 1 --pty /bin/bash
```

# On the compute node:

## Run the example
All the arguments after "-args" are referred to the training script:
```
xffl simulate training.py -v ${HOME}/xffl/.venv -p 2 -args --seed 42 -e 3 -wb -mode offline -name MLP -oc
```
To toggle on the federation: ```-fs 1```

## Run example 2
```
cd ../02_CNN/
xffl simulate training.py -v ${HOME}/xffl/.venv -p 4 -args --seed 42 -e 10 -fs 1 -wb -mode offline -name CNN_4c_10e_oc -oc
```

## Run example 3
```
cd ../03_LLM/
xffl simulate training.py -v ${HOME}/xffl/.venv -p 4 -args -m llama3.1-8b -d clean_mc4_it --seed 42 --subsampling 128 -fs 2 -wb -mode offline -name LLM_2c_128t
```

# Synch WandB runs from the frontend:
```
cd wandb_folder
wandb sync offline-run-*
```