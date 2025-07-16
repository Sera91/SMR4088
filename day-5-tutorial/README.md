# Train a Decoder-only Model for Machine Translation

In this guide we will go through the process of training a small language model (SLM) for a machine translation (MT) task.
It is different from the classic MT models because it uses a decoder only architecture like GPT.

We will go through the full process of data preparation, model setup and pretraining. The library used for training is [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/main) a research library developed by NVIDIA to train transformers models at scale.

### Set up the Singularity Image

At this point we assume you logged in the cluster successfully and your are in your `$HOME` directory.

First, go to your `$SCRATCH` folder and create a project folder.
```bash
cd $SCRATCH
mkdir -p slm_mt
```
We also need a `cache` and `tmp` folders to use for the singularity image cache and temp files.
```bash
mkdir -p cache
mkdir -p tmp
```

The reccomended way to run Megatron-LM is with a docker container from [NGC](https://catalog.ngc.nvidia.com/), where you can find highly optimized pre-built containers, allowing you to run as maximum speed on NVIDIA hardware.
On CINECA's Leonardo you can run singularity images but not docker containers. Thus you need to download the docker container and convert it to a singularity image.

The docker conversion can take some time, it is good practice to do that in a node that has access to internet but not on the login node, it would overload the login node if we all start doing operations on it. Furthermore, to ensure people do not misuse the login nodes CINECA kills any process that takes more than 10 minutes on the login nodes. We then need to connect to a node with internet access that we can use to perform long tasks, we'll use an interactive session for ease of use:
```bash
srun --time 04:00:00 --nodes 1 --ntasks=1 --cpus-per-task=4 --partition=lrd_all_serial -A <your-project-account> --pty /bin/bash
```
Then when resources are allocated we can again make sure we are in the right folder and convert docker to singularity image.
You will see that the job is allocated resources because the node name changes, before you are on a node like `login01-03-05-07` then you will see `login08` for example
```bash
# go to the project folder
cd $SCRATCH/slm_mt
# set cache and tmp dirs for singualarity
export SINGULARITY_CACHEDIR="$SCRATCH/cache"
export SINGULARITY_TMPDIR="$SCRATCH/tmp"
# pull the container, it will convert it to a .sif image
singularity pull docker://nvcr.io/nvidia/pytorch:25.06-py3
ls *.sif
```
Should be something like this `pytorch_25.06-py3.sif`.
Now you can exit the interactive session by simply typing 
```bash
exit
```
You should see that the node name changes again back to the first one you had.

This should have created a singularity image for us, just to check it works we can send a job to the cluster and see the output
```bash
# go to the project folder
cd $SCRATCH/slm_mt
# auto read the singularity image name, if you have more than one set it manually
SINGULARITY_IMG="*.sif"
# run a simple check
srun --time 00:00:10 --nodes 1 --ntasks=1 --cpus-per-task=1 --gres=gpu:4 --partition=boost_usr_prod -A <your-project-account> --pty singularity exec --nv $SINGULARITY_IMG nvidia-smi
```
This will print out a nvidia-smi output
```bash
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 530.30.02              Driver Version: 530.30.02    CUDA Version: 12.1     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM-64GB            On | 00000000:1D:00.0 Off |                    0 |
| N/A   43C    P0               63W / 479W|      0MiB / 65536MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-SXM-64GB            On | 00000000:56:00.0 Off |                    0 |
| N/A   42C    P0               64W / 472W|      0MiB / 65536MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA A100-SXM-64GB            On | 00000000:8F:00.0 Off |                    0 |
| N/A   43C    P0               62W / 469W|      0MiB / 65536MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA A100-SXM-64GB            On | 00000000:C8:00.0 Off |                    0 |
| N/A   42C    P0               60W / 458W|      0MiB / 65536MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```
We can take the chance to observe that Leonardo has 4 GPUs per node and they are A100 SXM with 64GB of vRAM.

### Setup Megatron-LM

Now that we have the singularity image converted we can start preparing the code.
Clone Megatron-LM
```bash
# go to the project folder
cd $SCRATCH/slm_mt
# clone the repo
git clone https://github.com/NVIDIA/Megatron-LM.git && cd Megatron-LM
```

In my latest experiments I had some issues so please go to the files below and comment some lines as showed below. Use the editor you prefer, either VSCode via remote ssh connection or vim/nano directly in bash.
 - `megatron/training/training.py`, comment lines 80 and 81
```bash
# from megatron.core.transformer.moe import upcycling_utils
# from megatron.core.transformer.moe.moe_utils import track_moe_metrics
```
 - `megatron/core/models/gpt/gpt_layer_specs.py`, comment line 8
 ```bash
 # from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
 ```
This should now be enough to have Megatron-LM up and running for our use case.

### Data Preparation

We now move on to prepare the data needed for the training. We will provide a pre-cleaned dataset derived from [OPUS](https://aclanthology.org/2016.eamt-2.8/) with Italian and English sentence pairs.

To download the dataset you can go to this link
```bash
# go to the project folder
cd $SCRATCH/slm_mt
# create folder
mkdir -p data
mkdir -p data/raw
cd data/raw
# move data here
cp /leonardo/pub/userexternal/apilzer0/data/. .
# unzip
gzip -cd allfiltered.en-it.it-en.shuf-000.txt.gz >> allfiltered_merged_raw.txt

# go back to project folder
cd $SCRATCH/slm_mt
```

> [!IMPORTANT]  
> In the event we can't access allfiltered data there is a small backup in github [here](https://github.com/andrea-pilzer/icpt/blob/main/first_500k.txt.gz).
> To download the command should be `wget https://raw.githubusercontent.com/andrea-pilzer/icpt/blob/main/first_500k.txt.gz`.
> To make your life easier rename `first_500k.txt.gz` to `allfiltered.en-it.it-en.shuf-000.txt.gz` so you don't need to change all the rest of the commands.

The dataset is a simple text file with all the senteces pairs in a very simple format
```
LANG_1 sentence LANG_2 sentece
 # or
LANG_2 sentence LANG_1 sentece
```
where LANG_1=ITA and LANG_2=ENG in our case and the two senteces are the tranlations in the other language.
You can see an example if you run the `tail` command on the file
```bash
tail data/raw/allfiltered_merged_raw.txt
```

Megatron-LM uses a specialized data format for efficient large-scale language model training. The core of this format consists of two files per dataset:
 - `.bin` file: Stores the actual tokenized data (sequences/documents).
 - `.idx` file: Stores metadata and indexing information for fast, random access to the data in the `.bin` file.
This structure is designed for high performance, scalability, and compatibility with distributed training.

Before getting to this we need to convert our dataset in `json` for the data preprocessing script to work.
A very basic script example is the following
```python
# copy the text and put it in a file txt2json.py
import json

input_file = "/workspace/data/raw/allfiltered_merged_raw.txt"
output_file = "/workspace/data/allfiltered_merged.jsonl"

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        line = line.strip()
        if line:
            #fout.write(f'{{"text": "{line}"}}\n')
            fout.write(json.dumps({"text": line}, ensure_ascii=False) + "\n")
```
you can copy the text above and paste it in a file, call it something easy to remember like `txt2json.py`
Then you can run it like this
```bash
srun --time 01:00:00 --nodes 1 --ntasks=1 --cpus-per-task=4 --partition=lrd_all_serial -A <your-project-account> --pty singularity exec --pwd /workspace --nv -B "$SCRATCH/slm_mt/:/workspace" $SINGULARITY_IMG bash -c "python txt2json.py"
```
If we decompose the command we wrote it means:
 - `--time 01:00:00` allocate 1 hour
 - `--pwd /workspace` start singularity at this path
 - `-B "$SCRATCH/slm_mt/:/workspace"` mount our project folder at path `/workspace`
 - `bash -c "python txt2json.py"` run this script
It should be fairly quick, jsonl is useful so each of our documents is on one line, in our case we have sentence pairs (ITA/ENG) not documents.

Again if you like you can run a `tail data/allfiltered_merged.jsonl` to check the result.

Now we are ready to use the `tools/preprocess_data.py` script to tokenize the text and prepare it for training.
First we need to download the tokenizer, we chose to use the one trained by the Minerva project, a model developed by Sapienza University of Rome and trained mainly for Italian and English. We expect the tokenizer of this model to work well on our task.
```bash
# go to the project folder
cd $SCRATCH/slm_mt
# load module of cineca to have huggingface-cli
module load profile/deeplrn cineca-ai
huggingface-cli download sapienzanlp/Minerva-7B-instruct-v1.0 --local-dir Minerva-7B-instruct-v1.0
module purge
```

Create a script `data_preprocess.sh` with the following commands
```bash
# use this after converting the data to jsonl format
# for the conversion to jsonl format, use the above script:
# python txt2json.py

# make sure that args.workers % args.partitions == 0

#!/bin/bash
INPUT_FILE="/workspace/data/allfiltered_merged.jsonl"
OUTPUT_FILE="/workspace/data/allfiltered_merged"
TOKENIZER_MODEL="/workspace/Minerva-7B-instruct-v1.0"

cd Megatron-LM

python tools/preprocess_data.py \
    --input $INPUT_FILE \
    --json-keys "text" \
    --output-prefix $OUTPUT_FILE \
    --workers 32 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --partitions 32 \
    --append-eod
```
Then run the conversion on a GPU node (this should take few minutes)
```bash
srun --time 02:00:00 --nodes 1 --ntasks=1 --cpus-per-task=32 --gres=gpu:4 --mem=0 --exclusive --partition=boost_usr_prod -A <your-project-account> --pty singularity exec --pwd /workspace --nv -B "$SCRATCH/slm_mt/:/workspace" $SINGULARITY_IMG bash -c "bash data_preprocess.sh"
```

### Run Experiments

Now we are almost ready for training with code, data and singularity image ready. One thing to do is to create a data cache where Megatron-LM will save the data mapping files used for different distributed training configurations.
```bash
# go to the project folder
cd $SCRATCH/slm_mt
# create folder
mkdir -p data_cache_path
```
Make sure that the `data_cache_path` is in folder where the singularity image has read/write permission (in our case `$SCRATCH` is fine).

Then to run the training we need to write a slurm script, you can copy paste the following one in a file `run_gpt_mt_singularity.sh`
```bash
#!/bin/bash
#SBATCH --account=<your-project-account>
##SBATCH --reservation=<your-reservation> # if you have some nodes reserved for your school
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=megatron-gpt-mt
#SBATCH --exclusive
##SBATCH --qos=boost_qos_dbg # if you need to run debug, max 30 mins 2 nodes
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclude=lrdn[1261-3456]

echo $SLURM_NODELIST

# for fsdp larger than 1 comment max connections
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
echo "SLURM_NTASKS="$SLURM_NTASKS
NTASKS_PER_NODE=$((SLURM_NTASKS / SLURM_JOB_NUM_NODES))
echo "NTASKS_PER_NODE="$NTASKS_PER_NODE
export WORLD_SIZE=$((GPUS_PER_NODE * SLURM_NNODES))
echo "WORLD_SIZE=$WORLD_SIZE"
MASTER_PORT=11111

NUM_NODES=$SLURM_NNODES
NODE_RANK=0

export WORLD_SIZE=$((GPUS_PER_NODE * SLURM_NNODES))
echo "WORLD_SIZE=$WORLD_SIZE"

# these two are needed at build time not at exec time
# export SINGULARITY_CACHEDIR="$SCRATCH/cache"
# export SINGULARITY_TMPDIR="$SCRATCH/tmp" 
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export WANDB_MODE=offline

export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

CHECKPOINT_SAVE="$SCRATCH/slm_mt/checkpoints" 
TENSORBOARD_LOGS_PATH="$SCRATCH/slm_mt/tensorboard-logs" #$2 #<Specify p$

DATA_PATH="$SCRATCH/slm_mt/data"
DATA_PATH_SINGULARITY="/workspace/data"
DATA_FILES=$DATA_PATH/allfiltered_merged_text_document
DATA_FILES_SINGULARITY=$DATA_PATH_SINGULARITY/allfiltered_merged_text_document

TOKENIZER_ARG="/workspace/Minerva-7B-instruct-v1.0"
# the data cache must be writable
# I put my scratch folder
DATA_CACHE_PATH="$SCRATCH/slm_mt/data_cache_path"
DATA_CACHE_PATH_SINGULARITY="/workspace/data_cache_path"

MICRO_BATCH=32
BATCH=$((MICRO_BATCH * WORLD_SIZE))

options="--tensor-model-parallel-size 1 \
--pipeline-model-parallel-size 1 \
--num-layers 6 \
--hidden-size 2048 \
--num-attention-heads 32 \
--seq-length 1024 \
--max-position-embeddings 4096 \
--attention-backend auto \
--micro-batch-size $MICRO_BATCH \
--global-batch-size $BATCH \
--train-iters 1000 \
--weight-decay 0.1 \
--adam-beta1 0.9 \
--adam-beta2 0.95 \
--init-method-std 0.006 \
--clip-grad 1.0 \
--fp16 \
--lr 1.0e-5 \
--lr-decay-style cosine \
--min-lr 6.0e-6 \
--lr-warmup-fraction .001 \
--lr-decay-iters 430000 \
"

# Note: --vocab-size might be inferred by HuggingFaceTokenizer or might need to be explicit.
DATA_ARGS_LIST="--data-path $DATA_FILES_SINGULARITY \
--tokenizer-type HuggingFaceTokenizer \
--tokenizer-model $TOKENIZER_ARG \
--data-cache-path $DATA_CACHE_PATH_SINGULARITY \
--object-storage-cache-path $DATA_CACHE_PATH_SINGULARITY \
--split '99,1,0' \
--no-create-attention-mask-in-dataloader \
--num-workers 2 \
--dataloader-type cyclic \
--vocab-size 51200 \
"
# --no-mmap-bin-files \

EVAL_AND_LOGGING_ARGS=(
--log-interval 100
--save-interval 10000 
--eval-interval 1000 
--save $CHECKPOINT_PATH 
--load $CHECKPOINT_PATH 
--eval-iters 10
--tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

echo "${DISTRIBUTED_ARGS}"
echo "${options}"
echo "${DATA_ARGS_LIST}"

#--- head_node_ip ---
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip
#--------------------  --rdzv_backend=c10d --rdzv_endpoint $head_node_ip:11111
DISTRIBUTED_ARGS="--nproc-per-node $GPUS_PER_NODE \
--nnodes $NUM_NODES \
--rdzv_backend=c10d \
--rdzv_endpoint $head_node_ip:11111 \
"

SINGULARITY_IMG="$SCRATCH/slm_mt/*.sif"

echo "starting training"

srun singularity exec --nv --pwd /workspace/Megatron-LM \
  -B "/leonardo_scratch/large/userexternal/apilzer0/raganato/:/workspace, ${DATA_PATH}/:${DATA_PATH_SINGULARITY}" \
  $SINGULARITY_IMG \
  bash -c "ls -ld ${DATA_PATH_SINGULARITY}; ls -ld ${DATA_CACHE_PATH_SINGULARITY}; torchrun ${DISTRIBUTED_ARGS} /workspace/Megatron-LM/pretrain_gpt.py ${options} ${DATA_ARGS_LIST} --log-throughput"

echo "end training"

```
On this script we can comment more things, for example you can see all the preparation for a potentially multinode run, a lot of arguments we pass for the training and then the actual singularity command to run the script.

To execute it
```bash
sbatch run_gpt_mt_singularity.sh
```

Try out some settings:
 - could you make it run on 1 GPU only?
 - could you make it run on 1 node?
 - could you make it run on multiple nodes?
 - could you check different batch sizes?
 - what about the scalability of the code? How does throughput change with increasing number of nodes? Think what it means to keep the model size the same and processing larger batches.
 - if you check well I did not properly set the logging part, if you want to customize, please go ahead :)
 - CINECA's Leonardo has different [storage partitions](https://docs.hpc.cineca.it/hpc/hpc_data_storage.html#file-systems-and-data-management), what do you think would happen if you store the data on the `$FAST` partition. Run a training and check the throughput.

### Wrap Up

We saw all the steps that allow us to train a model. In particular a SLM for MT with 6 layers and basic training settings.
If you want to explore more you can check out additional args to pass to the script and tune the training.
There is also a basic example in Megatron-LM about inference [here](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/inference).

# Appendix

If you want to use wandb on Leonardo you need to use offline mode, please find [here](https://docs.wandb.ai/support/run_wandb_offline/) the guidelines.


