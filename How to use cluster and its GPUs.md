# How to use cluster and its GPUs

Created: June 3, 2020 9:43 AM

Created by: Minghua Zheng

Last edited time: July 06, 2023 21:03 PM

# Account creation
You need to email helpdesk (helpdesk@herts.ac.uk) with `UHHPC` in the subject so that your ticket can be redirected to the UHHPC (cluster) admin.
Ask the admin to change the default shell to `bash` when creating your account. This will make your life a lot easier. Otherwise, the default shell in cluster, which is `T-C shell` used to serve `Starlink` users, will make you suffer.

# Get yourself familiar with linux
You do not have to understand everything before using the cluster. However, a quick read of this [tutorial](https://www.digitalocean.com/community/tutorials/an-introduction-to-linux-basics) can boost your productivity if it matters.

# Configure your own development environment
## Set up shell
Again, contact the admin to change your default shell to `bash` if you have not.
The example code in this document is based on bash shell.

## Set up python env 
### Option 1: use miniconda or conda
```
wget 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh'
sh Miniconda3-latest-Linux-x86_64.sh
```

Then create a new env.
```bash
conda create -n myenv python=3.9
conda activate myenv
pip3 install --upgrade pip
pip3 install --upgrade setuptools
pip install --no-cache-dir wheel
pip install --no-cache-dir tensorflow-gpu
```

You may encounter some permission issues when installing pip packages. Google can easily give you a solution.

### Option 2: use venv 
```bash
# create a new environment
python3 -m venv .env

# activate it, use .env/bin/activate.csh if you have not changed the default shell
source .env/bin/activate

# install tensorflow gpu
pip3 install --upgrade pip
pip3 install --upgrade setuptools
pip install --no-cache-dir wheel
pip install --no-cache-dir tensorflow-gpu
```

Then
```bash
source ~/.bashrc
```

### Optional: install Pytorch
Use pip to install `Pytorch` if you prefer it over Tensorflow. Try code below and deal with permission issues if any.
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


## Set up CUDA related stuff
Add code below to your `~/.bashrc`. It loads the CUDA module automatically for you.
```bash
module load cuda-11.4
```
Then
```bash
source ~/.bashrc
```

# Use GPU node Interactively
Run code below to request an interactive job.
```bash
# wall time means how long you want to use it
# nodes=1:ppn=16 means you use request **the whole node** to run your job
# -I means it is interactive mode
# -q gpu means you request gpu node
qsub -l walltime=00:05:00 -l nodes=1:ppn=16 -I -q gpu
```

You can also specify a different GPU node to use by
```
qsub -l walltime=00:05:00 -l nodes=gpu2:ppn=16 -I -q gpu
qsub -l walltime=00:05:00 -l nodes=gpu3:ppn=16 -I -q gpu
qsub -l walltime=00:05:00 -l nodes=gpu4:ppn=16 -I -q gpu
```

Now load CUDA libraries if you have not added it to your `.bashrc`.
```bash
module load cuda-11.4
```

Now test it in python.
```python
import tensorflow as tf

tf.print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```

You can use `screen` or `tmux` inside the shell to check VRAM usage in real time with code below. If VRAM does not change during code execution, it means your program is running on CPU rather than GPU.
```
watch -n 1 nvidia-smi
```

Because there is no system to prevent other users from using the same GPU, you may need to request the whole node to prevent any unexpected issues. But, please use it **fairly** since other users may need computing resources as well.

A big problem of interactive mode is once your internet is lost or your computer goes to sleep, all jobs will get terminated. Batch queue mode is recommended.


# Use GPU node through batch queue
Create a script called `jobs.sh` with code below. Update it to meet your needs.
```bash
#!/bin/tcsh -f
#PBS -N gpu_job
#PBS -m abe
#PBS -l walltime=00:50:00
#PBS -l nodes=1:ppn=16
#PBS -k oe
#PBS -q gpu

echo ------------------------------------------------------
echo -n 'Job is running on node '; cat $PBS_NODEFILE
echo ------------------------------------------------------
echo PBS: qsub is running on $PBS_O_HOST
echo PBS: originating queue is $PBS_O_QUEUE
echo PBS: executing queue is $PBS_QUEUE
echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: execution mode is $PBS_ENVIRONMENT
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo PBS: node file is $PBS_NODEFILE
echo PBS: current home directory is $PBS_O_HOME
echo PBS: PATH = $PBS_O_PATH
echo ------------------------------------------------------

source /home/ming/Repository/microbia_classification/.env/bin/activate.csh
eval `/usr/bin/modulecmd tcsh load cuda-11.4`

cd /home/ming/Repository/microbia_classification/
jupyter nbconvert --to notebook --inplace --execute microbia_classification.ipynb --ExecutePreprocessor.timeout=-1

echo ------------------------------------------------------
echo Job ends
```

`#PBS -N gpu_job`: gpu_job is the name of the job shows in the system, you can change it to avoid confusion

`#PBS -l walltime=00:50:00`: only 5 mins are requested here.

`jupyter nbconvert --to notebook --inplace --execute microbia_classification.ipynb --ExecutePreprocessor.timeout=-1`

This will run all cells in the jupyter notebook. If code execution takes too long, the output of some cells may get lost. A good solution is to use python logging. Alternatively, avoid jupyter notebook in the first place.

Now submit it by
```bash
qsub jobs.sh
```


# Use Jupyter Notebook interactively and remotely
If you request an interactive job, you can run code below directly.
```bash
jupyter notebook --no-browser --port=8889 --ip=gpu2.data
```
Note the `--ip=gpu2.data` must be the hostname + `.data`. Possible hostname for gpu nodes are `gpu2`, `gpu3` and `gpu4` (Unfortunately, `gpu1` is no longer available). If you do not know what the hostname is, run `echo $HOSTNAME`. Change the port number if it is already in use.

If you request a job through batch queue, add code below to your script.
```bash
jupyter notebook --no-browser --port=8889 --ip=`hostname`.data
```

Next, in your local machine, execute code below in the terminal (for macOS and linux users only). Do not forget to update `USERNAME` and the hostname which is `gpu2` in this example.
```bash
ssh -N -f -L localhost:8889:gpu2.data:8889 -i USERNAME@uhhpc.herts.ac.uk
```

Then, visit `localhost:8889` in the browser.

You can find jupyter token in a file called `job_name.e_job_id` from your home directory.

# Use IDE in the cluster
In theory, you are not allowed to run heavy program in the headnode. However, sometimes you want to debug your large machine learning models, such as, to identify the performance bottleneck or check data input pipeline, create breakpoints to check variables, etc, it is in fact doable.

You should follow procedures below at your own risk (may breach rules and get your cluster account revoked). And it does not work if you have an Apple Silicon based Mac (because Java does not behave properly for Xquartz on these Macs). I strongly suggest you contact the admin to help you first because they may have some machines hidden somewhere to meet some special requests.

## Set up x2go client in your local machine
Follow instructions [here](https://wiki.x2go.org/doku.php/doc:installation:x2goclient) to install x2go client.

Create a new session in x2go client, host is `uhhpc.herts.ac.uk`, login is your cluster username. Use `xfce` for the session type.

## Set up Pycharm IDE
You can follow [link](https://www.jetbrains.com/help/pycharm/installation-guide.html#f880e837) to install Pycharm in the headnode. Firefox can be launched to activate Pycharm.

Launch Pycharm in the desktop environment x2goclient just rendered for you, then in the `Remote development` section, create a ssh connection to the gpu node. Then your python code in Pycharm will run on the (remote) gpu node. Meanwhile, Pycharm is running on the headnode. 

Submit an interactive gpu job to fool the admin. Again, use it at your own risk.
