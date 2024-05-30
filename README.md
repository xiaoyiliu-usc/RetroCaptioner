# RetroCaptioner

## Title
RetroCaptioner：RetroCaptioner: Beyond Attention in End-to-end Retrosynthesis Transformer via Contrastively Captioned Learnable Graph Representation

## Abstract
In this work, we present RetroCaptioner, a novel end-to-end framework for one-step retrosynthesis that combines the power of a graph encoder, which integrates learnable structural information, with the capability to sequentially translate drugs, thereby efficiently capturing chemically plausible information. 

In the application of drug synthesis route planning, RetroCaptioner identifies shortened and optimal pathways that accurately correspond to established reactions, offering valuable insights for reliable and high-quality organic synthesis in drug discovery.
![image](model.png)

## Setup
RetroCaptioner requires anaconda with Python 3.7 or later, cudatoolkit=10.2.

Sugguest to install RetroCaptioner in a virtual environment to prevent conflicting dependencies.
```
conda create -n RetroCaptioner python==3.7
conda activate RetroCaptioner
conda install --yes --file requirements.txt
CUDA-enabled device for GPU training (optional)
```

## Training the model

To train the model, you will use the `train.py` script. This script accepts several command-line arguments to customize the training process.

## Command-Line Arguments:
* `--known_class`: Indicates whether the class is known (True) or unknown (False).
* `--checkpoint_dir`: Specifies the directory where the model checkpoints will be saved.
* `--device`: Designates the training device, either cuda:0 for GPU or cpu for CPU.

## Training command:

Run the following command to start the training process:

``` bash
$ python train_singleGPU.py --known_class 'True' --checkpoint_dir 'checkpoint' --device 'cuda:0'
```
Replace the argument values as per your requirements. For instance, use `--device`  cpu if you're training on a CPU.

## Multi-GPU Training command：

To run the training on multiple GPUs, use the following command:

``` bash
$ CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 train_multiGPU.py
```

* `CUDA_VISIBLE_DEVICES=0,2`: This specifies the GPU devices (in this case, GPUs 0 and 2) on which the training will be run.
* `--nproc_per_node=2`: This indicates the total number of GPUs to be used (2 GPUs in this example).
* `train_multiGPU.py`: This is the training script.

You can also add additional parameters to train_multiGPU.py to adjust whether you are running known reaction types or unknown reaction types. The parameters are the same as those used in the single GPU setup.

## Validating the model

After training, you can validate the model's accuracy using the `translate.py` script on testing set.
* `--known_class`: As in the training step, this indicates whether the class is known or unknown.
* `--checkpoint_dir`: The directory where your trained model checkpoints are stored.
* `--checkpoint`: The specific checkpoint file to use for validation. Replace `{training_step}` with the appropriate training step number. We provide a example checkpoint trained on uspto50k datasets, which takes about 10 days on one V100 GPUs. You can download the checkpoint [here](https://drive.google.com/drive/folders/12gNpyfM6zZJlaoHsL-2-Jwmt3qoU1_om?usp=sharing).
* `--device`: The device to run the validation on, either GPU (cuda:0) or CPU (cpu).

``` bash
$ python translate.py --known_class 'False' --checkpoint_dir 'checkpoint' --checkpoint 'a_model_{training_step}.pt' --device 'cuda:0'
```

## Perform the retrosynthesis step
After the training is completed, you can run the inference.py for one-step retrosynthesis prediction
* `--beam_size`: The top k predictions for a molecule 
``` bash
$ python inference.py --smiles 'Clc1cc(Cl)c(CBr)cn1' --beam_size 10 --checkpoint_dir 'checkpoint' --checkpoint 'unknown_model.pt'
```


## Planning

Planning code for multi-step planning has been placed in `api_for_multistep.py`

Here is a example for running a multi-step planning for a molecule
``` bash
 $ python api_for_multistep.py --smiles 'CCOC(=O)c1nc(N2CC[C@H](NC(=O)c3nc(C(F)(F)F)c(CC)[nH]3)[C@H](OC)C2)sc1C' --checkpoint_dir 'checkpoint' --checkpoint 'unknown_model.pt'
``` 

The key functions are shown in the following block, You may run your own molecule by changing the `args.smiles` in `api_for_multistep.py`.

Where `args` are the the parameters required for multi-step retrosynthesis. The default parameter settings are in file `./retro_star/common/parse_args.py`
``` python
planner = RSPlanner(
    gpu=args.gpu,
    use_value_fn=args.use_value_fn,
    model_dump = model_path,
    iterations=args.iterations,
    expansion_topk=args.expansion_topk,
    viz=args.viz,
)
result = planner.plan(args.smiles)
print(result)
``` 
