in this repo, i will focus on distributed training with torch(maybe jax later) and try to scale it

i will start from training a simple neural network
- with only cpu
- with single gpu
- with double gpu

> i will use kaggle and colab gpus for this, because gpu poor :(

note: i have trained gpt2 with help of karpathy :), but i want to master training

okay so there are two types of parallelism: 
1. data parallelism: when you train the same model with differnet sets of gpus in different models 
2. model parallelism: when your model can not fit on a single gpu you distribute it and share it on differnet gpus 

still, 
* all reduce will sync gradients -> data parallelism 
* reduce scatter will sync gradients -> model parallelism 

something to remember:
	•	Can your model fit on one GPU? → use DDP
	•	If not? → use FSDP or pipeline/tensor parallelism


**results** of comparision using `train_ddp_comp.py`:

note: it is for 10 epochs
**cpu** -> 60.38 seconds
**gpu[Tesla T4]** -> 59.48 seconds(not too less idk why)
**ddp(2 gpus[Tesla T4])** -> 39.79


**simple neural net(`simple nn`)**

- `train_ddp_comp.py` -> it trains the net on cpu, single gpu, and two gpus(using ddp)

*commands*:
```python

#for cpu 
python train_all_modes.py --mode cpu

#for gpu 
python train_all_modes.py --mode gpu

#for ddp 
torchrun --nproc_per_node=<N(GPUS)> train_all_modes.py --mode ddp
```
- `train_fsdp.py` -> it trains the net on differnet gpus 
```python
# to run fully sharded data parallel
python train_fsdp.py
```




todo:
- [ ] implement transformer with distributed training

