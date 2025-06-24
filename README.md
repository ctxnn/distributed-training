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
    
todo:
- [ ] implement transformer with distributed training
