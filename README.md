# RGLC
This is the code implementation of the method from the paper "Rethinking the impact of noisy labels in graph classification: A utility and privacy perspective."
[arXiv:2406.07314](https://arxiv.org/abs/2406.07314)  

# **Requirements**   

Hardware environment: NVIDIA Tesla V100-SXM2-32GB chip with 32GB memory.  
Software environment: python 3.8, CUDA 11.3, Ubuntu 18.04.6 and PyTorch 1.12.1


**Training**  

Here, we take NCI1 as an example: 
```python
python RgLc.py --dataset NCI1 --fold_idx 0 --ptb 0.3
