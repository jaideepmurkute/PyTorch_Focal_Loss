# PyTorch_Focal_Loss

Focal Loss implementation: focal_loss.py <br/>

```python
targets = torch.Tensor([0, 1, 0, 1, 1])
output_probs = torch.Tensor([0.2, 0.8, 0.3, 0.4])
loss = FocalLoss(args, gamma=torch.Tensor([1]), logits=False)(output_probs, targets)
```

Detailed usage example and test:  
```python
python3 focal_loss_test.py 
```
<br/>




<img src="focal_loss_plot.png" width="80%">
