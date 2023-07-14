# New pipeline notes

Current notes are for researching possible solutions to the problem of achieving near 0 mAP (using torchmetrics) on the new pipeline, as opposed to the accuracy of ~10% on the old pipeline. 

Current attempted fixes:


- train.py
    1. Setting ALPHA to 1 instead of 2 (based on old branch which does not use ALPHA for calculating total loss)
    2. Setting learning rate to 1e-3 as opposed to previous 1e-1 (also based on old branch)
    3. Setting batch size to 128 instead of 32 (idem)

- models.py
    1. Remove usage of clamp for calculating location loss, previously using 1 as minimum for `number_of_positives`
    2. Use `torch.no_grad()` when performing `compute()['map']` for calculating mAP during logging. 
        - calculating the mAP was also moved outside of self.log as to avoid performing the logging with `torch.no_grad()` as it may cause unknown side effects

Findings:

- Performing all fixes all at once caused a significant increase in mAP, reaching 2.4% mAP after 20 epochs as opposed to ~2e-4 on several earlier runs

- Scaling back the fixes and only using a batch size of 32 and learning rate of 1e-3 resulted in a similarly increasing mAP after 5 epochs.

- Training using 128 epochs and a learning rate of 1e-3 


TODO: 

1. Examine why training on GPU takes just as long as on a laptop CPU
2. Fix several tensors requiring manual conversion to cuda instead of getting automatically converted