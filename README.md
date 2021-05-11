## Keras-1D-SGAN-Classifier

This code has been redesigned to fit 1-D data based on the reference right.  [Reference](https://machinelearningmastery.com/semi-supervised-generative-adversarial-network/)

### sgan_flt_diagnosis.py

![20210511_140503](https://user-images.githubusercontent.com/71545160/117762152-8f437000-b263-11eb-9641-fc8e162e4929.png)

### sgan_flt_diagnosis_test.py

As shown in the code below, the classifier(=discriminator) model saved in **sgan_flt_diagnosis.py** can be loaded and evaluated.

```python
# load the model
model = load_model('model/c_model_12960.h5')
```

![20210511_141118](https://user-images.githubusercontent.com/71545160/117762159-8fdc0680-b263-11eb-9a04-4352749ebe85.png)
