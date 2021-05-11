## Keras-1D-SGAN-Classifier

This code has been redesigned to fit 1-D data based on the reference right.  [Reference](https://machinelearningmastery.com/semi-supervised-generative-adversarial-network/)

### What is the SGAN(Semi-Supervised GAN)?

As you can see the results on site [Refence2](https://github.com/nejlag/Semi-Supervised-Learning-GAN#semi-supervised-learning-with-generative-adversarial-networks-gans), you can train a high-performance classifier with little labeled data.
|10% labeled data|100% labeled data|
|------|---|
|0.9255|0.945|

### sgan_flt_diagnosis.py

The following result shows the result of classifier learning for 1-D data provided with this code.

![20210511_140503](https://user-images.githubusercontent.com/71545160/117762152-8f437000-b263-11eb-9641-fc8e162e4929.png)



### sgan_flt_diagnosis_test.py

As shown in the code below, the classifier(=discriminator) model saved in **sgan_flt_diagnosis.py** can be loaded and evaluated.


The following result shows the result of classifier test for 1-D data provided with this code.

```python
# load the model
model = load_model('model/c_model_12960.h5')
```

![20210511_141118](https://user-images.githubusercontent.com/71545160/117762159-8fdc0680-b263-11eb-9a04-4352749ebe85.png)
