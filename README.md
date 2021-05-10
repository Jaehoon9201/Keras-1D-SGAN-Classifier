## Keras-1D-VAE-Anomaly-Detection
This code is based on the code from the following three sites.

[Reference 1](https://keras.io/getting_started/intro_to_keras_for_researchers/) / 
[Reference 2](https://www.linkedin.com/pulse/supervised-variational-autoencoder-code-included-ibrahim-sobh-phd)


This code is an example of an 1-D data anomaly detection with a VAE model. 

If you run this code, you could get a similar result below one.

## Results
### Dataset Distribution in a Latent Space
![KakaoTalk_20210510_175811254](https://user-images.githubusercontent.com/71545160/117639279-b7c55e80-b1be-11eb-8858-2a79b498a5dd.png)

### AUROC(area under the receiver operating characteristic)
![KakaoTalk_20210510_175817096](https://user-images.githubusercontent.com/71545160/117639276-b6943180-b1be-11eb-9ad4-4f04d18c3e34.png)

## If you want to see a change in the results
You can change the following settings if you want to see a change in the results.
```python
# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(0.5* xent_loss + 0.5*kl_loss)
```
On the above code, vae_loss 
