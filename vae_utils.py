import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

L = 10

def get_mc_samples(dec_model, enc_model, x, samples=10):
    """Get the monte carlo samples for E_q(V)"""
    global L
    L = samples

    z_mean, log_z_var = enc_model.predict(x)

    z_std = np.exp(log_z_var/2)
    shape = z_std.shape

    x_preds = []
    z_samples = []

    for i in range(L):
        #epsilon = np.random.multivariate_normal(np.zeros(shape[-1]), np.eye(shape[-1]), size=shape[0])
        epsilon = np.random.normal(size=shape)
        z_sample = z_mean + z_std * epsilon
        z_samples.append(z_sample)

        x_pred = dec_model.predict(z_sample)
        x_preds.append(x_pred)
        
    if L == 0:
        x_preds = [dec_model.predict(z_mean)]
        L = 1
        
    x_preds = np.array(x_preds)
    z_samples = np.array(z_samples)

    return x_preds, z_samples

def elbo(y_true, y_pred, enc_model):
    z_mean, log_z_var = enc_model.predict(y_true)
    kl_loss = 1/2 * np.sum(1 + log_z_var - np.square(z_mean) - np.exp(log_z_var), axis=-1)
    im_loss = 1/L * np.sum(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred), axis=(0, -1, -2))
    return kl_loss + im_loss

def reconstruct(image, enc_dec_model):
    assert image.shape == (28, 28)
    out = enc_dec_model.predict_on_batch(np.expand_dims(image, axis=0))
    return out.reshape((28, 28))

def get_image_pairs(x, y, df_quant, model):
    image_pairs = []

    y_df = pd.DataFrame(y, columns=['label'])


    for num in range(10):
        idx = y_df[y_df['label'] == num].index
        df_nums = df_quant.iloc[idx]
        
        # find the min/max indices
        idx_min = df_nums.idxmin()
        idx_max = df_nums.idxmax()
        
        # get the min/max (original/reconstruction) pair
        x_min_or = x[idx_min]
        x_min_re = reconstruct(x_min_or, model)
        x_min_pair = (x_min_or, x_min_re)
        
        x_max_or = x[idx_max]
        x_max_re = reconstruct(x_max_or, model)
        x_max_pair = (x_max_or, x_max_re)
        
        # to find the image with an elbo closest to the mean, we simply find
        # the image with the smallest euclidean distance from the elbo
        
        elbo_dist = (df_nums - np.mean(df_nums))**2
        mean_idx = elbo_dist.idxmin()
        
        x_mean_or = x[mean_idx]
        x_mean_re = reconstruct(x_mean_or, model)
        x_mean_pair = (x_mean_or, x_mean_re)
        
        image_pairs.append([x_min_pair, x_max_pair, x_mean_pair])
    return image_pairs

def plot_min_max_mean(quantity, image_pairs):
    fig, axes = plt.subplots(10, 3, figsize=(5, 8), dpi=100)

    titles = ('Worst %s' %quantity, 'Best %s' %quantity, 'Ave %s' %quantity)

    for num in range(10):
        axes[num, 0].set_ylabel(num)
        for j, pair in enumerate(image_pairs[num]):
            cat = np.concatenate(pair, axis=1)
            axes[num, j].imshow(cat, cmap="Greys_r")
            
            # fix ticks/labels
            axes[0, j].set_title(titles[j])
            axes[num, j].set_xticks(())
            axes[num, j].set_yticks(())
            
    # save the image
    plt.show()

    return fig, axes

def get_z_min_max(x, y, df_elbo, df_std, enc_model, top=10):
    """Select the top elbo/std from our latent space."""
    y_df = pd.DataFrame(y, columns=['label'])

    z_elbo_min = dict()
    z_elbo_max = dict()

    z_std_min = dict()
    z_std_max = dict()

    z_mean, log_z_var = enc_model.predict(x)

    for num in range(10):
        idx = y_df[y_df['label'] == num].index
        
        elbo_min_idxs = df_elbo.iloc[idx].sort_values(ascending=True).head(top).index.tolist()
        elbo_max_idxs = df_elbo.iloc[idx].sort_values(ascending=False).head(top).index.tolist()
        
        z_elbo_min[num] = z_mean[elbo_min_idxs]
        z_elbo_max[num] = z_mean[elbo_max_idxs]
        
        std_min_idxs = df_std.iloc[idx].sort_values(ascending=True).head(top).index.tolist()
        std_max_idxs = df_std.iloc[idx].sort_values(ascending=False).head(top).index.tolist()
        
        z_std_min[num] = z_mean[std_min_idxs]
        z_std_max[num] = z_mean[std_max_idxs]

    return (z_elbo_min, z_elbo_max), (z_std_min, z_std_max)

def calc_V(enc_model, y_true, y_pred, z_samples):
    z_mean, log_z_var = enc_model.predict(y_true)
    z_var = np.exp(log_z_var)
    # the value H - Hq from Ulrichs notes
    # returns an L x T with L as the sample size and T the length of the data
    im_log = np.sum(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred), axis=(-1, -2)) #log p(x|z)
    kl_log = 1/2 * np.sum(1/z_var*(z_samples-z_mean)**2 + log_z_var - z_samples**2, axis=-1) # log (q(z|x)/p(z))
    return im_log + kl_log

def expectation(freq):
    return 1/L * np.sum(freq, axis=0)