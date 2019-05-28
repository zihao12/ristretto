import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)
# coding: utf-8

# In[1]:


import numpy as np
from ristretto import nmf_kl
from sklearn.decomposition import NMF
import time
np.random.seed(123)

## generate data
p = 5000
n = 2000
K = 5
W = np.exp(np.random.uniform(size = (p,K)))
H = np.exp(np.random.uniform(size = (K,n)))
Lam = W.dot(H)
A = np.random.poisson(Lam, size = (p,n))

# compute oracle loss
cost_oracle = nmf_kl.cost(A, W, H)
print("oracle loss: {}".format(cost_oracle))


print("rnmf_kl")
start = time.time()
(W_rnmfkl,H_rnmfkl) = nmf_kl.compute_rnmf_kl(A=A,rank = K, oversample = 50, maxiter = 2000)
runtime = time.time() - start
print("runtime: {}".format(runtime))
print("loss: {}".format(nmf_kl.cost(A,W_rnmfkl,H_rnmfkl)))



# print("skd.nmf")
# start = time.time()
# model = NMF(n_components=K, init = 'nndsvda', beta_loss='kullback-leibler', solver='mu', max_iter=200)
# model.fit(A)
# W_skdnmf = model.transform(A)
# H_skdnmf = model.components_
# runtime = time.time() - start
# print("runtime: {}".format(runtime))
# print("loss: {}".format(nmf_kl.cost(A,W_skdnmf,H_skdnmf)))


# In[9]:


# In[7]:


# print("nmf_kl")
# start = time.time()
# (W_nmfkl,H_nmfkl) = nmf_kl.compute_nmf_kl(A=A,rank = K, maxiter=200)
# runtime = time.time() - start
# print("runtime: {}".format(runtime))
# print("loss: {}".format(nmf_kl.cost(A,W_nmfkl,H_nmfkl)))


# In[4]:



