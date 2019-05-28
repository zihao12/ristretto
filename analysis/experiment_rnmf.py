
# coding: utf-8

# ## Generate data and compute oracle loss

# In[10]:


import warnings
#warnings.filterwarnings("ignore",category =RuntimeWarning)
import numpy as np
from ristretto import nmf
from sklearn.decomposition import NMF
import time
np.random.seed(123)

## generate data
p = 5000
n = 1000
K = 5
W = np.exp(np.random.uniform(size = (p,K)))
H = np.exp(np.random.uniform(size = (K,n)))
Lam = W.dot(H)
A = np.random.poisson(Lam, size = (p,n))

# compute oracle loss
cost_oracle = nmf.cost(A, W, H)
print("oracle loss: {}".format(cost_oracle))


# ## nmf

# In[2]:


print("nmf")
start = time.time()
(W_nmf,H_nmf) = nmf.compute_nmf(A=A,rank = K, maxiter=200)
runtime = time.time() - start
print("runtime: {}".format(runtime))
print("loss: {}".format(nmf.cost(A,W_nmf,H_nmf)))


# In[3]:


print("rnmf")
start = time.time()
(W_rnmf,H_rnmf) = nmf.compute_rnmf(A=A,rank = K, maxiter=200)
runtime = time.time() - start
print("runtime: {}".format(runtime))
print("loss: {}".format(nmf.cost(A,W_rnmf,H_rnmf)))


# In[11]:

print("rnmf2")
start = time.time()
(W_rnmf2,H_rnmf2) = nmf.compute_rnmf2(A=A,rank = K, maxiter=200)
runtime = time.time() - start
print("runtime: {}".format(runtime))
print("loss: {}".format(nmf.cost(A,W_rnmf2,H_rnmf2)))


# In[12]:


