from functionax.domains import LShape
from jax import random

from matplotlib import pyplot as plt

hbb = LShape()
X = hbb.sample_uniform(random.PRNGKey(0), N=1000)
print(X)
plt.scatter(X[:,0], X[:,1])
plt.show()


