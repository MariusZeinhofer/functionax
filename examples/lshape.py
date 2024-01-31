from functionax.domains import LShape, LShapeBoundary
from jax import random

from matplotlib import pyplot as plt

hbb = LShape()
X = hbb.sample_uniform(random.PRNGKey(0), N=1000)
print(X)
#plt.scatter(X[:, 0], X[:, 1])
#plt.show()
hbb_boundary = LShapeBoundary()  
Xb = hbb_boundary.sample_uniform(random.PRNGKey(1),side_number= None, N=50) 
print(Xb)
plt.scatter(Xb[:, 0], Xb[:, 1])
plt.show()



