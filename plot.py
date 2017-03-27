import numpy as np
import pickle

import matplotlib.pylab as plt

log = pickle.load(open("log.p", "rb"))

plt.subplot(111)
plt.title("Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
curve, = plt.plot([e[1][-1] for e in log], 'b')
plt.show()
