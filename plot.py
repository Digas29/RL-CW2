import numpy as np
import pickle
import matplotlib.pylab as plt

log = pickle.load(open("log.p", "r"))
log_q = pickle.load(open("log-q.p", "r"))

plt.subplot(211)
plt.title("Function Appoximation Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
curve, = plt.plot([e[1][-1] for e in log], 'b')
plt.subplot(212)
plt.title("Q-learning Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
curve, = plt.plot([e[1][-1] for e in log_q], 'r')


print 'Weights episode 0: ', log[0][2]
print 'Weights episode 450: ', log[450][2]
print 'Weights episode 499: ', log[-1][2]

plt.tight_layout()
plt.savefig('learning.pdf',dpi=200)
