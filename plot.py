import numpy as np
import pickle
import matplotlib.pylab as plt

log = pickle.load(open("log.p", "r"))
log2 = pickle.load(open("log2.p", "r"))
log_q = pickle.load(open("log-q.p", "r"))

plt.figure(figsize=(8, 12))
plt.subplot(311)
plt.title("Function Appoximation Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
curve, = plt.plot([e[1][-1] for e in log], 'b')
plt.subplot(312)
plt.title("Function Appoximation (+ Weights) Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
curve, = plt.plot([e[1][-1] for e in log2], 'b')
plt.subplot(313)
plt.title("Q-learning Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
curve, = plt.plot([e[1][-1] for e in log_q], 'r')


print 'Weights episode 0: ', log[0][2]
print 'Weights episode 450: ', log[450][2]
print 'Weights episode 499: ', log[-1][2]

print 'Weights 2 episode 0: ', log2[0][2]
print 'Weights 2 episode 499: ', log2[-1][2]

plt.tight_layout()
plt.savefig('learning.pdf',dpi=200)
