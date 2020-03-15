# %%
import numpy as np


# %%
def perceptron(x, y, T, theta=np.array([0, 0]), theta0=0):
    theta_progression = []
    theta0_progression = []
    num_mistakes = 0
    misclassified = np.zeros(len(x))
    maxt = 0
    for t in range(T):
        converge = 0
        print("t={}".format(t))
        for i in range(len(x)):
            print("i={}".format(i))
            if (y[i]*np.dot(theta, x[i]) + theta0) <= 0:
                num_mistakes += 1
                misclassified[i] += 1
                theta = theta + y[i]*x[i]
                theta_progression.append(theta)
                theta0 = theta0 + y[i]
                theta0_progression.append(theta0)
                print("theta={}, theta0={}".format(theta, theta0))
                print("num times misclassified={}".format(misclassified))
            else:
                converge += 1
                print(False)
        if converge == len(x):  # when returns False on all data points, done.
            # means it converged in the previous iteration, but counts from 1
            maxt = t - 1 + 1
            break

    return theta_progression, theta0_progression, num_mistakes, misclassified, \
           maxt


# %%
# 2)a):
x1 = np.array([[-4, 2], [-2, 1], [-1, -1], [2, 2], [1, -2]])
y1 = np.array([1, 1, -1, -1, -1])
# misclassified = np.array([1, 0, 2, 1, 0])
T1 = 3

theta_progression1, theta0_progression1, num_mistakes1, misclassified1, maxt1 \
    = perceptron(x1, y1, T1)

print("")
print("theta prog = {}".format(theta_progression1))
print("theta0 prog = {}".format(theta0_progression1))
print("num mistakes = {}".format(num_mistakes1))
print("num times misclassified = {}".format(misclassified1))
print("max t until converged = {}".format(maxt1))


# %%
# 2)b) attempt
theta_init2 = np.array([3, -1])
theta0_init2 = 3

theta_progression2, theta0_progression2, num_mistakes2, misclassified2, maxt2 \
    = perceptron(x1, y1, T1, theta_init2, theta0_init2)

print("")
print("theta prog = {}".format(theta_progression2))
print("theta0 prog = {}".format(theta0_progression2))
print("num mistakes = {}".format(num_mistakes2))
print("num times misclassified = {}".format(misclassified2))
print("max t until converged = {}".format(maxt2))

# %%
# Try just plotting the points and guessing a line

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1)

plus_points = list(y1 > 0)
minus_points = list(y1 < 0)
u = np.arange(min(x1[:, 0]), max(x1[:, 0]))
v = - (theta_init2[0] * u + theta0_init2) / theta_init2[1]

ax.scatter(x1[plus_points][:,0], x1[plus_points][:,1], marker='P')
ax.scatter(x1[minus_points][:, 0], x1[minus_points][:, 1], marker='o')
ax.plot(u, v)

plt.show()
