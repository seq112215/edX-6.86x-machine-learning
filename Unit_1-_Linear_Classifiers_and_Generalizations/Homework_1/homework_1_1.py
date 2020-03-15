# %%
import numpy as np


# %%
def perceptron(x, y, T):
    theta = np.array([0, 0])
    theta_progression = []
    num_mistakes = 0
    maxt = 0
    for t in range(T):
        converge = 0
        print("t={}".format(t))
        for i in range(len(x)):
            if y[i]*np.dot(theta, x[i]) <= 0:
                num_mistakes += 1
                theta = theta + y[i]*x[i]
                theta_progression.append(theta)
                print(r"True, \theta={}".format(theta))
            else:
                converge += 1
                print("False")
        if converge == len(x):  # when returns False on all data points, done.
            # means it converged in the previous iteration, but counts from 1
            maxt = t - 1 + 1
            break

    return theta_progression, num_mistakes, maxt


# %%
# Starting with x_1:
x1 = np.array([[-1, -1], [1, 0], [-1, 10]])
y1 = np.array([1, -1, 1])
T1 = 10

theta_prog1, num_mistakes1, maxt1 = perceptron(x1, y1, T1)

print(theta_prog1, num_mistakes1, maxt1)

# %%
# Starting with x_2:
x2 = np.array([[1, 0], [-1, -1], [-1, 10]])
y2 = np.array([-1, 1, 1])
T2 = 10

theta_prog2, num_mistakes2, maxt2 = perceptron(x2, y2, T2)

print(theta_prog2, num_mistakes2, maxt2)