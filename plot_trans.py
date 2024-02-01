import numpy as np
import matplotlib.pyplot as plt
def make_circle(r):
    theata = np.linspace(0, 2*np.pi, 1000)
    x = r*np.cos(theata)
    y = r*np.sin(theata)
    return np.vstack((x, y))

def get_long_short_axis(A):
    eigen_values, eigen_vectors = np.linalg.eig(A)
    long_axis_scale_index = np.argmax(eigen_values)
    short_axis_scale_index = np.argmin(eigen_values)

    long_axis_scale = eigen_values[long_axis_scale_index]
    short_axis_scale = eigen_values[short_axis_scale_index]


    long_axis = long_axis_scale*eigen_vectors[:, long_axis_scale_index]
    short_axis = short_axis_scale*eigen_vectors[:, short_axis_scale_index]

    return long_axis, short_axis
A = np.array([[2, 1], 
              [1, 2]])

before_trans = make_circle(1)
after_trans  = A @ before_trans
long_axis, short_axis = get_long_short_axis(A)
# plot the circle 
plt.scatter(before_trans[0], before_trans[1], s=0.5)
plt.scatter(after_trans[0], after_trans[1], s=0.5)

# plot klong axis and short axis
plt.plot([0, long_axis[0]], [0, long_axis[1]], color='r')
plt.plot([0, short_axis[0]], [0, short_axis[1]], color='g')

plt.show()
