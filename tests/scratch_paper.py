import numpy as np

def pprint(name, value):
    print("=" * 10)
    print(name + ":")
    print(value)

Xsig_pred = [[5.9374, 6.0640, 5.925, 5.9436, 5.9266, 5.9374, 5.9389, 5.9374, 5.8106, 5.9457, 5.9310, 5.9465, 5.9374, 5.9359, 5.93744],
             [1.48, 1.4436, 1.660, 1.4934, 1.5036, 1.48, 1.4868, 1.48, 1.5271, 1.3104, 1.4787, 1.4674, 1.48, 1.4851, 1.486],
             [2.204, 2.2841, 2.2455, 2.2958, 2.204, 2.204, 2.2395, 2.204, 2.1256, 2.1642, 2.1139, 2.204, 2.204, 2.1702, 2.2049],
             [0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337, 0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188, 0.5367, 0.535048],
             [0.352, 0.29997, 0.46212, 0.37633, 0.4841, 0.41872, 0.352, 0.38744, 0.40562, 0.24347, 0.32926, 0.2214, 0.28687, 0.352, 0.318159]]

Xsig_pred = np.array(Xsig_pred)

n_aug = 7
n_z = 2
lambd = 3 - n_aug

w_0 = lambd / (lambd + n_aug)
w_1 = 0.5 / (lambd + n_aug)

weights = np.ones(shape=(2 * n_aug + 1,)) * w_1
weights[0] = w_0

Z_sig = Xsig_pred[:n_z]
np.testing.assert_equal(Z_sig.shape, (n_z, 2* n_aug + 1))

x_ = np.sum(Z_sig * weights, axis=1)
np.testing.assert_equal(x_.shape, (n_z,))

left = Z_sig - x_.reshape(n_z, 1)
assert left.shape == Z_sig.shape
pprint("Z_sig - x_", left)


S_ = np.dot(weights * left, left.T)
assert S_.shape == (n_z, n_z)

R = np.zeros((n_z, n_z), dtype=np.float32)
R[0, 0] = 0.15**2
R[1, 1] = 0.15**2

pprint("z_pred", x_)
pprint("S_", S_)
pprint("R", R)
pprint("S_ + R", S_ + R)


