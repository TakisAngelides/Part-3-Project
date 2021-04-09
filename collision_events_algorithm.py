import time
from numpy import pi, cos, sin, e, tan, arctan
from clifford.g3 import blades
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from random import uniform, seed, randint
from sympy import LeviCivita as eps

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

e1, e2, e3 = blades['e1'], blades['e2'], blades['e3']
I = e1*e2*e3 # Pseudoscalar of 3D Euclidean geometric algebra

def plot_state(ax, p, q, a, b, c, d):

    X = (0)
    Y = (0)
    Z = (0)

    p1, p2, p3 = p[0], p[1], p[2]
    q1, q2, q3 = q[0], q[1], q[2]
    a1, a2, a3 = a[0], a[1], a[2]
    b1, b2, b3 = b[0], b[1], b[2]
    c1, c2, c3 = c[0], c[1], c[2]
    d1, d2, d3 = d[0], d[1], d[2]

    ax.quiver(X, Y, Z, p1, p2, p3, color='r', linestyle='-', label='p')
    ax.quiver(X, Y, Z, q1, q2, q3, color='k', linestyle='-', label='q')
    ax.quiver(X, Y, Z, a1, a2, a3, color='b', linestyle='-', label='a')
    ax.quiver(X, Y, Z, b1, b2, b3, color='g', linestyle='-', label='b')
    ax.quiver(X, Y, Z, c1, c2, c3, color='y', linestyle='-', label='c')
    ax.quiver(X, Y, Z, d1, d2, d3, color='orange', linestyle='-', label='d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    limit = 10
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.view_init(elev=1, azim=pi / 2)
    plt.legend()

def plot_parity_state(ax, p, q, a, b, c, d):

    X = (0)
    Y = (0)
    Z = (0)

    p1, p2, p3 = -p[0], -p[1], -p[2]
    q1, q2, q3 = -q[0], -q[1], -q[2]
    a1, a2, a3 = -a[0], -a[1], -a[2]
    b1, b2, b3 = -b[0], -b[1], -b[2]
    c1, c2, c3 = -c[0], -c[1], -c[2]
    d1, d2, d3 = -d[0], -d[1], -d[2]

    ax.quiver(X, Y, Z, p1, p2, p3, color='r', linestyle='--', label='p')
    ax.quiver(X, Y, Z, q1, q2, q3, color='k', linestyle='--', label='q')
    ax.quiver(X, Y, Z, a1, a2, a3, color='b', linestyle='--', label='a')
    ax.quiver(X, Y, Z, b1, b2, b3, color='g', linestyle='--', label='b')
    ax.quiver(X, Y, Z, c1, c2, c3, color='y', linestyle='--', label='c')
    ax.quiver(X, Y, Z, d1, d2, d3, color='orange', linestyle='--', label='d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    limit = 10
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.view_init(elev=1, azim=pi / 2)
    plt.legend()
    plt.show()

# -------------------------------------------

# For a collision event:

def parity(S):
    return [-v for v in S]

def rotate(S, R):
    return [R*v*~R for v in S]

def multivec_to_vec(a):
    return np.array([a[1], a[2], a[3]])

def energy(m,p):
    p = multivec_to_vec(p)
    return np.sqrt(m**2 + np.linalg.norm(p)**2)

def epsilon(a, b, c, d):

    summation = 0

    for i in range(0, 4):
        for j in range(0, 4):
            for k in range(0, 4):
                for l in range(0, 4):
                    summation += eps(i, j, k, l) * a[i] * b[j] * c[k] * d[l]

    return summation

def dot(a, b):

    #  Minkowski metric

    return a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]

def Gram_det_2(a,b,c,d):

    # a b
    # c d

    return (dot(a, c))*(dot(b, d)) - (dot(a, d))*(dot(b, c))

def sym_2_Gram_det(a,b):
    return Gram_det_2(a,b,a,b)

def sym_3_Gram_det(a,b,c):
    M = [[dot(a,a),dot(a,b),dot(a,c)],[dot(b,a),dot(b,b),dot(b,c)],[dot(c,a),dot(c,b),dot(c,c)]]
    return np.linalg.det(M)

def swap(S,idx_1,idx_2):

    tmp = S[idx_1]
    S[idx_1] = S[idx_2]
    S[idx_2] = tmp

    return S

def permute_with_idx(M, E, idx_to_permute):

    same_mass_with_idx = [idx for idx in range(len(M)) if M[idx] == M[idx_to_permute] and idx != idx_to_permute and idx != 0 and idx != 1]
    same_energy_with_idx = [idx for idx in range(len(E)) if E[idx] == E[idx_to_permute] and idx != idx_to_permute]

    return list(set(same_mass_with_idx) and set(same_energy_with_idx))

def permutation_boolean(M, E, idx_1, idx_2):

    if (M[idx_1] == M[idx_2]) and (E[idx_1] == E[idx_2]):
        return True
    else:
        return False

def logic_statement_true_for_non_chiral(S, E):

    p = multivec_to_vec(S[0])
    p = np.insert(p, 0, E[0])
    q = multivec_to_vec(S[1])
    q = np.insert(q, 0, E[1])
    a = multivec_to_vec(S[2])
    a = np.insert(a, 0, E[2])
    b = multivec_to_vec(S[3])
    b = np.insert(b, 0, E[3])
    c = multivec_to_vec(S[4])
    c = np.insert(c, 0, E[4])
    d = multivec_to_vec(S[5])
    d = np.insert(d, 0, E[5])
    RF = p + q

    case_1 = ((epsilon(a,b,p,q) == 0) and (epsilon(a,c,p,q) == 0) and (epsilon(a,d,p,q) == 0) and (epsilon(b,d,p,q) == 0)
             and (epsilon(b,c,p,q) == 0) and (epsilon(c,d,p,q) == 0))

    case_2 =  ((dot(a,a) == dot(b,b)) and (Gram_det_2(a-b,RF,p-q,RF) == 0) and (Gram_det_2(a-b,RF,a+b,RF) == 0)
              and (((sym_3_Gram_det(a,p,q) == 0) and (sym_3_Gram_det(b,p,q) == 0) and (sym_3_Gram_det(c,p,q) != 0) and (sym_3_Gram_det(d,p,q) == 0))
               or ((sym_3_Gram_det(a,p,q) != 0) and (sym_3_Gram_det(b,p,q) != 0) and (sym_3_Gram_det(c,p,q) != 0) and (sym_3_Gram_det(d,p,q) == 0) and (Gram_det_2(a-b,RF,c,RF) == 0))
               or ((sym_3_Gram_det(a,p,q) != 0) and (sym_3_Gram_det(b,p,q) != 0) and (sym_3_Gram_det(d,p,q) != 0) and (sym_3_Gram_det(c,p,q) == 0) and (Gram_det_2(a-b,RF,d,RF) == 0))
               or ((sym_3_Gram_det(a,p,q) !=0) and (sym_3_Gram_det(b,p,q) != 0) and (sym_3_Gram_det(d,p,q) != 0) and (sym_3_Gram_det(c,p,q) != 0) and (Gram_det_2(a-b,RF,c,RF) == 0) and (Gram_det_2(a-b,RF,d,RF) == 0))))

    def case_2_symmetry(p, q, a, b, c, d):

        return ((dot(a,a) == dot(b,b)) and (Gram_det_2(a-b,RF,p-q,RF) == 0) and (Gram_det_2(a-b,RF,a+b,RF) == 0)
              and (((sym_3_Gram_det(a,p,q) == 0) and (sym_3_Gram_det(b,p,q) == 0) and (sym_3_Gram_det(c,p,q) != 0) and (sym_3_Gram_det(d,p,q) == 0))
               or ((sym_3_Gram_det(a,p,q) != 0) and (sym_3_Gram_det(b,p,q) != 0) and (sym_3_Gram_det(c,p,q) != 0) and (sym_3_Gram_det(d,p,q) == 0) and (Gram_det_2(a-b,RF,c,RF) == 0))
               or ((sym_3_Gram_det(a,p,q) != 0) and (sym_3_Gram_det(b,p,q) != 0) and (sym_3_Gram_det(d,p,q) != 0) and (sym_3_Gram_det(c,p,q) == 0) and (Gram_det_2(a-b,RF,d,RF) == 0))
               or ((sym_3_Gram_det(a,p,q) !=0) and (sym_3_Gram_det(b,p,q) != 0) and (sym_3_Gram_det(d,p,q) != 0) and (sym_3_Gram_det(c,p,q) != 0) and (Gram_det_2(a-b,RF,c,RF) == 0) and (Gram_det_2(a-b,RF,d,RF) == 0))))

    case_3 = case_2_symmetry(p, q, c, d, a, b)  # h = (cd)
    case_4 = case_2_symmetry(p, q, b, c, a, d)  # h = (bc)
    case_5 = case_2_symmetry(p, q, b, d, a, c)  # h = (bd)
    case_6 = case_2_symmetry(p, q, a, c, b, d)  # h = (ac)
    case_7 = case_2_symmetry(p, q, a, d, c, b)  # h = (ad)

    # TODO: In case 16 green and blue bracket are the negation of each other so I can evaluate once for efficiency

    case_16 = ((dot(a,a) == dot(b,b)) and (dot(c,c) == dot(d,d)) and (Gram_det_2(a-b,RF,p-q,RF) == 0) and (Gram_det_2(a-b,RF,a+b,RF) == 0) and (Gram_det_2(c-d,RF,p-q,RF) == 0) and (Gram_det_2(c-d,RF,c+d,RF) == 0)
              and (((sym_3_Gram_det(a,p,q)==0) and (sym_3_Gram_det(b,p,q)==0) and (sym_3_Gram_det(c,p,q)!=0) and (sym_3_Gram_det(d,p,q)!=0) or (sym_3_Gram_det(c,p,q)==0) and (sym_3_Gram_det(d,p,q)==0) and (sym_3_Gram_det(a,p,q)!=0) and (sym_3_Gram_det(b,p,q)!=0))
                   or ((Gram_det_2(a-b,RF,c+d,RF)==0) or (Gram_det_2(c-d,RF,b,RF)==0) or (Gram_det_2(a-b,RF,d,RF)==0))))

    def case_16_symmetry(p,q,a,b,c,d):

        return ((dot(a,a) == dot(b,b)) and (dot(c,c) == dot(d,d)) and (Gram_det_2(a-b,RF,p-q,RF) == 0) and (Gram_det_2(a-b,RF,a+b,RF) == 0) and (Gram_det_2(c-d,RF,p-q,RF) == 0) and (Gram_det_2(c-d,RF,c+d,RF) == 0)
              and (((sym_3_Gram_det(a,p,q)==0) and (sym_3_Gram_det(b,p,q)==0) and (sym_3_Gram_det(c,p,q)!=0) and (sym_3_Gram_det(d,p,q)!=0) or (sym_3_Gram_det(c,p,q)==0) and (sym_3_Gram_det(d,p,q)==0) and (sym_3_Gram_det(a,p,q)!=0) and (sym_3_Gram_det(b,p,q)!=0))
                   or ((Gram_det_2(a-b,RF,c+d,RF)==0) or (Gram_det_2(c-d,RF,b,RF)==0) or (Gram_det_2(a-b,RF,d,RF)==0))))

    case_17 = case_16_symmetry(p, q, a, c, b, d)  # h = (ac)(bd)
    case_18 = case_16_symmetry(p, q, a, d, c, b)  # h = (ad)(bc)

    case_19 = ((dot(a,a) == dot(b,b) == dot(c,c) == dot(d,d)) and (Gram_det_2(a-b,RF,a+b,RF) == 0) and (Gram_det_2(b-c,RF,b+c,RF) == 0) \
              and (Gram_det_2(c-d,RF,c+d,RF) == 0) and (Gram_det_2(a-b,RF,p-q,RF) == 0) and (Gram_det_2(b-c,RF,p-q,RF) == 0) \
              and (Gram_det_2(c-d,RF,p-q,RF) == 0) and (a == c) and (b == d))

    def case_19_symmetry(p,q,a,b,c,d):

        return ((dot(a,a) == dot(b,b) == dot(c,c) == dot(d,d)) and (Gram_det_2(a-b,RF,a+b,RF) == 0) and (Gram_det_2(b-c,RF,b+c,RF) == 0) \
              and (Gram_det_2(c-d,RF,c+d,RF) == 0) and (Gram_det_2(a-b,RF,p-q,RF) == 0) and (Gram_det_2(b-c,RF,p-q,RF) == 0) \
              and (Gram_det_2(c-d,RF,p-q,RF) == 0) and (a == c) and (b == d))

    case_20 = case_19_symmetry(p,q,a,b,d,c) # h = (dbac)
    case_22 = case_19_symmetry(p,q,a,c,d,b) # h = (bcad)

    case_25 = ((dot(p,p) == dot(q,q)) and (Gram_det_2(a,RF,p-q,RF) == 0) and (Gram_det_2(b,RF,p-q,RF) == 0) \
             and (Gram_det_2(c,RF,p-q,RF) == 0) and (Gram_det_2(d,RF,p-q,RF) == 0))

    case_26 = ((dot(p,p) == dot(q,q)) and (dot(a,a) == dot(b,b)) and (Gram_det_2(a+b,RF,p-q,RF) == 0) and (Gram_det_2(a-b,RF,a+b,RF) == 0)
               and (Gram_det_2(c,RF,p-q,RF) == 0) and (Gram_det_2(d,RF,p-q,RF) == 0)
               and (sym_3_Gram_det(a,p,q) == 0 or sym_3_Gram_det(a+b,p,q) == 0 or sym_3_Gram_det(a-b,p,q) == 0))

    def case_26_symmetry(p,q,a,b,c,d):

        return ((dot(p,p) == dot(q,q)) and (dot(a,a) == dot(b,b)) and (Gram_det_2(a+b,RF,p-q,RF) == 0) and (Gram_det_2(a-b,RF,a+b,RF) == 0)
               and (Gram_det_2(c,RF,p-q,RF) == 0) and (Gram_det_2(d,RF,p-q,RF) == 0)
               and (sym_3_Gram_det(a,p,q) == 0 or sym_3_Gram_det(a+b,p,q) == 0 or sym_3_Gram_det(a-b,p,q) == 0))

    case_27 = case_26_symmetry(p, q, c, d, a, b)  # h = (cd)
    case_28 = case_26_symmetry(p, q, c, b, a, d)  # h = (bc)
    case_29 = case_26_symmetry(p, q, d, b, c, a)  # h = (bd)
    case_30 = case_26_symmetry(p, q, a, c, b, d)  # h = (ac)
    case_31 = case_26_symmetry(p, q, a, d, c, b)  # h = (ad)


    case_32 = ((dot(p,p) == dot(q,q)) and (dot(b,b) == dot(c,c) == dot(d,d)) and (Gram_det_2(a,RF,p-q,RF) == 0) and (Gram_det_2(b,RF,p-q,RF) == 0) \
              and (Gram_det_2(c,RF,p-q,RF) == 0) and (Gram_det_2(d,RF,p-q,RF) == 0) and (Gram_det_2(b-c,RF,b+c,RF) == 0) \
              and (Gram_det_2(c-d,RF,c+d,RF) == 0) and (Gram_det_2(a,RF,p-q,RF) == 0) and (epsilon((b+d),c,p,q) == 0) \
              and (epsilon((b+c),d,p,q) == 0) and (epsilon((c+d),b,p,q) == 0))

    def case_32_symmetry(p,q,a,b,c,d):

        return ((dot(p,p) == dot(q,q)) and (dot(b,b) == dot(c,c) == dot(d,d)) and (Gram_det_2(a,RF,p-q,RF) == 0) and (Gram_det_2(b,RF,p-q,RF) == 0) \
              and (Gram_det_2(c,RF,p-q,RF) == 0) and (Gram_det_2(d,RF,p-q,RF) == 0) and (Gram_det_2(b-c,RF,b+c,RF) == 0) \
              and (Gram_det_2(c-d,RF,c+d,RF) == 0) and (Gram_det_2(a,RF,p-q,RF) == 0) and (epsilon((b+d),c,p,q) == 0) \
              and (epsilon((b+c),d,p,q) == 0) and (epsilon((c+d),b,p,q) == 0))

    case_34 = case_32_symmetry(p, q, d, b, c, a)  # h = (cba)
    case_35 = case_32_symmetry(p, q, c, a, d, b)  # h = (dba)
    case_37 = case_32_symmetry(p, q, b, a, c, d)  # h = (dca)

    case_40 = ((dot(p,p) == dot(q,q)) and (dot(a,a) == dot(b,b)) and (dot(c,c) == dot(d,d)) and (Gram_det_2(a+b,RF,p-q,RF) == 0) \
              and (Gram_det_2(c+d,RF,p-q,RF) == 0) \
              and ((sym_3_Gram_det(a+b,p,q) == 0 or sym_3_Gram_det(a-b,p,q) == 0)
                   or ((sym_3_Gram_det(c+d,p,q) == 0) or (sym_3_Gram_det(c-d,p,q) == 0))))

    def case_40_symmetry(p,q,a,b,c,d):

        return ((dot(p,p) == dot(q,q)) and (dot(a,a) == dot(b,b)) and (dot(c,c) == dot(d,d)) and (Gram_det_2(a+b,RF,p-q,RF) == 0) \
              and (Gram_det_2(c+d,RF,p-q,RF) == 0) \
              and ((sym_3_Gram_det(a+b,p,q) == 0 or sym_3_Gram_det(a-b,p,q) == 0)
                   or ((sym_3_Gram_det(c+d,p,q) == 0) or (sym_3_Gram_det(c-d,p,q) == 0))))

    case_41 = case_40_symmetry(p, q, a, c, b, d)  # h = (ac)(bd)
    case_42 = case_40_symmetry(p, q, a, d, c, b)  # h = (ad)(cb)

    case_43 = ((dot(p,p) == dot(q,q)) and (dot(a,a) == dot(b,b) == dot(c,c) == dot(d,d)) and (Gram_det_2(a-b,RF,a+b,RF) == 0)
    and (Gram_det_2(b-c,RF,b+c,RF) == 0) and (Gram_det_2(c-d,RF,c+d,RF) == 0) and (Gram_det_2(a+b,RF,p-q,RF) == 0) and (Gram_det_2(b+c,RF,p-q,RF) == 0)
    and (Gram_det_2(c+d,RF,p-q,RF) == 0) and ((sym_3_Gram_det(a,p,q) == 0)
    or ((epsilon(p+q,a-c,p,b) == 0) and (epsilon(p+q,b-d,a,p) == 0))
    or ((sym_3_Gram_det(a-c,p,q) == 0) and (sym_3_Gram_det(b-d,p,q) == 0)
        and ((sym_3_Gram_det(a+b,p,q) == 0) or ((Gram_det_2(a+b,RF,p-q,RF) == 0) and (sym_3_Gram_det(a-b,p,q) == 0))))))

    def case_43_symmetry(p,q,a,b,c,d):

        return ((dot(p,p) == dot(q,q)) and (dot(a,a) == dot(b,b) == dot(c,c) == dot(d,d)) and (Gram_det_2(a-b,RF,a+b,RF) == 0)
    and (Gram_det_2(b-c,RF,b+c,RF) == 0) and (Gram_det_2(c-d,RF,c+d,RF) == 0) and (Gram_det_2(a+b,RF,p-q,RF) == 0) and (Gram_det_2(b+c,RF,p-q,RF) == 0)
    and (Gram_det_2(c+d,RF,p-q,RF) == 0) and ((sym_3_Gram_det(a,p,q) == 0)
    or ((epsilon(p+q,a-c,p,b) == 0) and (epsilon(p+q,b-d,a,p) == 0))
    or ((sym_3_Gram_det(a-c,p,q) == 0) and (sym_3_Gram_det(b-d,p,q) == 0)
        and ((sym_3_Gram_det(a+b,p,q) == 0) or ((Gram_det_2(a+b,RF,p-q,RF) == 0) and (sym_3_Gram_det(a-b,p,q) == 0))))))

    case_44 = case_43_symmetry(p, q, a, b, d, c)  # h = (dbac)
    case_45 = case_43_symmetry(p, q, b, a, c, d)  # h = (dcab)
    case_46 = case_43_symmetry(p, q, a, c, b, d)  # h = (dbca)
    case_47 = case_43_symmetry(p, q, c, b, a, d)  # h = (dabc)
    case_48 = case_43_symmetry(p, q, b, c, a, d)  # h = (dacb)

    return (case_1 or case_2 or case_3 or case_4 or case_5 or case_6 or case_7 or case_16 or case_17 or case_18
            or case_19 or case_20 or case_22 or case_25 or case_26 or case_27 or case_28 or case_29 or case_30
            or case_31 or case_32 or case_34 or case_35 or case_37 or case_40 or case_41 or case_42 or case_43
            or case_44 or case_45 or case_46 or case_47 or case_48)

def construct_state():

    mp, mq, ma, mb, mc, md = randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 10), randint(0,10)

    p = uniform(-10, 10) * e3
    q = -p

    a = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3
    b = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3
    c = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3
    d = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3

    Ep, Eq, Ea, Eb, Ec, Ed = energy(mp, p), energy(mq, q), energy(ma, a), energy(mb, b), energy(mc, c), energy(md, d)

    M = [mp, mq, ma, mb, mc, md]
    E = [Ep, Eq, Ea, Eb, Ec, Ed]
    S = [p, q, a, b, c, d]

    return S, E, M

def construct_non_chiral_state(): # Trick

    mp, mq, ma, mb, mc, md = 1, 1, 1, 1, 1, 1

    p = uniform(-10, 10) * e3
    q = -p

    a = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3
    b = a[1] * e1 - a[2] * e2 + a[3] * e3
    R = e**(uniform(0,2*pi)*e1*e2)
    c = R*a*~R
    d = c[1] * e1 - c[2] * e2 + c[3] * e3

    Ep, Eq, Ea, Eb, Ec, Ed = energy(mp, p), energy(mq, q), energy(ma, a), energy(mb, b), energy(mc, c), energy(md, d)

    M = [mp, mq, ma, mb, mc, md]
    E = [Ep, Eq, Ea, Eb, Ec, Ed]
    S = [p, q, a, b, c, d]

    return S, E, M

def chirality_test():

    chiral_states = []
    non_chiral_states = []

    S, E, M = construct_non_chiral_state()
    a, b, c, d = S[2], S[3], S[4], S[5]

    # These lists hold indices (as they appear in S) of the particles that can be permuted with a,b,c and d respectively
    permute_with_a = permute_with_idx(M, E, 2)
    permute_with_b = permute_with_idx(M, E, 3)
    permute_with_c = permute_with_idx(M, E, 4)
    permute_with_d = permute_with_idx(M, E, 5)

    permutation_dictionary = {3: permute_with_b, 4: permute_with_c, 5: permute_with_d}

    S_parity = parity(S)  # Perform parity on the set of momenta

    flag = False

    if not permutation_boolean(M, E, 0, 1): # If we cant permute p (index 0) with q (index 1)

        if (a[1] == 0 and a[2] == 0) or (a == 0):

            if (b[1] == 0 and b[2] == 0) or (b == 0):

                if (c[1] == 0 and c[2] == 0) or (c == 0):

                    flag = True

                else: # Here we consider that a,b are either e3 collinear or 0, only c and d are in the 1-2 plane

                    # We can map c_12 to its original or d_12 to the original c_12 and swap (cd) if they can be
                    # permuted.

                    # Since a and b are fixed by R1 = e1e3 we only want to consider permutations (cd)
                    # Remember that idx 2 corresponds to a and idx 3 corresponds to b in S
                    permute_with_c_new = list([idx for idx in permute_with_c if (idx != 2) and (idx != 3)])

                    R1 = e1*e3

                    S1 = rotate(S_parity, R1)

                    for idx in permute_with_c_new+[4]: # The index of c is 4 in the list S

                        # Map x_12_rotated to c_12_original

                        x = S1[idx]
                        x_12 = x - x[3]*e3
                        c_12 = c - c[3]*e3
                        n = (x_12 + c_12).normal()
                        if n == 0: # If x_12 and c_12 are anti-parallel then we need a pi rotation
                            R2 = e1*e2
                        else: # Otherwise we construct the rotor as usual with the form R = (final destination)*n
                            R2 = c_12.normal()*n
                        S2 = rotate(S1, R2)
                        S3 = swap(S2, idx, 4)

                        if S == S3:
                            flag = True

            else: # Here we consider that a is collinear with e3 or is 0 but b is not

                # We map x_12 to b_12_original and consider permutations between (cd), where x_12 can be (b,c,d)_12

                permute_with_b_new = list([idx for idx in permute_with_b if (idx != 2)])

                R1 = e1 * e3

                S1 = rotate(S_parity, R1)

                for idx in permute_with_b_new + [3]:  # The index of b is 3 in the list S

                    # Map x_12_rotated to b_12_original

                    x = S1[idx]
                    x_12 = x - x[3] * e3
                    b_12 = b - b[3] * e3
                    n = (x_12 + b_12).normal()
                    if n == 0:  # If x_12 and b_12 are anti-parallel then we need a pi rotation
                        R2 = e1 * e2
                    else:  # Otherwise we construct the rotor as usual with the form R = (final destination)*n
                        R2 = b_12.normal() * n
                    S2 = rotate(S1, R2)
                    S3 = swap(S2, idx, 3)

                    if S == S3:
                        flag = True

                    if permutation_boolean(M, E, 4, 5): # If (cd) is possible

                        S4 = swap(S3, 4, 5)

                        if S == S4:

                            flag = True

        else: # Here we consider that a is not collinear with e3 and not 0

            # We need to map x_12 to a_12_original and consider permutations (bcd)

            R1 = e1 * e3

            S1 = rotate(S_parity, R1)

            for idx in permute_with_a + [2]:  # The index of a is 2 in the list S

                # Map x_12_rotated to a_12_original

                x = S1[idx]
                x_12 = x - x[3] * e3
                a_12 = a - a[3] * e3
                n = (x_12 + a_12).normal()
                if n == 0:  # If x_12 and a_12 are anti-parallel then we need a pi rotation
                    R2 = e1 * e2
                else:  # Otherwise we construct the rotor as usual with the form R = (final destination)*n
                    R2 = a_12.normal() * n
                S2 = rotate(S1, R2)
                S3 = swap(S2, idx, 2)

                if S == S3:
                    flag = True

                flag_tmp_1 = permutation_boolean(M, E, 3, 4) # This checks if (bc) is available

                if flag_tmp_1:
                    S4 = swap(S3, 3, 4)
                    if S == S4:
                        flag = True

                flag_tmp_2 = permutation_boolean(M, E, 3, 5) # This checks if (bd) is available

                if flag_tmp_2:
                    S5 = swap(S3, 3, 5)
                    if S == S5:
                        flag = True

                if flag_tmp_1 and flag_tmp_2: # If we have (bc) and (bd) then we have (bcd)
                    # The following achieves b->c->d->b
                    S6 = swap(S3, 3, 4) # (bc)
                    S6 = swap(S6, 3, 5) # (bd), here in the 3 index lies c but we name it b still, notice we use S6
                    if S == S6:
                        flag = True
                    # The following achieves b->d->c->b
                    S7 = swap(S3, 3, 5)  # (bd)
                    S7 = swap(S7, 3, 4)  # (bc), here in the 3 index lies d but we name it b still, notice we use S7
                    if S == S7:
                        flag = True

                if permutation_boolean(M, E, 4, 5):  # If we have (cd)

                    S8 = swap(S3, 4, 5)
                    if S == S8:
                        flag = True


    else: # Now we consider the case where (pq) is available

        # We can either try the above or omit using R1 and just use (pq)
        # Whenever we check for non-chirality we can can also perform a pi rotation in the plane that contains the
        # 3-axis and has normal the final_state_particle_12 we matched in the 1-2 plane (usually this is a unless a has
        # no 1,2 components)

        # The following is repeating the above but incorporating the (pq) degree of freedom

        if (a[1] == 0 and a[2] == 0) or (a == 0):

            if (b[1] == 0 and b[2] == 0) or (b == 0):

                if (c[1] == 0 and c[2] == 0) or (c == 0):

                    flag = True

                else: # Here we consider that a,b are either e3 collinear or 0, only c and d are in the 1-2 plane

                    # We can map c_12 to its original or d_12 to the original c_12 and swap (cd) if they can be
                    # permuted.

                    # Since a and b are fixed by R1 = e1e3 we only want to consider permutations (cd)
                    # Remember that idx 2 corresponds to a and idx 3 corresponds to b in S
                    permute_with_c_new = list([idx for idx in permute_with_c if (idx != 2) and (idx != 3)])

                    R1 = e1*e3

                    S1 = rotate(S_parity, R1)

                    for idx in permute_with_c_new+[4]: # The index of c is 4 in the list S

                        # Map x_12_rotated to c_12_original

                        x = S1[idx]
                        x_12 = x - x[3]*e3
                        c_12 = c - c[3]*e3
                        n = (x_12 + c_12).normal()
                        if n == 0: # If x_12 and c_12 are anti-parallel then we need a pi rotation
                            R2 = e1*e2
                        else: # Otherwise we construct the rotor as usual with the form R = (final destination)*n
                            R2 = c_12.normal()*n
                        S2 = rotate(S1, R2)
                        S3 = swap(S2, idx, 4)

                        if S == S3:
                            flag = True

                        # Now we can try the pi rotation in the plane v-e3 where v is perpendicular c_12 and e3

                        v = (-I*(e3^c_12)).normal() # Cross product in geometric algebra, I is the pseudoscalar, v = e3 x c_12
                        R3 = e3^v

                        S4 = rotate(S3, R3)
                        S5 = swap(S4, 0, 1)

                        if S == S5:
                            flag = True

            else: # Here we consider that a is collinear with e3 or is 0 but b is not

                # We map x_12 to b_12_original and consider permutations between (cd), where x_12 can be (b,c,d)_12

                permute_with_b_new = list([idx for idx in permute_with_b if (idx != 2)])

                R1 = e1 * e3

                S1 = rotate(S_parity, R1)

                for idx in permute_with_b_new + [3]:  # The index of b is 3 in the list S

                    # Map x_12_rotated to b_12_original

                    x = S1[idx]
                    x_12 = x - x[3] * e3
                    b_12 = b - b[3] * e3
                    n = (x_12 + b_12).normal()
                    if n == 0:  # If x_12 and b_12 are anti-parallel then we need a pi rotation
                        R2 = e1 * e2
                    else:  # Otherwise we construct the rotor as usual with the form R = (final destination)*n
                        R2 = b_12.normal() * n
                    S2 = rotate(S1, R2)
                    S3 = swap(S2, idx, 3)

                    if S == S3:
                        flag = True

                    v = (-I * (e3 ^ b_12)).normal()  # Cross product in geometric algebra, I is the pseudoscalar, v = e3 x c_12
                    R3 = e3 ^ v # (pq) degree of freedom, rotation by pi in the plane e3-v

                    S4 = rotate(S3, R3)
                    S5 = swap(S4, 0, 1)

                    if S == S5:
                        flag = True

                    if permutation_boolean(M, E, 4, 5): # If (cd) is possible

                        S6 = swap(S3, 4, 5)

                        if S == S6:

                            flag = True

                        S7 = rotate(S6, R3)
                        S8 = swap(S7, 0, 1)

                        if S == S8:
                            flag = True

        else: # Here we consider that a is not collinear with e3 and not 0

            # We need to map x_12 to a_12_original and consider permutations (bcd)

            R1 = e1 * e3

            S1 = rotate(S_parity, R1)

            for idx in permute_with_a + [2]:  # The index of a is 2 in the list S

                # Map x_12_rotated to a_12_original

                x = S1[idx]
                x_12 = x - x[3] * e3
                a_12 = a - a[3] * e3
                n = (x_12 + a_12).normal()
                if n == 0:  # If x_12 and a_12 are anti-parallel then we need a pi rotation
                    R2 = e1 * e2
                else:  # Otherwise we construct the rotor as usual with the form R = (final destination)*n
                    R2 = a_12.normal() * n
                S2 = rotate(S1, R2)
                S3 = swap(S2, idx, 2)

                if S == S3:
                    flag = True

                v = (-I * (e3 ^ a_12)).normal()  # Cross product in geometric algebra, I is the pseudoscalar, v = e3 x c_12
                R3 = e3 ^ v  # (pq) degree of freedom, rotation by pi in the plane e3-v

                S4 = rotate(S3, R3)
                S5 = swap(S4, 0, 1)

                if S == S5:
                    flag = True

                flag_tmp_1 = permutation_boolean(M, E, 3, 4) # This checks if (bc) is available

                if flag_tmp_1:
                    S6 = swap(S3, 3, 4)
                    if S == S6:
                        flag = True

                    S7 = rotate(S6, R3)
                    S8 = swap(S7, 0, 1)

                    if S == S8:
                        flag = True

                flag_tmp_2 = permutation_boolean(M, E, 3, 5) # This checks if (bd) is available

                if flag_tmp_2:
                    S9 = swap(S3, 3, 5)
                    if S == S9:
                        flag = True
                    S10 = rotate(S9, R3)
                    S11 = swap(S10, 0, 1)

                    if S == S11:
                        flag = True



                if flag_tmp_1 and flag_tmp_2: # If we have (bc) and (bd) then we have (bcd)
                    # The following achieves b->c->d->b
                    S12 = swap(S3, 3, 4) # (bc)
                    S13 = swap(S12, 3, 5) # (bd), here in the 3 index lies c but we name it b still, notice we use S6
                    if S == S13:
                        flag = True
                    S14 = rotate(S13, R3)
                    S15 = swap(S14, 0, 1)

                    if S == S15:
                        flag = True
                    # The following achieves b->d->c->b
                    S16 = swap(S3, 3, 5)  # (bd)
                    S17 = swap(S16, 3, 4)  # (bc), here in the 3 index lies d but we name it b still, notice we use S7
                    if S == S17:
                        flag = True
                    S18 = rotate(S17, R3)
                    S19 = swap(S18, 0, 1)

                    if S == S19:
                        flag = True

                if permutation_boolean(M, E, 4, 5):  # If we have (cd)

                    S20 = swap(S3, 4, 5)
                    if S == S20:
                        flag = True
                    S21 = rotate(S20, R3)
                    S22 = swap(S21, 0, 1)

                    if S == S22:
                        flag = True


        # The following is omitting R1 and just uses (pq), it again incorporates the (pq) degree of freedom

        if (a[1] == 0 and a[2] == 0) or (a == 0):

            if (b[1] == 0 and b[2] == 0) or (b == 0):

                if (c[1] == 0 and c[2] == 0) or (c == 0):

                    flag = True

                else:  # Here we consider that a,b are either e3 collinear or 0, only c and d are in the 1-2 plane

                    # We can map c_12 to its original or d_12 to the original c_12 and swap (cd) if they can be
                    # permuted.

                    # Since a and b are fixed by R1 = e1e3 we only want to consider permutations (cd)
                    # Remember that idx 2 corresponds to a and idx 3 corresponds to b in S
                    permute_with_c_new = list([idx for idx in permute_with_c if (idx != 2) and (idx != 3)])

                    S1 = swap(S_parity, 0, 1)

                    for idx in permute_with_c_new + [4]:  # The index of c is 4 in the list S

                        # Map x_12_rotated to c_12_original

                        x = S1[idx]
                        x_12 = x - x[3] * e3
                        c_12 = c - c[3] * e3
                        n = (x_12 + c_12).normal()
                        if n == 0:  # If x_12 and c_12 are anti-parallel then we need a pi rotation
                            R2 = e1 * e2
                        else:  # Otherwise we construct the rotor as usual with the form R = (final destination)*n
                            R2 = c_12.normal() * n
                        S2 = rotate(S1, R2)
                        S3 = swap(S2, idx, 4)

                        if S == S3:
                            flag = True

                        # Now we can try the pi rotation in the plane v-e3 where v is perpendicular c_12 and e3

                        v = (-I * (e3 ^ c_12)).normal()  # Cross product in geometric algebra, I is the pseudoscalar, v = e3 x c_12
                        R3 = e3 ^ v

                        S4 = rotate(S3, R3)
                        S5 = swap(S4, 0, 1)

                        if S == S5:
                            flag = True

            else:  # Here we consider that a is collinear with e3 or is 0 but b is not

                # We map x_12 to b_12_original and consider permutations between (cd), where x_12 can be (b,c,d)_12

                permute_with_b_new = list([idx for idx in permute_with_b if (idx != 2)])

                S1 = swap(S_parity, 0, 1)

                for idx in permute_with_b_new + [3]:  # The index of b is 3 in the list S

                    # Map x_12_rotated to b_12_original

                    x = S1[idx]
                    x_12 = x - x[3] * e3
                    b_12 = b - b[3] * e3
                    n = (x_12 + b_12).normal()
                    if n == 0:  # If x_12 and b_12 are anti-parallel then we need a pi rotation
                        R2 = e1 * e2
                    else:  # Otherwise we construct the rotor as usual with the form R = (final destination)*n
                        R2 = b_12.normal() * n
                    S2 = rotate(S1, R2)
                    S3 = swap(S2, idx, 3)

                    if S == S3:
                        flag = True

                    v = (-I * (e3 ^ b_12)).normal()  # Cross product in geometric algebra, I is the pseudoscalar, v = e3 x c_12
                    R3 = e3 ^ v # (pq) degree of freedom, rotation by pi in the plane e3-v

                    S4 = rotate(S3, R3)
                    S5 = swap(S4, 0, 1)

                    if S == S5:
                        flag = True

                    if permutation_boolean(M, E, 4, 5):  # If (cd) is possible

                        S6 = swap(S3, 4, 5)

                        if S == S6:
                            flag = True

                        S7 = rotate(S6, R3)
                        S8 = swap(S7, 0, 1)

                        if S == S8:
                            flag = True

        else:  # Here we consider that a is not collinear with e3 and not 0

            # We need to map x_12 to a_12_original and consider permutations (bcd)

            S1 = swap(S_parity, 0, 1)

            for idx in permute_with_a + [2]:  # The index of a is 2 in the list S

                # Map x_12_rotated to a_12_original

                x = S1[idx]
                x_12 = x - x[3] * e3
                a_12 = a - a[3] * e3
                n = (x_12 + a_12).normal()
                if n == 0:  # If x_12 and a_12 are anti-parallel then we need a pi rotation
                    R2 = e1 * e2
                else:  # Otherwise we construct the rotor as usual with the form R = (final destination)*n
                    R2 = a_12.normal() * n
                S2 = rotate(S1, R2)
                S3 = swap(S2, idx, 2)

                if S == S3:
                    flag = True

                v = (-I * (e3 ^ a_12)).normal()  # Cross product in geometric algebra, I is the pseudoscalar, v = e3 x c_12
                R3 = e3 ^ v  # (pq) degree of freedom, rotation by pi in the plane e3-v

                S4 = rotate(S3, R3)
                S5 = swap(S4, 0, 1)

                if S == S5:
                    flag = True

                flag_tmp_1 = permutation_boolean(M, E, 3, 4)  # This checks if (bc) is available

                if flag_tmp_1:
                    S6 = swap(S3, 3, 4)
                    if S == S6:
                        flag = True

                    S7 = rotate(S6, R3)
                    S8 = swap(S7, 0, 1)

                    if S == S8:
                        flag = True

                flag_tmp_2 = permutation_boolean(M, E, 3, 5)  # This checks if (bd) is available

                if flag_tmp_2:
                    S9 = swap(S3, 3, 5)
                    if S == S9:
                        flag = True
                    S10 = rotate(S9, R3)
                    S11 = swap(S10, 0, 1)

                    if S == S11:
                        flag = True

                if flag_tmp_1 and flag_tmp_2:  # If we have (bc) and (bd) then we have (bcd)
                    # The following achieves b->c->d->b
                    S12 = swap(S3, 3, 4)  # (bc)
                    S13 = swap(S12, 3, 5)  # (bd), here in the 3 index lies c but we name it b still, notice we use S6
                    if S == S13:
                        flag = True
                    S14 = rotate(S13, R3)
                    S15 = swap(S14, 0, 1)

                    if S == S15:
                        flag = True
                    # The following achieves b->d->c->b
                    S16 = swap(S3, 3, 5)  # (bd)
                    S17 = swap(S16, 3, 4)  # (bc), here in the 3 index lies d but we name it b still, notice we use S7
                    if S == S17:
                        flag = True
                    S18 = rotate(S17, R3)
                    S19 = swap(S18, 0, 1)

                    if S == S19:
                        flag = True

                if permutation_boolean(M, E, 4, 5):  # If we have (cd)

                    S20 = swap(S3, 4, 5)
                    if S == S20:
                        flag = True
                    S21 = rotate(S20, R3)
                    S22 = swap(S21, 0, 1)

                    if S == S22:
                        flag = True

    if flag:
        non_chiral_states.append([S, E])
    else:
        chiral_states.append([S, E])

    return non_chiral_states, chiral_states

# chiral_states = []
# non_chiral_states = []
#
# for _ in range(10):
#
#     non_chiral_states_tmp, chiral_states_tmp = chirality_test()
#     non_chiral_states += non_chiral_states_tmp
#     chiral_states += chiral_states_tmp
#
# # The first slot is incremented for every true and the second for every false
# non_chiral_evaluation_on_logic_statement = [0, 0]
# chiral_evaluation_on_logic_statement = [0, 0]
#
# for non_chiral_state in non_chiral_states:
#     flag = logic_statement_true_for_non_chiral(non_chiral_state[0], non_chiral_state[1])
#     if flag:
#         non_chiral_evaluation_on_logic_statement[0] += 1
#     else:
#         non_chiral_evaluation_on_logic_statement[1] += 1
#
# for chiral_state in chiral_states:
#     flag = logic_statement_true_for_non_chiral(chiral_state[0], chiral_state[1])
#     if flag:
#         chiral_evaluation_on_logic_statement[0] += 1
#     else:
#         chiral_evaluation_on_logic_statement[1] += 1
#
# x = ['True', 'False']
# height_non_chiral = [non_chiral_evaluation_on_logic_statement[0], non_chiral_evaluation_on_logic_statement[1]]
# height_chiral = [chiral_evaluation_on_logic_statement[0], chiral_evaluation_on_logic_statement[1]]
#
# plt.bar(x, height_non_chiral, color = 'k', width = 0.1)
# plt.title('Non-chiral states evaluated on the logic statement\nwhich is true iff the input is non-chiral')
# plt.ylabel('Frequency')
# plt.show()
# # plt.savefig('non_chiral_collision_logic_statement_test.pdf', bbox_inches='tight')
# #
# plt.bar(x, height_chiral, color = 'k', width = 0.1)
# plt.title('Chiral states evaluated on the logic statement\nwhich is true iff the input is non-chiral')
# plt.ylabel('Frequency')
# plt.show()
# plt.savefig('chiral_collision_logic_statement_test.pdf', bbox_inches='tight')

# -------------------------------------------
