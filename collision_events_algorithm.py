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

    same_mass_with_a = [idx for idx in range(len(M)) if M[idx] == M[2] and idx != 2]
    same_energy_with_a = [idx for idx in range(len(E)) if E[idx] == E[2] and idx != 2]
    # Holds the indices of particles that can be permuted with a
    permute_with_a = list(set(same_mass_with_a) and set(same_energy_with_a))

    same_mass_with_b = [idx for idx in range(len(M)) if M[idx] == M[3] and idx != 3]
    same_energy_with_b = [idx for idx in range(len(E)) if E[idx] == E[3] and idx != 3]
    # Holds the indices of particles that can be permuted with b
    permute_with_b = list(set(same_mass_with_b) and set(same_energy_with_b))

    same_mass_with_c = [idx for idx in range(len(M)) if M[idx] == M[4] and idx != 4]
    same_energy_with_c = [idx for idx in range(len(E)) if E[idx] == E[4] and idx != 4]
    # Holds the indices of particles that can be permuted with c
    permute_with_c = list(set(same_mass_with_c) and set(same_energy_with_c))

    same_mass_with_d = [idx for idx in range(len(M)) if M[idx] == M[5] and idx != 5]
    same_energy_with_d = [idx for idx in range(len(E)) if E[idx] == E[5] and idx != 5]
    # Holds the indices of particles that can be permuted with b
    permute_with_d = list(set(same_mass_with_d) and set(same_energy_with_d))

    permutation_dictionary = {3: permute_with_b, 4: permute_with_c, 5: permute_with_d}

    S_parity = parity(S)  # Perform parity on the set of momenta

    flag = False

    if M[0] != M[1] or E[0] != E[1]: # If we cant permute p and q

        # Map p and q back with R = e1e3

        R1 = e1*e3
        S1 = rotate(S_parity, R1)

        # For every x that can be permuted with a, rotate x to a in the 1-2 plane and swap them

        for idx in permute_with_a:

            idx_list = [3, 4, 5] # To be used later to check if we can permute the other 2 other than a and x
            idx_list.remove(idx)

            x = S1[idx]
            x_proj = x - x[3]*e3 # Project x into the 1-2 plane
            a = S[2]
            a_proj = a - a[3]*e3 # Project original a into the 1-2 plane
            n = (a_proj+x_proj).normal()
            if n == 0: # If a_proj and x_proj are anti-parallel we need R2 to be a pi rotation
                R2 = e1*e2
            else:
                R2 = a_proj.normal()*n
            S2 = rotate(S1, R2)
            S3 = swap(S2, 2, idx) # Swap a and x where the index of a is 2 and the index of x is idx

            if S == S3:
                flag = True # The flag is set to true if the state is non-chiral

            # Check if the other 2 other a and x can be permuted

            idx_1, idx_2 = idx_list[0], idx_list[1]
            permute_with_1 = permutation_dictionary[idx_1]
            if idx_2 in permute_with_1:
                S4 = swap(S3, idx_1, idx_2)
                if S == S4:
                    flag = True

    else: # If we can permute p and q

        # Map p and q back with (pq) and without using R = e1e3

        S1 = swap(S_parity.copy(), 0, 1) # Swaps p and q

        # Now we act as if p and q are fixed, although we know that we can perform e2e3 or e1e3 while using (pq)
        # Using e1e3 has been tried in the case where (pq) is not available so non-chirality will be catched there if
        # the state needs indeed e1e3 to be mapped back

        # We now go to the subspace (1-2 plane) perpendicular to p,q and map a as before

        for idx in permute_with_a:

            idx_list = [3, 4, 5] # To be used later to check if we can permute the other 2 other than a and x
            idx_list.remove(idx)

            x = S1[idx]
            x_proj = x - x[3]*e3 # Project x into the 1-2 plane
            a_proj = S[2] - S[2][3]*e3 # Project original a into the 1-2 plane
            n = (a_proj+x_proj).normal()
            if n == 0: # If a_proj and x_proj are anti-parallel we need R2 to be a pi rotation
                R2 = e1*e2
            else:
                R2 = a_proj.normal()*n
            S2 = rotate(S1, R2)
            S3 = swap(S2, 2, idx) # Swap a and x where the index of a is 2 and the index of x is idx

            if S == S3:
                flag = True # The flag is set to true if the state is non-chiral

            # Check if the other 2 other a and x can be permuted

            idx_1, idx_2 = idx_list[0], idx_list[1]
            permute_with_1 = permutation_dictionary[idx_1]
            if idx_2 in permute_with_1:
                S4 = swap(S3, idx_1, idx_2)
                if S == S4:
                    flag = True

        # Now try with using R = e1e3 with no (pq) even though we can do (pq)

        # Map p and q back with R = e1e3

        R1 = e1*e3
        S1 = rotate(S_parity, R1)  # Swaps p and q

        # Now we act as if p and q are fixed, although we know that we can perform e2e3 or e1e3 while using (pq)

        # We now go to the subspace (1-2 plane) perpendicular to p,q and map a as before

        for idx in permute_with_a:

            idx_list = [3, 4, 5]  # To be used later to check if we can permute the other 2 other than a and x
            idx_list.remove(idx)

            x = S1[idx]
            x_proj = x - x[3] * e3  # Project x into the 1-2 plane
            a = S[2]
            a_proj = a - a[3] * e3  # Project original a into the 1-2 plane
            n = (a_proj + x_proj).normal()
            if n == 0:  # If a_proj and x_proj are anti-parallel we need R2 to be a pi rotation
                R2 = e1 * e2
            else:
                R2 = a_proj.normal() * n
            S2 = rotate(S1, R2)
            S3 = swap(S2, 2, idx)  # Swap a and x where the index of a is 2 and the index of x is idx

            if S == S3:
                flag = True  # The flag is set to true if the state is non-chiral

            # Check if the other 2 other a and x can be permuted

            idx_1, idx_2 = idx_list[0], idx_list[1]
            permute_with_1 = permutation_dictionary[idx_1]
            if idx_2 in permute_with_1:
                S4 = swap(S3, idx_1, idx_2)
                if S == S4:
                    flag = True

    # We now try without using permutations for a. We can map p,q,a simultaneously and check (bcd)

    a = S[2]
    a_proj = a - a[3]*e3
    R = a_proj.normal()*e3 # Performs a pi rotation in the plane spanned by the 3 axis and a
    S1 = rotate(S_parity, R)

    if S == S1:
        flag = True

    # Also check (bcd)

    if permute_with_b and (2 not in permute_with_b): # If b can be permuted with c or d (but not with a which is fixed)

        for idx in permute_with_b:

            S2 = swap(S1, 3, idx)

            if S == S2: # Remember b has index 3 in the state list S
                flag = True

            if 4 in permute_with_d: # If c (with index 4 in the S list) can be permuted with d

                S3 = swap(S2, 4, 5)

                if S == S3:
                    flag = True

    if permute_with_c and (2 not in permute_with_c):  # If c can be permuted with b or d (but not with a which is fixed)

        for idx in permute_with_c:

            S2 = swap(S1, 4, idx) # The index 4 corresponds to c

            if S == S2:
                flag = True

            if 3 in permute_with_d:  # If b (with index 3 in the S list) can be permuted with d

                S3 = swap(S2, 3, 5)

                if S == S3:
                    flag = True

    if permute_with_d and (2 not in permute_with_d):  # If d can be permuted with b or c (but not with a which is fixed)

        for idx in permute_with_d:

            S2 = swap(S1, 5, idx) # The index 5 corresponds to d

            if S == S2:
                flag = True

            if 3 in permute_with_c:  # If b (with index 3 in the S list) can be permuted with c

                S3 = swap(S2, 3, 4)

                if S == S3:
                    flag = True

    if flag:
        non_chiral_states.append([S,E])
    else:
        chiral_states.append([S,E])

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

print(chirality_test())
