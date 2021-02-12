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

    S, E, M = construct_state()

    S_parity = parity(S)  # Perform parity on the set of momenta

    same_mass_with_a = [idx for idx in range(len(M)) if M[idx] == M[2] and idx != 2]
    same_energy_with_a = [idx for idx in range(len(E)) if E[idx] == E[2] and idx != 2]
    # Holds the indices of particles that can be permuted with a
    permute_with_a = list(set(same_mass_with_a) and set(same_energy_with_a))

    same_mass_with_b = [idx for idx in range(len(M)) if M[idx] == M[3] and idx != 3]
    same_energy_with_b = [idx for idx in range(len(E)) if E[idx] == E[3] and idx != 3]
    # Holds the indices of particles that can be permuted with b
    permute_with_b = list(set(same_mass_with_b) and set(same_energy_with_b))

    # The indices of b,c,d in S are 3,4,5 which sum to 12. If I can permute a with eg b then I need to check if I can
    # silmutaneously permute c,d with indices [4,5] so the dictionary for 12-3 points to [4,5] where 3 is b
    swap_dictionary = {9: [4, 5], 8: [3, 5], 7: [3, 4]}

    a = S[2] # initial 3-momentum of a
    mp, mq, mc, md = M[0], M[1], M[4], M[5]
    Ec, Ed = E[4], E[5]

    if not permute_with_a:
        # For no permutations at all and for pq permutations
        # ie pq permutations don't change anything if there are no final state permutations
        a_12 = a - a[3] * e3
        R = a_12.normal() * e3
        S_final = rotate(S_parity, R)
        if S == S_final:
            non_chiral_states.append([S, E])
        elif permute_with_b and 2 not in permute_with_b:  # Check if non-chiral for (bc) or (bd)
            for idx in permute_with_b:
                if S == swap(S_final, 3, idx):
                    non_chiral_states.append([S, E])
        elif mc == md and Ec == Ed:  # Final check available (cd)
            if S == swap(S_final, 4, 5):
                non_chiral_states.append([S, E])
        else:
            chiral_states.append([S, E])
    else:
        flag = False  # Flag is set to true when one of the permutations managed to map everything back
        for idx in permute_with_a:
            # For ax permutations but no pq permutations
            Ea, Ex = E[2], E[idx]
            a, x = S[2], S[idx]
            # The other 2 final state particles besides a and x have indices
            idx_1 = swap_dictionary[12 - idx][0]
            idx_2 = swap_dictionary[12 - idx][1]
            if a[3] == x[idx] and mp != mq:
                R1 = e1 * e3  # Forced to do R1 since no pq permutation
                S_1 = rotate(S_parity, R1)
                x_new = S_1[3]
                x_12 = x_new - x_new[3] * e3
                a_initial = S[2]
                a_12 = a_initial - a_initial[3] * e3
                if a_12 == -x_12:  # If we need a pi rotation then use this construction
                    R2 = e1 * e2
                else:
                    n = (a_12 + x_12).normal()
                    R2 = a_12.normal() * n
                S_2 = rotate(S_1, R2)
                # (ax permutation) index 2 corresponds to a and index idx corresponds to x in the list S
                S_final = swap(S_2, 2, idx)
                if S == S_final:
                    non_chiral_states.append([S, E])
                    flag = True  # Flag is set to true when one of the permutations managed to map everything back
                    break
                # Last check to make here is to swap the other 2 final particles if we can
                elif E[idx_1] == E[idx_2] and M[idx_1] == M[idx_2] and S == swap(S_final, idx_1, idx_2):
                    non_chiral_states.append([S, E])
                    flag = True  # Flag is set to true when one of the permutations managed to map everything back
                    break
                else:
                    continue
            # For ax and pq permutations
            # If i have pq permutation and energies of a, x match and z component of a,x match in magnitude
            else:
                a_initial = S[2]
                a_12 = a_initial - a_initial[3] * e3
                if a[3] == -x[3]:  # if the z comp of a,b is opposite
                    x_new = S_parity[idx]
                    x_12 = x_new - x_new[3] * e3  # Project x into the 1-2 plane
                    # If the following is true it means we need to construct a rotor to do a pi rotation in the 1-2 plane
                    if a_12 == -x_12:
                        R2 = e1 * e2
                        S_final = rotate(S_parity, R2)  # Only use R2 and not R1
                        S_final = swap(S_final, 0, 1)  # Swap p and q
                        S_final = swap(S_final, 2, idx)  # Swap a and x
                        if S == S_final:
                            non_chiral_states.append([S, E])
                            flag = True
                            break
                        elif E[idx_1] == E[idx_2] and M[idx_1] == M[idx_2] and S == swap(S_final, idx_1, idx_2):
                            non_chiral_states.append([S, E])
                            flag = True  # Flag is set to true when one of the permutations managed to map everything back
                            break
                        else:
                            continue
                    else:
                        n = (a_12 + x_12).normal()
                        R2 = a_12.normal() * n
                        S_final = rotate(S_parity, R2)  # only use R2 and not R1
                        S_final = swap(S_final, 0, 1)  # swap p and q
                        S_final = swap(S_final, 2, idx)  # swap a and b
                        if S == S_final:
                            non_chiral_states.append([S, E])
                            flag = True
                            break
                        elif E[idx_1] == E[idx_2] and M[idx_1] == M[idx_2] and S == swap(S_final, idx_1, idx_2):
                            non_chiral_states.append([S, E])
                            flag = True  # Flag is set to true when one of the permutations managed to map everything back
                            break
                        else:
                            continue
                else:  # If the z component of a,x is equal
                    # Forced to do R1 since no pq permutation and so we have to bring the parity b
                    # which has opposite sign z back to positive sign
                    R1 = e1 * e3
                    S_1 = rotate(S_parity, R1)
                    x_new = S_1[idx]
                    x_12 = x_new - x_new[3] * e3
                    a_initial = S[2]
                    a_12 = a_initial - a_initial[3] * e3
                    if a_12 == -x_12:
                        R2 = e1 * e2
                    else:
                        n = (a_12 + x_12).normal()
                        R2 = a_12.normal() * n
                    S_2 = rotate(S_1, R2)
                    # (ax permutation) index 2 corresponds to a and index 3 corresponds to b in the list S
                    S_final = swap(S_2, 2, idx)
                    if S == S_final:
                        non_chiral_states.append([S, E])
                        flag = True
                        break
                    elif E[idx_1] == E[idx_2] and M[idx_1] == M[idx_2] and S == swap(S_final, idx_1, idx_2):
                        non_chiral_states.append([S, E])
                        flag = True  # Flag is set to true when one of the permutations managed to map everything back
                        break
                    else:
                        continue
        if not flag:  # If any available permutation has not managed to map everything back to itself then print chiral
            chiral_states.append([S, E])

    return non_chiral_states, chiral_states

chiral_states = []
non_chiral_states = []

for _ in range(10):

    non_chiral_states_tmp, chiral_states_tmp = chirality_test()
    non_chiral_states += non_chiral_states_tmp
    chiral_states += chiral_states_tmp

# The first slot is incremented for every true and the second for every false
non_chiral_evaluation_on_logic_statement = [0, 0]
chiral_evaluation_on_logic_statement = [0, 0]

for non_chiral_state in non_chiral_states:
    flag = logic_statement_true_for_non_chiral(non_chiral_state[0], non_chiral_state[1])
    if flag:
        non_chiral_evaluation_on_logic_statement[0] += 1
    else:
        non_chiral_evaluation_on_logic_statement[1] += 1

for chiral_state in chiral_states:
    flag = logic_statement_true_for_non_chiral(chiral_state[0], chiral_state[1])
    if flag:
        chiral_evaluation_on_logic_statement[0] += 1
    else:
        chiral_evaluation_on_logic_statement[1] += 1

x = ['True', 'False']
height_non_chiral = [non_chiral_evaluation_on_logic_statement[0], non_chiral_evaluation_on_logic_statement[1]]
height_chiral = [chiral_evaluation_on_logic_statement[0], chiral_evaluation_on_logic_statement[1]]

plt.bar(x, height_non_chiral, color = 'k', width = 0.1)
plt.title('Non-chiral states evaluated on the logic statement\nwhich is true iff the input is non-chiral')
plt.ylabel('Frequency')
plt.show()
# plt.savefig('non_chiral_collision_logic_statement_test.pdf', bbox_inches='tight')
#
plt.bar(x, height_chiral, color = 'k', width = 0.1)
plt.title('Chiral states evaluated on the logic statement\nwhich is true iff the input is non-chiral')
plt.ylabel('Frequency')
plt.show()
# plt.savefig('chiral_collision_logic_statement_test.pdf', bbox_inches='tight')

# -------------------------------------------