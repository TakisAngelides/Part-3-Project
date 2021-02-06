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

    E = [Ep, Eq, Ea, Eb, Ec, Ed]
    S = [p, q, a, b, c, d]

    return S,E

def construct_non_chiral_state():

    mp, mq, ma, mb, mc, md = 1, 1, 1, 1, 1, 1

    p = uniform(-10, 10) * e3
    q = -p

    a = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3
    b = a[1] * e1 - a[2] * e2 + a[3] * e3
    R = e**(uniform(0,2*pi)*e1*e2)
    c = R*a*~R
    d = c[1] * e1 - c[2] * e2 + c[3] * e3

    Ep, Eq, Ea, Eb, Ec, Ed = energy(mp, p), energy(mq, q), energy(ma, a), energy(mb, b), energy(mc, c), energy(md, d)

    E = [Ep, Eq, Ea, Eb, Ec, Ed]
    S = [p, q, a, b, c, d]

    return S, E

def chirality_test_distinct_masses():

    S, E = construct_state()

    # Construct the R2 rotor for particle a by projecting into the e1e2 plane
    a = S[2]
    a_proj = a - a[3] * e3
    a_rot = R1 * (-a) * ~R1
    a_rot_proj = a_rot - a_rot[3] * e3
    v = a_proj + a_rot_proj
    n = v.normal()
    R2 = a_proj.normal() * n

    # Map p, q and a back to themselves
    S1 = parity(S)
    S2 = rotate(S1, R1)
    S3 = rotate(S2, R2)
    final = S3

    return S, E, final

def swap(S,idx_1,idx_2):

    tmp = S[idx_1]
    S[idx_1] = S[idx_2]
    S[idx_2] = tmp

    return S

def chirality_test_same_masses():

    S, E = construct_non_chiral_state()

    # Map p, q and a back to themselves
    S1 = parity(S)
    S2 = rotate(S1, R1)
    S3 = swap(S2,2,3)  # input 2 corresponds to S[2] = a and 3 to b since S = p q a b c d
    S4 = swap(S3,4,5)  # input 4 corresponds to a and 5 to b since S = p q a b c d
    final = S4

    return S, E, final

e1, e2, e3 = blades['e1'], blades['e2'], blades['e3']

R1 = e1*e3  # Rotates -p and -q back to themselves

chiral_states = []
non_chiral_states = []

for _ in range(5):

    S, E, final = chirality_test_distinct_masses()

    if S == final:
        non_chiral_states.append([S, E])
    else:
        chiral_states.append([S, E])

    S, E, final = chirality_test_same_masses()

    if S == final:
        non_chiral_states.append([S, E])
    else:
        chiral_states.append([S, E])

# The first is slot is incremented for every true and the second for every false
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





# -------------------------------------------