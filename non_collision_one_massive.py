import time
from numpy import pi, cos, sin, e, tan, arctan
from clifford.g3 import blades
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from random import uniform, seed, randint
from sympy import LeviCivita as eps
from main import parity, rotate, energy, epsilon, Gram_det_2
from pytest import approx

# Works only for a,b,c,d != 0 (it is very unlikely that any one will be randomly generated in momentum 0)

e1, e2, e3 = blades['e1'], blades['e2'], blades['e3']
I = e1^e2^e3

def dot(a,b):
    dot = a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
    return dot

def multivec_to_vec(a):
    return np.array([a[1], a[2], a[3]])

def swap(S,idx_1,idx_2):

    tmp = S[idx_1]
    S[idx_1] = S[idx_2]
    S[idx_2] = tmp

    return S

def logic_statement_true_for_non_chiral(S, E, mp, mq):

    p = np.array([mp, 0, 0, 0])
    q = np.array([mq, 0, 0, 0])
    a = multivec_to_vec(S[0])
    a = np.insert(a, 0, E[0])
    b = multivec_to_vec(S[1])
    b = np.insert(b, 0, E[1])
    c = multivec_to_vec(S[2])
    c = np.insert(c, 0, E[2])
    d = multivec_to_vec(S[3])
    d = np.insert(d, 0, E[3])
    RF = p + q

    case_1 = ((epsilon(a, b, c, RF) == approx(0)) and (epsilon(a, b, d, RF) == approx(0)) and (epsilon(b, c, d, RF) == approx(0)) and (
                epsilon(a, c, d, RF) == approx(0)))

    case_2 = ((dot(a,a) == approx(dot(b,b))) and (not (a == b).all()) and (Gram_det_2(a,RF,a,RF) == approx(Gram_det_2(b,RF,b,RF))) and (Gram_det_2(a,RF,c,RF) == approx((Gram_det_2(b,RF,c,RF)))) and (Gram_det_2(a,RF,d,RF) == approx(Gram_det_2(b,RF,d,RF))))

    def case_2_symmetry(a,b,c,d):
        return ((dot(a,a) == approx(dot(b,b))) and (not (a == b).all()) and (Gram_det_2(a,RF,a,RF) == approx(Gram_det_2(b,RF,b,RF))) and (Gram_det_2(a,RF,c,RF) == approx((Gram_det_2(b,RF,c,RF) == 0))) and (Gram_det_2(a,RF,d,RF) == approx(Gram_det_2(b,RF,d,RF))))

    case_3 = case_2_symmetry(a,c,b,d)
    case_4 = case_2_symmetry(a,d,c,b)
    case_5 = case_2_symmetry(b,c,a,d)
    case_6 = case_2_symmetry(b,d,c,a)
    case_7 = case_2_symmetry(c,d,a,b)

    case_8 = ((dot(a,a) == approx(dot(b,b))) and (dot(c,c) == approx(dot(d,d))) and (Gram_det_2(a,RF,a,RF) == approx(Gram_det_2(b,RF,b,RF))) and (Gram_det_2(c,RF,c,RF) == approx(Gram_det_2(d,RF,d,RF))) and ((((dot(a+b,a+b) == approx(4*dot(a,a)) == approx(4*dot(b,b)))) and ((dot(c+d,c+d) == approx(4*dot(c,c)) == approx(4*dot(d,d))))) or ((Gram_det_2(a+b,RF,c-d,RF) == approx(0)) or (not (c == d).all()))))

    def case_8_symmetry(a,b,c,d):
        return ((dot(a,a) == approx(dot(b,b))) and (dot(c,c) == approx(dot(d,d))) and (Gram_det_2(a,RF,a,RF) == approx(Gram_det_2(b,RF,b,RF))) and (Gram_det_2(c,RF,c,RF) == approx(Gram_det_2(d,RF,d,RF))) and ((((dot(a+b,a+b) == approx(4*dot(a,a)) == approx(4*dot(b,b)))) and ((dot(c+d,c+d) == approx(4*dot(c,c)) == approx(4*dot(d,d))))) or ((Gram_det_2(a+b,RF,c-d,RF) == approx(0)) or (not (c == d).all()))))

    case_9 = case_8_symmetry(a,c,b,d)
    case_10 = case_8_symmetry(a,d,b,c)

    case_19 = ((dot(a,a) == approx(dot(b,b)) == approx(dot(c,c)) == approx(dot(d,d))) and (Gram_det_2(a,RF,a,RF) == approx(Gram_det_2(b,RF,b,RF)) == approx(Gram_det_2(c,RF,c,RF)) == approx(Gram_det_2(d,RF,d,RF))) and approx((Gram_det_2(a - c,RF,b - d,RF) == 0)))

    def case_19_symmetry(a,b,c,d):
        return ((dot(a,a) == approx(dot(b,b)) == approx(dot(c,c)) == approx(dot(d,d))) and (Gram_det_2(a,RF,a,RF) == approx(Gram_det_2(b,RF,b,RF)) == approx(Gram_det_2(c,RF,c,RF)) == approx(Gram_det_2(d,RF,d,RF))) and approx((Gram_det_2(a - c,RF,b - d,RF) == 0)))

    case_20 = case_19_symmetry(a,b,d,c)
    case_21 = case_19_symmetry(a,c,b,d)
    case_22 = case_19_symmetry(a,c,d,b)
    case_23 = case_19_symmetry(a,d,c,b)
    case_24 = case_19_symmetry(a,d,b,c)

    return (case_1 or case_2 or case_3 or case_4 or case_5 or case_6 or case_7 or case_8 or case_9 or case_10
            or case_19 or case_20 or case_21 or case_22 or case_23 or case_24)

def construct_state():

    B = (uniform(-10, 10) * (e1 ^ e2) + uniform(-10, 10) * (e1 ^ e3) + uniform(-10, 10) * (e2 ^ e3)).normal()
    R = (e ^ (uniform(0, 2 * pi) * B)).normal()

    rdm = randint(0, 4)

    if rdm == 0:
        ma, mb, mc, md = randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 10)
        a = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3
        b = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3
        c = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3
        d = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3

    # Type of non-chiral (checked) (ab)
    if rdm == 1:
        ma, mb, mc, md = 1, 1, 2, 3
        a = e1
        b = e2
        c = e3
        d = (-e3)

    # Type of non-chiral (checked) (ab)(cd)
    if rdm == 2:
        ma, mb, mc, md = 1, 1, 2, 2
        n = uniform(-10, 10)*e1 + uniform(-10, 10)*e2 + uniform(-10, 10)*e3
        n = n.normal()
        a = uniform(-10, 10)*e1 + uniform(-10, 10)*e2 + uniform(-10, 10)*e3
        b = -n*a*n
        c = uniform(-10, 10)*e1 + uniform(-10, 10)*e2 + uniform(-10, 10)*e3
        d = -n*c*n

    # Type of non-chiral (checked) (abc)
    if rdm == 3:
        ma, mb, mc, md = 1, 1, 1, 3
        mag = uniform(-10, 10)
        a = mag*e1 + mag*e2
        b = mag*e1 - mag*e2
        c = -mag*e1 + mag*e2
        d = mag*e1 - mag*e2

    # Type of non-chiral (checked) (abcd)
    if rdm == 4:
        ma, mb, mc, md = 1, 1, 1, 1
        a = R*(-e1 + e3)*~R
        b = R*(e1 + e2)*~R
        c = R*(-e1 - e3)*~R
        d = R*(e1 - e2)*~R

    Ea, Eb, Ec, Ed = energy(ma, a), energy(mb, b), energy(mc, c), energy(md, d)

    M = [ma, mb, mc, md]
    E = [Ea, Eb, Ec, Ed]
    S = [a, b, c, d]

    print(rdm)

    return S, E, M

def permute_with_idx(M, E, idx_to_permute):

    same_mass_with_idx = [idx for idx in range(len(M)) if M[idx] == M[idx_to_permute] and idx != idx_to_permute]
    same_energy_with_idx = [idx for idx in range(len(E)) if E[idx] == approx(E[idx_to_permute]) and idx != idx_to_permute]

    return list(set(same_mass_with_idx) and set(same_energy_with_idx))

def permutation_boolean(M, E, idx_1, idx_2):

    if (M[idx_1] == M[idx_2]) and (E[idx_1] == E[idx_2]):
        return True
    else:
        return False

def chirality_test():

    chiral_states = []
    non_chiral_states = []

    S, E, M = construct_state()
    a, b, c, d = S[0], S[1], S[2], S[3] # p and q are 0 so we do not carry them around in the S list

    # These lists hold indices (as they appear in S) of the particles that can be permuted with a,b,c and d respectively
    permute_with_a = permute_with_idx(M, E, 0)
    permute_with_b = permute_with_idx(M, E, 1)
    permute_with_c = permute_with_idx(M, E, 2)
    permute_with_d = permute_with_idx(M, E, 3)

    S_parity = parity(S)  # Perform parity on the set of momenta

    flag = False # The flag is set to true if the state is non-chiral

    for idx in permute_with_a+[0]: # For every x that can be permuted with a, map x to a and perform (ax)

        x = S_parity[idx]
        n = a+x
        if n == 0: # If a and x are collinear we construct any v perpendicular to a and do a pi rotation in the plane av
            if a[2] != 0 or a[3] != 0:
                v = -I*(a^e1) # Cross product a x e1 in geometric algebra, I is the pseudoscalar I = e1e2e3
            else:
                v = -I*(a^e2)
            R1 = (a^v).normal()
        else:
            R1 = a.normal()*n.normal()

        S1 = rotate(S_parity, R1)
        S2 = swap(S1, 0, idx) # Performs (ax) where the index 0 corresponds to a in the S list

        # At this point a is fixed to its original state

        # Now we need to fix b in the plane perpendicular to a, if b has no component (a^b==0) there then we try c

        if a^b != 0: # If b is not collinear to a then it has components in the plane perpendicular to a

            for idx_plane in permute_with_b + [1]:

                if idx_plane == 0: # Since a is already fixed
                    continue

                # If we have a plane P with its perpendicular being a and we have a vector y, then the component of y
                # in the plane P is given in geometric algebra by the rejection y_plane = a*(a^y)

                b_plane = a*(a^b)
                y = S2[idx_plane]
                y_plane = a*(a^y)

                # If y is collinear with a then it has no component in the plane perpendicular to a, so even if it can
                # be permuted with b, we cannot map y_plane to b_plane and then swap because y_plane is 0

                if y_plane == 0:
                    continue
                else: # Here we map y_plane to b_plane and perform (yb)
                    m = y_plane + b_plane
                    if m == 0: # In this case we need a pi rotation in this plane we are working in
                        # We construct another vector in this plane with the cross product a x b_plane
                        w_plane = -I*(a^b_plane)
                        R2 = (w_plane^b_plane).normal()
                    else:
                        R2 = b_plane.normal()*m.normal()

                S3 = rotate(S2, R2) # Map y_plane to b_plane
                S4 = swap(S3, 1, idx_plane) # Perform (yb)

                if S == S4:
                    flag = True

                if permutation_boolean(M, E, 2, 3): # If we can perform (cd)
                    S5 = swap(S4, 2, 3)
                    if S == S5:
                        flag = True


        elif a^c != 0:

            for idx_plane in permute_with_c + [2]:

                # Since a and b are already fixed, b is fixed because we fixed a and to get into this elif we need
                # b to be collinear with a and hence when we fixed a we automatically fixed b
                if idx_plane == 0 or idx_plane == 1:
                    continue

                # If we have a plane P with its perpendicular being a and we have a vector y, then the component of y
                # in the plane P is given in geometric algebra by the rejection y_plane = a*(a^y)

                c_plane = a * (a ^ c)
                y = S2[idx_plane]
                y_plane = a * (a ^ y)

                # If y is collinear with a then it has no component in the plane perpendicular to a, so even if it can
                # be permuted with c, we cannot map y_plane to c_plane and then swap because y_plane is 0

                if y_plane == 0:
                    continue
                else:  # Here we map y_plane to c_plane and perform (yc)
                    m = y_plane + c_plane
                    if m == 0:  # In this case we need a pi rotation in this plane we are working in
                        # We construct another vector in this plane with the cross product a x c_plane
                        w_plane = -I * (a ^ c_plane)
                        R2 = (w_plane ^ c_plane).normal()
                    else:
                        R2 = c_plane.normal() * m.normal()

                S6 = rotate(S2, R2)  # Map y_plane to c_plane
                S7 = swap(S6, 2, idx_plane)  # Perform (yc)

                if S == S7:
                    flag = True

                if permutation_boolean(M, E, 1, 3): # If we can perform (bd)
                    S8 = swap(S7, 1, 3)
                    if S == S8:
                        flag = True

        else:
            flag = True # If a,b,c are collinear then the state is non-chiral

    if flag:
        non_chiral_states.append([S, E])
    else:
        chiral_states.append([S, E])

    return non_chiral_states, chiral_states

# non_chiral_states_list = []
# chiral_states_list = []
# for iterations in range(1000):
#     S, E, M = construct_state()
#     non_chiral_states, chiral_states = chirality_test()
#     non_chiral_states_list += non_chiral_states
#     chiral_states_list += chiral_states
#
# print(len(non_chiral_states_list), len(chiral_states_list))
#
# non_chiral_evaluation_on_logic_statement = [0, 0]
# for non_chiral_state in non_chiral_states_list:
#     mp, mq = uniform(1, 10), uniform(1, 10)
#     flag = logic_statement_true_for_non_chiral(non_chiral_state[0], non_chiral_state[1], mp, mq)
#     if flag:
#         non_chiral_evaluation_on_logic_statement[0] += 1
#     else:
#         non_chiral_evaluation_on_logic_statement[1] += 1
#
# chiral_evaluation_on_logic_statement = [0, 0]
# for chiral_state in chiral_states_list:
#     mp, mq = uniform(1, 10), uniform(1, 10)
#     flag = logic_statement_true_for_non_chiral(chiral_state[0], chiral_state[1], mp, mq)
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
# #plt.show()
# #plt.savefig('non_chiral_non_collision_logic_statement_test.pdf', bbox_inches='tight')
#
# plt.bar(x, height_chiral, color = 'k', width = 0.1)
# plt.title('Chiral states evaluated on the logic statement\nwhich is true iff the input is non-chiral')
# plt.ylabel('Frequency')
# #plt.show()
# plt.savefig('chiral_non_collision_logic_statement_test.pdf', bbox_inches='tight')


















