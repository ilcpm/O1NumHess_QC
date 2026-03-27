"""
Swart model Hessian (modified).

Implemented by Zikuan Wang

Calculate Swart's model Hessian (slightly modified to suit our needs).
This is used in generating the initial displacement directions of O1NumHess.

Original reference: Swart, Bickelhaupt, IJQC, 2006, 106, 2536.

Different from the original implementation, herein we:
    1. treat all pairs of atoms as bonds
    2. add linear angles
    3. do not penalize near-linear bonds

Warning:
    1. some extremely loose, linear molecules might prove problematic
    2. the frequencies of loose modes tend to be overestimated
"""
import numpy as np
import math
from .constants import covalent_radii
from .geometry import bond, cosangle

def Bmat_bond(xyz: np.ndarray, i: int, j: int) -> np.ndarray:
    """Wilson B-matrix for a bond length internal coordinate.

    Args:
        xyz (np.ndarray): Cartesian coordinates in Bohr, shape ``(natom, 3)``.
        i (int): Serial number of atom i.
        j (int): Serial number of atom j.

    Returns:
        np.ndarray: B-matrix for bond i-j, shape ``(6,)``.
    """
    vec = xyz[i, :] - xyz[j, :]
    l = np.linalg.norm(vec)
    # \partial l/\partial x, etc.
    B = np.zeros(6)
    B[0:3] = vec/l
    B[3:6] = -vec/l
    return B

def Bmat_angle(xyz: np.ndarray, i: int, j: int, k: int) -> np.ndarray:
    """Wilson B-matrix for a non-linear angle internal coordinate.

    Args:
        xyz (np.ndarray): Cartesian coordinates in Bohr, shape ``(natom, 3)``.
        i (int): Serial number of atom i.
        j (int): Serial number of atom j (central atom).
        k (int): Serial number of atom k.

    Returns:
        np.ndarray: B-matrix for angle i-j-k, shape ``(9,)``.
    """
    vec1 = xyz[i, :] - xyz[j, :]
    vec2 = xyz[k, :] - xyz[j, :]
    l1 = np.linalg.norm(vec1)
    l2 = np.linalg.norm(vec2)
    nvec1 = vec1/l1
    nvec2 = vec2/l2
    # angle = acos(nvec1 \cdot nvec2)
    # acos(x)' = 1/sqrt(1-x^2)
    # \therefore angle' = -(nvec1 \cdot nvec2)'/sqrt(1-(nvec1 \cdot nvec2)^2)
    dl = np.zeros([2,6]) # dl(a,b) = d la/d b
    dl[0,0:3] = nvec1
    dl[0,3:6] = -nvec1
    dl[1,0:3] = nvec2
    dl[1,3:6] = -nvec2
    dnvec = np.zeros([2,3,6]) # nvec(a,b,c) = d nveca(b) /d c
    for ii in range(6):
        dnvec[0,0:3,ii] = -nvec1 * dl[0,ii]/l1
        dnvec[1,0:3,ii] = -nvec2 * dl[1,ii]/l2
    for ii in range(3):
        dnvec[0,ii,ii] += 1.0/l1
        dnvec[1,ii,ii] += 1.0/l2
        dnvec[0,ii,ii+3] -= 1.0/l1
        dnvec[1,ii,ii+3] -= 1.0/l2
    # Assemble the derivatives of the angle
    dinprod = np.zeros(9)
    for ii in range(3):
        dinprod[ii] = np.dot(dnvec[0,:,ii],nvec2)
        dinprod[ii+3] = np.dot(dnvec[0,:,ii+3],nvec2) + np.dot(dnvec[1,:,ii+3],nvec1)
        dinprod[ii+6] = np.dot(dnvec[1,:,ii],nvec1)

    # regularization for linear angles
    B = -dinprod/math.sqrt(max(1e-15,1.0-np.dot(nvec1,nvec2)**2))
    return B

def Bmat_linangle(xyz: np.ndarray, i: int, j: int, k: int) -> np.ndarray:
    """Wilson B-matrix for a linear angle internal coordinate.

    Args:
        xyz (np.ndarray): Cartesian coordinates in Bohr, shape ``(natom, 3)``.
        i (int): Serial number of atom i.
        j (int): Serial number of atom j (central atom).
        k (int): Serial number of atom k.

    Returns:
        np.ndarray: B-matrix for linear angle i-j-k, shape ``(2, 9)``.
    """
    vec1 = xyz[i, :] - xyz[j, :]
    vec2 = xyz[k, :] - xyz[j, :]
    l1 = np.linalg.norm(vec1)
    l2 = np.linalg.norm(vec2)
    nvec1 = vec1/l1
    nvec2 = vec2/l2
    # First, generate two random vectors perpendicular to the bonds: vn and vn2
    vn = np.cross(vec1,vec2)
    nvn = np.linalg.norm(vn)
    if nvn < 1e-15: # vec1 and vec2 are collinear
        vn = np.array([1.0, 0.0, 0.0])
        vn = vn - np.dot(vn, vec1) / l1**2 * vec1
        nvn = np.linalg.norm(vn)
        if nvn < 1e-15:  # vec1 is along the x axis
            vn = np.array([0.0, 1.0, 0.0])
            vn = vn - np.dot(vn, vec1) / l1**2 * vec1
            nvn = np.linalg.norm(vn)
            # Now nvn should be usable - otherwise it means that norm(vec1)==0

    vn = vn / nvn
    vn2 = np.cross(vec1 - vec2, vn)
    vn2 = vn2 / np.linalg.norm(vn2)
    # Then, assuming that the angle is exactly linear, generate the B matrix elements
    # Note: for non-ideal linear angles, B(1,:) is the traditional
    # angle-bending mode while B(2,:) is the out-of-plane,
    # rotational-invariance-violating mode
    B = np.zeros([2, 9])
    B[1, 0:3] = vn / l1
    B[1, 6:9] = vn / l2
    B[1, 3:6] = -B[1, 0:3] - B[1, 6:9]
    B[0, 0:3] = vn2 / l1
    B[0, 6:9] = vn2 / l2
    B[0, 3:6] = -B[0, 0:3] - B[0, 6:9]
    return B

def Swart(xyz: np.ndarray, atomic_num: np.ndarray) -> np.ndarray:
    """Modified Swart model Hessian.

    Args:
        xyz (np.ndarray): Cartesian coordinates in Bohr, shape ``(natom, 3)``.
        atomic_num (np.ndarray): Atomic numbers, shape ``(natom,)``.

    Returns:
        np.ndarray: Hessian matrix, shape ``(3*natom, 3*natom)``.
    """

    # Number of atoms
    N = atomic_num.size

    covrad = np.zeros(N)
    for i in range(N):
        covrad[i] = covalent_radii[atomic_num[i]]

    distance = np.zeros([N, N])
    screenfunc = np.zeros([N, N])

    for i in range(N):
        for j in range(i + 1, N):
            distance[i, j] = bond(xyz, i, j)
            distance[j, i] = distance[i, j]
            equildist = covrad[i] + covrad[j]
            screenfunc[i, j] = math.exp(1.0 - distance[i, j] / equildist)
            screenfunc[j, i] = screenfunc[i, j]

    # Hessian
    H = np.zeros([3 * N, 3 * N])

    # bonds - we keep all bonds, no matter how long they are
    for i in range(N):
        for j in range(i + 1, N):
            Hint = 0.35 * screenfunc[i, j] ** 3
            B = Bmat_bond(xyz, i, j)
            rangeint = list(range(3 * i, 3 * (i + 1))) + list(range(3 * j, 3 * (j + 1)))
            H[np.ix_(rangeint, rangeint)] += Hint * np.outer(B, B)

    # angles
    wthr = 0.3
    f = 0.12
    tolth = 0.2
    eps1 = wthr**2
    eps2 = wthr**2 / math.exp(1)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if screenfunc[i, j] < eps2:
                continue
            for k in range(i + 1, N):
                if k == j:
                    continue
                s_ijjk = screenfunc[i, j] * screenfunc[j, k]
                if s_ijjk < eps1:
                    continue

                costh = cosangle(xyz, i, j, k)
                sinth = math.sqrt(max(0.0, 1.0 - costh**2))
                # the value 0.075 seems better than the original value 0.15
                Hint = 0.075 * s_ijjk**2 * (f + (1 - f) * sinth) ** 2
                B = Bmat_angle(xyz, i, j, k)

                if costh > 1 - tolth:
                    th1 = 1.0 - costh
                else:
                    th1 = 1.0 + costh

                rangeint = (
                    list(range(3 * i, 3 * (i + 1)))
                    + list(range(3 * j, 3 * (j + 1)))
                    + list(range(3 * k, 3 * (k + 1)))
                )

                if th1 < tolth:  # i-j-k is close to either 180 degrees or 0 degree
                    scalelin = (1.0 - (th1 / tolth) ** 2) ** 2
                    if costh > 1 - tolth:  # i-j-k angle is close to 180 degrees
                        # for linear angle, there is one additional internal coordinate
                        Blin = Bmat_linangle(xyz, i, j, k)
                        B = scalelin * Blin[0, :] + (1.0 - scalelin) * B
                        H[np.ix_(rangeint, rangeint)] += Hint * np.outer(Blin[1, :], Blin[1, :])
                    else:  # i-j-k angle is close to 0 degree
                        B = (1.0 - scalelin) * B
                H[np.ix_(rangeint, rangeint)] += Hint * np.outer(B, B)

    # Numerically, we found that even without additional dihedral and inversion terms,
    # the Hessian is already good enough

    return H

if __name__ == "__main__":
    # test

    # geometry: H2O2
    xyz = np.array([[0.,2.,0.],[0.,0.,0.],[0.,0.,3.],[2.,0.,3.]])
    atoms = np.array([1,8,8,1])

    H = Swart(xyz,atoms)
    print(H)

    # Reference:
    #[[ 1.45100659e-03 -5.98194146e-05  1.00649622e-03 -4.08268394e-04
    #  -6.11178062e-04 -9.16767093e-04 -9.82918780e-04 -6.11178062e-04
    #   9.16767093e-04 -5.98194146e-05  1.28217554e-03 -1.00649622e-03]
    # [-5.98194146e-05  2.54271977e-01 -8.31705023e-03  0.00000000e+00
    #  -2.48719845e-01  6.79203323e-04  0.00000000e+00 -5.49231333e-03
    #   7.54811778e-03  5.98194146e-05 -5.98194146e-05  8.97291220e-05]
    # [ 1.00649622e-03 -8.31705023e-03  2.28988945e-02 -9.16767093e-04
    #  -6.94307405e-03 -1.12379384e-02  0.00000000e+00  1.42536281e-02
    #  -1.15263625e-02 -8.97291220e-05  1.00649622e-03 -1.34593683e-04]
    # [-4.08268394e-04  0.00000000e+00 -9.16767093e-04  1.03541605e-02
    #   6.11178062e-04  8.59418131e-03 -4.45357874e-03  0.00000000e+00
    #   6.57621384e-03 -5.49231333e-03 -6.11178062e-04 -1.42536281e-02]
    # [-6.11178062e-04 -2.48719845e-01 -6.94307405e-03  6.11178062e-04
    #   2.54156342e-01  6.81377761e-03  0.00000000e+00 -4.45357874e-03
    #   1.29296434e-04  0.00000000e+00 -9.82918780e-04  0.00000000e+00]
    # [-9.16767093e-04  6.79203323e-04 -1.12379384e-02  8.59418131e-03
    #   6.81377761e-03  1.84144835e-01 -1.29296434e-04 -6.57621384e-03
    #  -1.61380534e-01 -7.54811778e-03 -9.16767093e-04 -1.15263625e-02]
    # [-9.82918780e-04  0.00000000e+00  0.00000000e+00 -4.45357874e-03
    #   0.00000000e+00 -1.29296434e-04  2.54156342e-01  6.11178062e-04
    #  -6.81377761e-03 -2.48719845e-01 -6.11178062e-04  6.94307405e-03]
    # [-6.11178062e-04 -5.49231333e-03  1.42536281e-02  0.00000000e+00
    #  -4.45357874e-03 -6.57621384e-03  6.11178062e-04  1.03541605e-02
    #  -8.59418131e-03  0.00000000e+00 -4.08268394e-04  9.16767093e-04]
    # [ 9.16767093e-04  7.54811778e-03 -1.15263625e-02  6.57621384e-03
    #   1.29296434e-04 -1.61380534e-01 -6.81377761e-03 -8.59418131e-03
    #   1.84144835e-01 -6.79203323e-04  9.16767093e-04 -1.12379384e-02]
    # [-5.98194146e-05  5.98194146e-05 -8.97291220e-05 -5.49231333e-03
    #   0.00000000e+00 -7.54811778e-03 -2.48719845e-01  0.00000000e+00
    #  -6.79203323e-04  2.54271977e-01 -5.98194146e-05  8.31705023e-03]
    # [ 1.28217554e-03 -5.98194146e-05  1.00649622e-03 -6.11178062e-04
    #  -9.82918780e-04 -9.16767093e-04 -6.11178062e-04 -4.08268394e-04
    #   9.16767093e-04 -5.98194146e-05  1.45100659e-03 -1.00649622e-03]
    # [-1.00649622e-03  8.97291220e-05 -1.34593683e-04 -1.42536281e-02
    #   0.00000000e+00 -1.15263625e-02  6.94307405e-03  9.16767093e-04
    #  -1.12379384e-02  8.31705023e-03 -1.00649622e-03  2.28988945e-02]]


