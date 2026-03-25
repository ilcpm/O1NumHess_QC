"""Geometry and mode-preprocessing helpers for QC Hessian workflows.

This module centralizes coordinate math and translational/rotational mode
preprocessing, so ``O1NumHess_QC.py`` can focus on IO and workflow orchestration.
"""

import math
from typing import Tuple

import numpy as np


def bond(xyz: np.ndarray, A: int, B: int) -> float:
    """Return the bond length between two atoms.

    Args:
        xyz (np.ndarray): Cartesian coordinates with shape ``(natom, 3)``, in Bohr.
        A (int): Serial number of the first atom.
        B (int): Serial number of the second atom.

    Returns:
        float: Bond length in Bohr.
    """
    return np.linalg.norm(xyz[A, :] - xyz[B, :]) # pyright: ignore[reportReturnType]


def angle(xyz: np.ndarray, A: int, B: int, C: int) -> float:
    """Return the A-B-C angle in radians.

    Args:
        xyz (np.ndarray): Cartesian coordinates with shape ``(natom, 3)``, in Bohr.
        A (int): Serial number of the first atom.
        B (int): Serial number of the second atom.
        C (int): Serial number of the third atom.

    Returns:
        float: Angle in radians, in the range ``[0, pi]``.
    """
    BA = xyz[A, :] - xyz[B, :]
    BC = xyz[C, :] - xyz[B, :]
    return math.acos(np.dot(BA, BC) / np.linalg.norm(BA) / np.linalg.norm(BC))


def cosangle(xyz: np.ndarray, A: int, B: int, C: int) -> float:
    """Return the cosine of the A-B-C angle.

    Args:
        xyz (np.ndarray): Cartesian coordinates with shape ``(natom, 3)``, in Bohr.
        A (int): Serial number of the first atom.
        B (int): Serial number of the second atom.
        C (int): Serial number of the third atom.

    Returns:
        float: Cosine of the angle.
    """
    BA = xyz[A, :] - xyz[B, :]
    BC = xyz[C, :] - xyz[B, :]
    return np.dot(BA, BC) / np.linalg.norm(BA) / np.linalg.norm(BC)


def dihedral(xyz: np.ndarray, A: int, B: int, C: int, D: int) -> float:
    """Return the signed A-B-C-D dihedral angle in radians.

    Args:
        xyz (np.ndarray): Cartesian coordinates with shape ``(natom, 3)``, in Bohr.
        A (int): Serial number of the first atom.
        B (int): Serial number of the second atom.
        C (int): Serial number of the third atom.
        D (int): Serial number of the fourth atom.

    Returns:
        float: Dihedral angle in radians, in the range ``(-pi, pi]``.
    """
    AB = xyz[B, :] - xyz[A, :]
    CB = xyz[B, :] - xyz[C, :]
    CD = xyz[D, :] - xyz[C, :]
    normal1 = np.cross(AB, CB)
    normal2 = np.cross(CB, CD)
    dihed = math.acos(
        np.dot(normal1, normal2) / np.linalg.norm(normal1) / np.linalg.norm(normal2)
    )
    if np.dot(np.cross(normal1, normal2), CB) > 0.0:
        return -dihed
    return dihed


def mominertia(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the moments of inertia, principal axes, and barycenter.

    All atomic masses are assumed to be 1.

    Args
    ----
    xyz : np.ndarray
        Cartesian coordinates with shape ``(natom, 3)``, in Bohr.

    Returns
    -------
    I : np.ndarray
        Moments of inertia with shape ``(3,)``, assuming all atomic masses are 1, in Bohr^2.
    ax : np.ndarray
        Eigenvectors of the moment of inertia tensor, with shape ``(3, 3)``.
    barycen : np.ndarray
        Barycenter with shape ``(3,)``, in Bohr, assuming all atomic masses are 1.
    """ # Only numpy-style docstrings can accommodate multiple return values.
    N = xyz.shape[0]
    # Determine barycenter
    barycen = np.sum(xyz, axis=0) / N

    # Moment of inertia tensor
    Imat0 = 0.
    for k in range(N):
        vec = xyz[k,:] - barycen
        Imat0 += np.dot(vec,vec)
    Imat = Imat0*np.eye(3)
    for i in range(3):
        for j in range(3):
            for k in range(N):
                Imat[i,j] -= (xyz[k,i]-barycen[i])*(xyz[k,j]-barycen[j])

    I, ax = np.linalg.eig(Imat)
    return I, ax, barycen


def isLinear(xyz: np.ndarray, thresh: float = 1e-4) -> bool:
    """Determine whether a molecule is linear.

    Args:
        xyz (np.ndarray): Cartesian coordinates with shape ``(natom, 3)``, in Bohr.
        thresh (float): Threshold for the smallest eigenvalue of the moment of inertia tensor.

    Returns:
        bool: ``True`` if the minimum principal moment is smaller than ``thresh``.
    """
    I, _, __ = mominertia(xyz)
    return min(I) < thresh


def vecTransRot(xyz: np.ndarray, thresh_lin: float = 1e-4) -> Tuple[np.ndarray, int]:
    """Return the projection vectors for translational and rotational degrees of freedom.

    Note that the masses of all atoms are treated as the same. Therefore, the projection
    vectors for rotation are not the same as the rotational modes as would be given by
    a vibrational analysis.

    Args
    ----
        xyz (np.ndarray): Cartesian coordinates with shape ``(natom, 3)``, in Bohr.
        thresh_lin (float): Threshold for the smallest eigenvalue of the moment of
            inertia tensor; used for determining whether the molecule is linear.

    Returns
    -------
        P : np.ndarray
            Projection vectors with shape ``(3*natom, 6)``. When the molecule is
            linear, ``P[:,5]`` is zero.
        Ntr : int
            Number of translations and rotations, either 5 or 6.
    """ # Only numpy-style docstrings can accommodate multiple return values.
    I, ax, barycen = mominertia(xyz)

    N = xyz.shape[0]
    P = np.zeros([3*N,6])
    # Translations
    for i in range(N):
        P[3*i,0] = 1.0/math.sqrt(N)
        P[3*i+1,1] = 1.0/math.sqrt(N)
        P[3*i+2,2] = 1.0/math.sqrt(N)
    # Rotations
    Ntr = 3
    for j in range(3):
        if I[j] < thresh_lin:
            continue
        for i in range(N):
            P[3*i:3*i+3,Ntr] = np.cross(ax[:,j],xyz[i,:]-barycen)
        P[:,Ntr] /= np.linalg.norm(P[:,Ntr])
        Ntr += 1

    return P, Ntr


def symmetricBreathing(xyz: np.ndarray) -> np.ndarray:
    """Return the vibrational mode for the symmetric vibration of the whole molecule.

    This is the mode where the shape of the molecule is unchanged and the molecule
    merely changes its size.

    Args:
        xyz (np.ndarray): Cartesian coordinates with shape ``(natom, 3)``, in Bohr.

    Returns:
        np.ndarray: Normalized mode vector with shape ``(3*natom,)``.
    """
    _, __, barycen = mominertia(xyz)
    n_atom = xyz.shape[0]
    P = np.zeros(3 * n_atom)
    for i in range(n_atom):
        P[3 * i : 3 * i + 3] = xyz[i, :] - barycen
    P /= np.linalg.norm(P)
    return P


def rotationGradient(xyz: np.ndarray, g0: np.ndarray, Nrot: int) -> np.ndarray:
    """Compute the gradient change along the rotational axes.

    When a molecule is not at its equilibrium geometry, perturbing the Cartesian
    coordinates along the rotational directions can result in a non-zero second-order
    change of the energy, contrary to when the molecule is at its equilibrium geometry.

    This function calculates the gradient change when the molecule is perturbed by an
    infinitesimal step length dx along the rotational axes, divided by dx.

    Args:
        xyz (np.ndarray): Cartesian coordinates with shape ``(natom, 3)``, in Bohr.
            The caller must guarantee that there are at least 2 atoms.
        g0 (np.ndarray): Gradients at the equilibrium geometry, with shape ``(3*natom,)``.
        Nrot (int): Number of rotational degrees of freedom, either 2 or 3.

    Returns:
        np.ndarray: Gradient change vectors with shape ``(3*natom, Nrot)``.
    """
    N = xyz.shape[0]

    # generate the rotational axes
    I, ax, barycen = mominertia(xyz)
    if Nrot==2:
        ax = ax[:,I!=min(I)] # in this case the minimum I is guaranteed to be non-degenerate

    # loop over the rotational axes
    Prot = np.zeros([3*N,Nrot])
    g = np.zeros([3*N,Nrot])
    for j in range(Nrot):
        for i in range(N):
            Prot[3*i:3*i+3,j] = np.cross(ax[:,j],xyz[i,:]-barycen)
        Prot_norm = np.linalg.norm(Prot[:,j])

        for i in range(N):
            g[3*i:3*i+3,j] = np.cross(ax[:,j],g0[3*i:3*i+3])/Prot_norm

    return g

if __name__ == "__main__":
    # geometry: H2O2
    xyz = np.array([[0.,2.,0.],[0.,0.,0.],[0.,0.,3.],[2.,0.,3.]])
    atoms = np.array([1,8,8,1])

    print(bond(xyz,1,2)) # 3.0
    print(angle(xyz,0,1,2)) # 1.5707963267948966
    print(dihedral(xyz,0,1,2,3)) # -1.5707963267948966

    [I,ax,barycen] = mominertia(xyz)
    print(I) # [ 3.5755711 13.        13.4244289]
    print(ax)
    # [[ 3.50830058e-01 -7.07106781e-01 -6.13936699e-01]
    # [-3.50830058e-01 -7.07106781e-01  6.13936699e-01]
    # [ 8.68237606e-01  3.29961394e-15  4.96148626e-01]]
    print(barycen) # [0.5 0.5 1.5]

    print(vecTransRot(xyz))
    # (array([[ 5.00000000e-01,  0.00000000e+00,  0.00000000e+00,
    #    -4.10441542e-01,  2.94174203e-01, -4.54464235e-01],
    #   [ 0.00000000e+00,  5.00000000e-01,  0.00000000e+00,
    #     4.87203995e-02, -2.94174203e-01, -3.19050136e-01],
    #...
    #     1.85534247e-01,  3.92232270e-01, -1.67562058e-01]]), 6)
