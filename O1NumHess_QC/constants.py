"""Chemistry constants used by QC-Hessian helpers."""

import numpy as np

# Useful constants
bohr2angstrom = 0.529177249
angstrom2bohr = 1.0 / bohr2angstrom

# Pyykko's covalent radii, elements count from 1
# The radius of oxygen was too low. A value of 0.68 is more appropriate,
# but for simplicity we don't change it
covalent_radii = np.array([
    0.0,                                                                                                            # Z=0 placeholder
    0.32, 0.46,                                                                                                     # Z=1-2 (H-He)
    1.33, 1.02, 0.85, 0.75, 0.71, 0.63, 0.64, 0.67,                                                                 # Z=3-10 (Li-Ne)
    1.55, 1.39, 1.26, 1.16, 1.11, 1.03, 0.99, 0.96,                                                                 # Z=11-18 (Na-Ar)
    1.96, 1.71, 1.48, 1.36, 1.34, 1.22, 1.19, 1.16, 1.1, 1.11, 1.12, 1.18, 1.24, 1.21, 1.21, 1.16, 1.14, 1.17,      # Z=19-36 (K-Kr)
    2.1, 1.85, 1.63, 1.54, 1.47, 1.38, 1.28, 1.25, 1.25, 1.2, 1.28, 1.36, 1.42, 1.4, 1.4, 1.36, 1.33, 1.31,         # Z=37-54 (Rb-Xe)
    2.32, 1.96, 1.8, 1.63, 1.76, 1.74, 1.73, 1.72, 1.68, 1.69, 1.68, 1.67, 1.66, 1.65, 1.64, 1.7, 1.62,             # Z=55-71 (Cs-Lu)
    1.52, 1.46, 1.37, 1.31, 1.29, 1.22, 1.23, 1.24, 1.33, 1.44, 1.44, 1.51, 1.45, 1.47, 1.42,                       # Z=72-86 (Hf-Rn)
    2.23, 2.01, 1.86, 1.75, 1.69, 1.7, 1.71, 1.72, 1.66, 1.66, 1.68, 1.68, 1.65, 1.67, 1.73, 1.76, 1.61,            # Z=87-103 (Fr-Lr)
]) * angstrom2bohr

# vdw radii: from UFF
# For large conjugated systems, it's beneficial to divide the hydrogen
# radius by 2. For other systems it's generally bad to do so
vdw_radii = np.array([
    0.0,                                                                                                                            # Z=0 placeholder
    2.886, 2.362,                                                                                                                   # Z=1-2 (H-He)
    2.451, 2.745, 4.083, 3.851, 3.66, 3.5, 3.364, 3.243,                                                                            # Z=3-10 (Li-Ne)
    2.983, 3.021, 4.499, 4.295, 4.147, 4.035, 3.947, 3.868,                                                                         # Z=11-18 (Na-Ar)
    3.812, 3.399, 3.295, 3.175, 3.144, 3.023, 2.961, 2.912, 2.872, 2.834, 3.495, 2.763, 4.383, 4.28, 4.23, 4.205, 4.189, 4.141,     # Z=19-36 (K-Kr)
    4.114, 3.641, 3.345, 3.124, 3.165, 3.052, 2.998, 2.963, 2.929, 2.899, 3.148, 2.848, 4.463, 4.392, 4.42, 4.47, 4.5, 4.404,       # Z=37-54 (Rb-Xe)
    4.517, 3.703, 3.522, 3.556, 3.606, 3.575, 3.547, 3.52, 3.493, 3.368, 3.451, 3.428, 3.409, 3.391, 3.374, 3.355, 3.64,            # Z=55-71 (Cs-Lu)
    3.141, 3.17, 3.069, 2.954, 3.12, 2.84, 2.754, 3.293, 2.705, 4.347, 4.297, 4.37, 4.709, 4.75, 4.765,                             # Z=72-86 (Hf-Rn)
    4.9, 3.677, 3.478, 3.396, 3.424, 3.395, 3.424, 3.424, 3.381, 3.326, 3.339, 3.313, 3.299, 3.286, 3.274, 3.248, 3.236,            # Z=87-103 (Fr-Lr)
]) / 2.0 * angstrom2bohr

# Element symbols indexed by atomic number; index 0 is a placeholder
periodic_table = [
    'X',                                                                                                        # Z=0 placeholder
    'H', 'He',                                                                                                  # Z=1-2 (period 1)
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',                                                                  # Z=3-10 (period 2)
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',                                                               # Z=11-18 (period 3)
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',   # Z=19-36 (period 4)
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',   # Z=37-54 (period 5)
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',       # Z=55-71 (period 6)
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',                    # Z=72-86 (period 6 cont.)
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',        # Z=87-103 (period 7)
]
