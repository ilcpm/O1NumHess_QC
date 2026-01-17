# O1NumHess_QC

 This is a Python library related to quantum chemistry calculations. This module interfaces with chemical software such as BDF, ORCA, etc., as well as the O1NumHess library, implementing the functionality to calculate the Hessian matrix by only needing to calculate gradients O(1) times when calling chemical software like BDF.

Users can calculate the Hessian matrix of molecules by specifying the `.xyz` molecular coordinate file and the corresponding input file for gradient calculations.

O1NumHess is unrelated to quantum chemistry and calculates the Hessian matrix by accepting a vector x and a user-provided gradient function g. O1NumHess perturbs each component of the input vector x separately, calls function g to calculate the gradient after each perturbation, and finally uses the gradients calculated from multiple perturbations to derive the Hessian.

Details of the O1NumHess algorithm, as well as preliminary benchmark results, can be found in our preprint paper: https://arxiv.org/abs/2508.07544

<!-- TODO 其他软件 O1的复杂度 -->

## requirement

* Linux System
* python >= 3.6
* numpy
* O1NumHess

<!-- TODO link -->

## install

```bash
python3 setup.py install
```

this will create config files in `~/.O1NumHess_QC`, copy the files `xxx_config_example.py` to `xxx_config.py` and modify it with your own condition.

Note that the above command installs the package for all users, and requires root privileges. If the user does not have access to root privileges, or if it is not desired to install the package for all users, then one should use the following command instead:

```bash
python3 setup.py install --prefix ~/.local
```

## example

Detailed concept and instruction can be found in the documentation.

<!-- TODO documentation link -->

Here is an example shows how to use O1NumHess_QC and BDF to calculate Hessian.

test files:

* `benzene.xyz`:

    ```plain
    12

    C         -0.26396        0.29101        0.87006
    C         -0.72047       -0.95180        0.41795
    C          0.12718       -2.06420        0.45530
    C          1.43129       -1.93382        0.94486
    C          1.88770       -0.69106        1.39714
    C          1.04011        0.42134        1.35968
    H         -0.91966        1.15163        0.84107
    H          1.39331        1.38276        1.70952
    H          2.89652       -0.59018        1.77601
    H          2.08713       -2.79433        0.97377
    H         -1.72939       -1.05266        0.03928
    H         -0.22596       -3.02564        0.10546
    ```

* `benzene.inp`:

    ```plain
    $COMPASS
    Title
    benzene grad
    Basis
    cc-pvdz
    Geometry
    file=benzene.xyz
    End geometry
    $END

    $xuanyuan
    $end

    $scf
    RKS
    dft
    B3LYP
    spinmulti
    1
    $end

    $resp
    geom
    $end
    ```

* `testQC.py`:

    ```python
    import numpy as np
    from O1NumHess_QC import O1NumHess_QC

    # read XYZ from file
    qc = O1NumHess_QC("../benzene.xyz")
    # qc = O1NumHess_QC("../benzene.xyz", unit="angstrom")  # specify the unit of xyz file manually
    # print(qc.xyz_angstrom)

    # parallel calculate Hessian
    hessian: np.ndarray = qc.calcHessian_BDF(
        method = "single",
        delta = 1e-3,
        core = 4,
        mem = "4G",
        inp = "../benzene.inp",
        encoding = "utf-8",
        tempdir = "~/tmp",      # BDF tempdir, each calculate will generate a subfolder at there
        task_name = "abc",      # all the output file will start with this string
        config_name = "",       # config name in your ~/.O1NumHess_QC folder config file
    )
    print(hessian)
    ```

it's recommended to run like this:

```plain
.
├── BDF_RUN/         # run in this folder
├── benzene.inp
├── benzene.xyz
└── testQC.py
```

run in `BDF_RUN` folder with `python3 ../testQC.py`, so that all the output file will in `BDF_RUN`.

## unit test

run this command under project root folder:

```python
python3 -m unittest
```

or simply run files like `test_BDF.py` in the `test` folder.
