import numpy as np

from textwrap import dedent
from typing import List, Sequence, Tuple, Union
from pathlib import Path
import os
import re
import time

from O1NumHess import O1NumHess
from .utils import *
from .utils import getAbsPath, getConfig

class O1NumHess_QC:

    def __init__(
        self,
        xyz_path: Union[str, Path],
        unit: str = "angstrom",
        encoding: str = "utf-8",
        verbosity: int = 0,
    ):
        self.verbosity = verbosity
        # read the XYZ file, get path, coordinates and atoms
        self.xyz_path, self.xyz_bohr, self.atoms = self._readXYZ(xyz_path, encoding, unit)
        # self.atoms is tuple of element str
        if self.verbosity > 1:
            print("Successfully read coordinates from %s" % self.xyz_path)
            print("Atomic coordinates in Bohr:")
            for i_atom in range(len(self.atoms)):
                print("%5s %20.12f %20.12f %20.12f"%(self.atoms[i_atom], self.xyz_bohr[i_atom,0], \
                    self.xyz_bohr[i_atom,1], self.xyz_bohr[i_atom,2]))

        # generate atomic numbers, for future use
        self.atomic_num = self._atoms2AtomicNum(self.atoms)

    def setVerbosity(self, verbosity: int):
        self.verbosity = verbosity

    @property
    def xyz_angstrom(self) -> np.ndarray:
        return self.xyz_bohr * bohr2angstrom

    @property
    def _effDistMat(self) -> np.ndarray:
        """
        Effective distance matrix.
        Output: distmat(np.ndarray, dimensions(3*self.atoms.size,3*self.atoms.size))
                The nine elements distmat([3*i:3*i+2,3*j:3*j+2]) are the same;
                They are the distance between atoms i and j (Bohr) minus the sum of
                their vdW radii
        """
        N = len(self.atoms)
        distmat = np.zeros([3*N,3*N])
        for i in range(N):
            for j in range(i,N):
                R = bond(self.xyz_bohr,i,j) - vdw_radii[self.atomic_num[i]] - vdw_radii[self.atomic_num[j]]
                distmat[3*i:3*i+3,3*j:3*j+3] = R
                distmat[3*j:3*j+3,3*i:3*i+3] = R
        return distmat

    @staticmethod
    def _atoms2AtomicNum(atoms: Tuple[str, ...]) -> np.ndarray:
        """
        Convert an array of element names to an array of atomic numbers.
        Input: atoms (tuple, dimension (N), element names)
        Output: atomic_num (np.array, dimension (N), atomic numbers)
        """
        N = len(atoms)
        atomic_num = np.zeros(N, dtype=int)
        for i in range(N):
            try:
                atomic_num[i] = periodic_table.index(atoms[i].capitalize())
            except (IndexError, ValueError):
                raise ValueError("Unsupported element: %s"%atoms[i])
        return atomic_num

    @staticmethod
    def _readXYZ(
        path: Union[str, Path],
        encoding: str = "utf-8",
        unit: str = "angstrom",
    ) -> Tuple[Path, np.ndarray, Tuple[str, ...]]:
        """
        Read atomic coordinates from an XYZ file into a numpy array. `~` will be treated as user home.

        Return filepath, coordinates, atoms. Coordinates are in Bohr!

        There must be **only one** molecule in the file!

        XYZ format:
            <number of atoms>
            <comment (can be empty)>
            <symbol> <x> <y> <z>
            <symbol> <x> <y> <z>
            ...
        """
        # unit
        if unit.casefold() not in ["angstrom".casefold(), "bohr".casefold()]:
            raise ValueError(f"unit must be 'angstrom' or 'bohr' (case insensitive), '{unit}' is given")
        isBohr = False
        if unit.casefold() == "bohr".casefold():
            isBohr = True

        # handle path
        path = getAbsPath(path) # make path absolute
        if not path.is_file():
            raise FileNotFoundError(f"input XYZ file {path} not exists or not a file")
        # read file, remove empty lines and strip whitespace, keep first 2 lines (second line may be empty)
        lines = [line.strip() for i, line in enumerate(path.read_text(encoding).splitlines()) if i < 2 or line.strip()]

        # parse number of atoms from the first line
        try:
            n_atoms = int(lines[0])
        except ValueError:
            raise ValueError("First line of XYZ file must contain the number of atoms")
        # Check if the file contains **only one** molecule
        assert len(lines) == n_atoms + 2, "lines not match, XYZ file is incorrect or contains more than one molecule"

        # Extract coordinates (skip the first two lines)
        try:
            coordinates: np.ndarray = np.array([[float(s) for s in line.split()[1:]] for line in lines[2:]])
            atoms = tuple(line.split()[0] for line in lines[2:]) # use tuple to make sure it is not editable
        except (IndexError, ValueError):
            raise ValueError("Could not parse XYZ file, the file may be incorrect")
        assert coordinates.shape == (n_atoms, 3), f"the coordinates shape {coordinates.shape} is incorrect"

        if not isBohr:
            coordinates = coordinates * angstrom2bohr

        return path, coordinates, atoms

    @staticmethod
    def _writeXYZ(xyz_bohr: np.ndarray, atoms: Sequence[str], path: Path, useBohr: bool=False, comment: str = "", encoding: str = "utf-8"):
        assert xyz_bohr.shape == (xyz_bohr.size // 3, 3) and xyz_bohr.shape[0] == len(atoms)
        xyz_out = xyz_bohr if useBohr else xyz_bohr * bohr2angstrom
        xyz_str = f"{len(atoms)}\n" + \
            comment.strip() + "\n" + \
            "\n".join([
                f"{atom:<3}{x:>26.13f}{y:>26.13f}{z:>26.13f}"
                for atom, (x, y, z) in zip(atoms, xyz_out)] # type: ignore
            ) + \
            "\n"
        path.write_text(xyz_str, encoding)

    @staticmethod
    def _readEgrad1(egrad1_path: Union[str, Path]) -> Tuple[float, np.ndarray]:
        """
        read energy and gradient from BDF output .egrad1 file.

        TODO Note: BDF output units
        """
        egrad1_path = getAbsPath(egrad1_path)
        if not egrad1_path.exists():
            raise FileNotFoundError(f"BDF output .egrad1 file: {egrad1_path} not found, error may occurred during calculating")
        lines = [line.strip() for line in egrad1_path.read_text().splitlines() if line.strip()]

        try:
            energy = float(lines[0].split()[-1])
            grad: np.ndarray = np.array([[float(s) for s in line.split()[1:]] for line in lines[2:]])
            n_atoms = len(lines)-2
            assert grad.shape == (n_atoms, 3)
            return energy, grad
        except (AssertionError, IndexError, ValueError):
            raise ValueError(f"Could not parse BDF output .egrad1 file: {egrad1_path}")

    @staticmethod
    def _readEngrad(engrad_path: Union[str, Path]) ->  Tuple[float, np.ndarray]:
        """TODO output dimension is 1"""
        engrad_path = getAbsPath(engrad_path)
        if not engrad_path.exists():
            raise FileNotFoundError(f"ORCA output .engrad file: {engrad_path} not found, error may occurred during calculating")
        lines = engrad_path.read_text().splitlines()

        try:
            is_eng = False
            for line in lines:
                if is_eng and "#" not in line:
                    eng = float(line)
                    break
                if "energy".casefold() in line.casefold():
                    is_eng = True

            is_grad = False
            _grad = []
            for line in lines:
                if is_grad and "the".casefold() in line.casefold():
                    is_grad = False
                if is_grad and "#" not in line:
                    _grad.append(float(line))
                if "gradient".casefold() in line.casefold():
                    is_grad = True
            grad: np.ndarray = np.array(_grad)
            # print(grad)
            return eng, grad # type: ignore
        except NameError:
            raise ValueError(f"Could not parse ORCA output .engrad file: {engrad_path}")

    def _O1NH(
        self,
        grad_func: (...),
        method: str,
        delta: float,
        core: int,
        total_cores: Union[int, None],
        task_name: str = "",
        if_exists: str = "overwrite",
        dmax: float = 1.0,
        thresh_imag: float = 1e-8,
        has_g0: bool = False,
        transinvar: bool = True,
        rotinvar: bool = True,
        verbosity: int = 0,
        **kwargs_for_grad_func,
    ) -> np.ndarray:
        """
        interface with O1NH
        """
        # ========== initialize O1NH
        kwargs_for_grad_func["task_name"] = task_name # both O1NH and grad_func need task_name
        o1nh = O1NumHess(
            x=self.xyz_bohr.reshape((self.xyz_bohr.size,)),
            grad_func=grad_func,
            **kwargs_for_grad_func
        )
        o1nh.setVerbosity(verbosity)
        # ========== use o1nh to calculate hessian
        # use method as default task_name if not specified
        checkpoint_task_name = task_name if task_name else method
        if method.casefold() == "single".casefold():
            self.hessian = o1nh.singleSide(delta=delta, core=core, total_cores=total_cores, task_name=checkpoint_task_name, if_exists=if_exists)
        elif method.casefold() == "double".casefold():
            self.hessian = o1nh.doubleSide(delta=delta, core=core, total_cores=total_cores, task_name=checkpoint_task_name, if_exists=if_exists)
        elif method.casefold() == "o1numhess".casefold():
            self.thresh_imag = thresh_imag
            self.hessian = self.runO1NumHess(delta=delta, core=core, total_cores=total_cores,\
                o1nh=o1nh, config="BDF", dmax=dmax, has_g0=has_g0, transinvar=transinvar, rotinvar=rotinvar)
        else:
            raise ValueError(f"method {method} is not supported, only supported 'single', 'double' and 'o1numhess'")
        return self.hessian

    def runO1NumHess(self,
                     delta: float,
                     core: int,
                     total_cores: Union[int, None],
                     o1nh: O1NumHess,
                     config: str = "BDF",
                     dmax: float = 1.0,
                     has_g0: bool = False,
                     transinvar: bool = False,
                     rotinvar: bool = False,
                     ) -> np.ndarray:
        """
        Prepare all the pre-requisites of o1nh.O1NumHess, and call it.
        """
        from .Swart import Swart

        # We do not permit utilizing rotational invariance without utilizing translational
        # invariance
        if rotinvar and not transinvar:
            raise ValueError('rotinvar==True and transinvar==False is not permitted')

        N = self.xyz_bohr.shape[0]
        # Special case: for a single atom, and if translational invariance is present,
        # return the zero Hessian
        if N == 1:
            if transinvar:
                if self.verbosity > 0:
                    print("Warning: there is only one atom. The Hessian is the zero matrix.")
                self.hessian: np.ndarray = np.zeros(3)
            else:
                # If the user does not assume translational invariance, this may mean there
                # is an external electric field etc. so that translational invariance is not
                # guaranteed.
                # do a double-sided differentiation
                self.hessian = o1nh.doubleSide(core=core, delta=delta, total_cores=total_cores)
            return self.hessian

        # effective distance matrix. Note that this is not simply the matrix of Cartesian
        # distances, but is rather Cartesian distances minus sum of vdW radii
        if self.verbosity > 1:
            print('Evaluate effective distance matrix...')
            tstart = time.time()

        distmat = self._effDistMat

        if self.verbosity > 1:
            tend = time.time()
            print('Effective distance matrix done, total time: %.2f sec'%(tend-tstart))

        # modified Swart model Hessian
        if self.verbosity > 1:
            print('Evaluate Swart Hessian...')
            tstart = time.time()

        H0 = Swart(self.xyz_bohr, self.atomic_num)

        if self.verbosity > 1:
            tend = time.time()
            print('Swart Hessian done, total time: %.2f sec'%(tend-tstart))

        # the following displacement directions should be included regardless of the molecule:
        # (1) translations and rotations
        # (2) the symmetric breathing mode
        displdir = np.zeros([3*N,7])
        # prepare the first Ntr displacement directions
        # Ntr = 5 (for linear molecules) or 6 (for nonlinear molecules)
        displdir[:,0:6], Ntr = vecTransRot(self.xyz_bohr)
        if not rotinvar:
            Ntr = 3
        if not transinvar:
            Ntr = 0
        # prepare the symmetric breathing mode
        displdir[:,Ntr] = symmetricBreathing(self.xyz_bohr)
        displdir = displdir[:,0:(Ntr+1)]

        # Get the gradient of the unperturbed geometry, if there is one
        if has_g0:
            if self.verbosity > 1:
                print('Gradient at equilibrium geometry will be read from disk')
            if config == "BDF":
                inp = getAbsPath(o1nh.kwargs["inp"])
                task_name = inp.stem
                egrad1_in_path = getAbsPath(f"{task_name}.egrad1")
                _, g0 = self._readEgrad1(egrad1_in_path)
                g0 = g0.reshape((self.xyz_bohr.size,))
            else:
                raise Exception('Unsupported config: %s'%config)
        else:
            if self.verbosity > 1:
                print('Evaluate gradient at equilibrium geometry...')
                tstart = time.time()
            # do a gradient calculation at the unperturbed geometry
            if total_cores == None:
                total_cores0 = os.cpu_count()
                if not isinstance(total_cores0, int):
                    total_cores0 = core
            else:
                total_cores0 = total_cores
            # Use 6*N as the index, to avoid clashing with the calculations of
            # displaced geometries
            g0 = o1nh.grad_func(self.xyz_bohr,6*N,total_cores0,**o1nh.kwargs)
            if self.verbosity > 1:
                tend = time.time()
                print('Gradient done, total time: %.2f sec'%(tend-tstart))
                if self.verbosity > 5:
                    print('Gradient at the equilibrium geometry:')
                    print(g0)

        # The gradients along the translational and rotational directions
        # Note: g are not the gradients themselves, but are
        # ([gradients at displaced geometries] - g0)/delta, in the delta->0 limit
        # Translations: zero
        g = np.zeros([3*N,Ntr])
        # The gradients along the rotational directions are not zero when the geometry
        # is not an equilibrium geometry. We now account for this fact
        if rotinvar:
            g[:,3:Ntr] = rotationGradient(self.xyz_bohr,g0,Ntr-3)
        if self.verbosity > 5:
            print('Gradient derivatives along the trans/rot directions:')
            print(g)

        # The only double-sided differentiation is along the symmetric breathing mode
        doublesided = np.zeros(3*N, dtype = bool)
        doublesided[:] = False
        doublesided[Ntr] = True

        # finally, the actual calculation
        # (1) Initial Hessian
        if self.verbosity > 1:
            print("Stage 1: Initial estimation of the Hessian")
            tstart = time.time()

        self.hessian, displdir, gout = o1nh.O1NumHess(core=core, delta=delta,\
            total_cores=total_cores, dmax=dmax, distmat=distmat, \
            H0=H0, displdir=displdir, g=g, g0=g0, doublesided=doublesided)

        if self.verbosity > 1:
            tend = time.time()
            print('Stage 1 done, total time: %.2f sec'%(tend-tstart))

        # (2) Check if there are imaginary modes
        if self.verbosity > 1:
            print("Stage 2: Check imaginary modes")
            tstart = time.time()

        eigval, eigvec = np.linalg.eig(self.hessian)

        # append all imaginary modes (if there are any) to displdir
        Nimag = np.sum(eigval<-self.thresh_imag)
        if self.verbosity > 1:
            print(" - %d imaginary mode(s) found"%Nimag)
            if self.verbosity > 5:
                print("Negative eigenvalues of the Hessian:")
                print(eigval[eigval<-self.thresh_imag])
            tend = time.time()
            print('Stage 2 done, total time: %.2f sec'%(tend-tstart))
        if Nimag>3*N-displdir.shape[1]:
            # There are more imaginary frequencies than the number of remaining displacements.
            # In this case we only displace along the modes with the most negative eigenvalues.
            Nimag = 3*N-displdir.shape[1]

        # (3) Run O1NumHess again, with the imaginary modes added to the list of displacements
        if Nimag>0:
            if self.verbosity > 1:
                print("Stage 3: Displace along the imaginary modes")
            # We use the property that eigval is sorted in ascending order.
            displdir = np.hstack((displdir, eigvec[:,0:Nimag]))
            self.hessian, _, __ = o1nh.O1NumHess(core=core, delta=delta, \
                total_cores=total_cores, dmax=dmax, distmat=distmat, \
                H0=H0, displdir=displdir, g=gout, g0=g0, doublesided=doublesided)
            if self.verbosity > 1:
                tend = time.time()
                print('Stage 3 done, total time: %.2f sec'%(tend-tstart))
        else:
            if self.verbosity > 1:
                print("Skip stage 3 as there are no further displacements to be made")

        if self.verbosity > 1:
            print("Exit runO1NumHess")
        return self.hessian

    def calcHessian_BDF(
        self,
        method: str,
        delta: float,
        core: int,
        mem: str,
        total_cores: Union[int, None] = None,
        inp: Union[str, Path] = ...,
        encoding: str = "utf-8",
        tempdir: Union[Path, str] = "~/tmp",
        task_name: str = "",
        if_exists: str = "overwrite",
        config_name: str = "",
        dmax: float = 1.0,
        thresh_imag: float = 1e-8,
        has_g0: bool = False,
        transinvar: bool = True,
        rotinvar: bool = True,
    ) -> np.ndarray:
        """
        TODO Note: Units are read directly from inp file, no need to pass in
        """
        if self.verbosity > 2:
            print("Start calculating numerical Hessian (BDF)...")
            print("Parameters:")
            print(" - Method: %s"%method)
            print(" - Step length: %e Bohr"%delta)
            print(" - Number of cores used in the calculation: %d"%core)
            print(" - Maximum memory per core: %s"%mem)
            print("")
            tstart = time.time()

        # ========== check params
        _ = getConfig("BDF", config_name)

        inp = getAbsPath(inp)
        if not inp.is_file():
            raise FileNotFoundError(f"input .inp file: {inp} not exists or not a file")
        tempdir = getAbsPath(tempdir)
        if not tempdir.exists():
            os.makedirs(tempdir, exist_ok=True)

        # ========== interface with O1NH
        hessian = self._O1NH(
            grad_func=self._calcGrad_BDF,
            method=method,
            delta=delta,
            core=core,
            total_cores=total_cores,
            task_name=task_name,
            if_exists=if_exists,
            dmax=dmax,
            thresh_imag=thresh_imag,
            has_g0=has_g0,
            transinvar=transinvar,
            rotinvar=rotinvar,
            verbosity=self.verbosity,
            **{
                "mem": mem,
                "inp": inp,
                "encoding": encoding,
                "tempdir": tempdir,
                "config_name": config_name,
            }
        )

        if self.verbosity > 1:
            tend = time.time()
            print('BDF numerical Hessian done, total time: %.2f sec'%(tend-tstart))
            print('calcHessian_BDF terminated successfully')

        return hessian

    def _calcGrad_BDF(
        self,
        x_bohr: np.ndarray,
        index: int,
        core: int,
        mem: str, # "1G"
        inp: Union[str, Path], # TODO Note: inp file must contain "file=xxx.xyz" in "Geometry" block, otherwise error
        encoding: str = "utf-8",
        tempdir: Union[str, Path] = "~/tmp",
        task_name: str = "", # TODO Note: BDF input filename (because BDF requires both xyz and inp files, but the two filenames may be inconsistent), if not set, use inp filename as task name
        config_name: str = "", # TODO Note: BDF config filename, if not specified, use the first config file, if no config file exists, raise error
    ) -> np.ndarray:
        """
        Call BDF to calculate gradient once and read the result

        Given perturbed x (in Bohr units), generate new coordinate XYZ file
        Given BDF calculation parameter file inp (not required to be in current working path), read calculation units and parameters from it, generate new config file
        Generate .sh file to call BDF, execute it, and read .egrad1 file to extract gradient after completion

        output to specified folder (not supported by BDF now) - Due to BDF limitations, output path can only be in current folder
        Output gradient units are always in Bohr, output shape is 1D vector
        """
        if self.verbosity > 4:
            print("Start calculating gradient %d"%index, flush=True)
            tstart = time.time()

        # ========== check params
        config = getConfig("BDF", config_name)
        assert 0 < core and isinstance(core, int) # <= os.cpu_count()
        inp =getAbsPath(inp)
        if not inp.is_file():
            raise FileNotFoundError(f"input .inp file: {inp} not exists or not a file")
        tempdir = getAbsPath(tempdir)
        if not tempdir.exists():
            os.makedirs(tempdir, exist_ok=True)

        task_name = task_name if task_name else inp.stem

        # ========== make sure the input x is valid
        x_bohr = np.array(x_bohr)
        assert x_bohr.size == self.xyz_bohr.size, f"the input size of x is {x_bohr.size}, different with the initial molecular size {self.xyz_bohr.size}"

        # ========== generate filename for BDF files
        # print(Path("."))
        suffix = str(index).zfill(len(str(x_bohr.size * 2))) # use index and x.size to generate a suffix with proper length
        task_name = f"{task_name}_{suffix}"     # name for current task instance
        tempdir = tempdir / task_name           # make tempdir for current task instance
        xyz_out_path = getAbsPath(f"{task_name}.xyz")
        inp_out_path = getAbsPath(f"{task_name}.inp")
        sh_out_path = getAbsPath(f"{task_name}.sh")
        egrad1_in_path = getAbsPath(f"{task_name}.egrad1")

        # ========== generate new .inp file for BDF
        # read inp file, drop comments and right spaces
        inp_str = inp.read_text(encoding).splitlines()
        inp_str = [line.rstrip() if "#" not in line else line.split("#")[0].rstrip() for line in inp_str if "#" not in line or line.split("#")[0].strip()]
        # print("\n".join(inp_str))

        # find "unit"
        useBohr = False
        for i in range(len(inp_str)-1):
            if (inp_str[i].strip().casefold() == "Unit".casefold() and \
                inp_str[i+1].strip().casefold() == "Bohr".casefold()
            ) or inp_str[i].casefold().find("unit=bohr".casefold()) > -1:
                useBohr = True

        # find the .xyz file line
        file_line = []
        for i in range(1, len(inp_str)-1):
            if (
                inp_str[i-1].strip().casefold() == "Geometry".casefold() and \
                inp_str[i].strip().casefold().startswith("file=".casefold()) and \
                inp_str[i+1].strip().casefold() == "End geometry".casefold() and \
                inp_str[i].strip()[5:] == self.xyz_path.name
                # the filename must be equal with the xyz file
            ):
                file_line.append(i)
        if file_line == []:
            raise ValueError(f"the inp file {inp} does not contain proper Geometry info, it must be written as 'file={self.xyz_path.name}' in 'Geometry' part, no spaces and slash '/'")

        # replace the .xyz file with the new one
        for i in file_line:
            inp_str[i] = inp_str[i].replace(self.xyz_path.name, xyz_out_path.name)
        # print("\n".join(inp_str))

        # output file
        inp_out_path.write_text("\n".join(inp_str), encoding)

        # ========== generate new .xyz file for BDF
        self._writeXYZ(x_bohr.reshape(x_bohr.size // 3, 3), self.atoms, xyz_out_path, useBohr)

        # ========== generate new .sh file to run BDF
        sh_out_path.write_text(config["bash"] + \
            dedent(f"""
            export OMP_NUM_THREADS={core}
            export OMP_STACKSIZE={mem}

            rm -rf {tempdir}
            {config["path"]} -tmpdir {tempdir} -r {inp_out_path.name} > {task_name}.out
            rm -rf .{task_name}.wrk
            """
        ), "utf-8")

        # ========== calculate
        os.system(f"bash {sh_out_path}")

        # ========== read result
        energy, grad = self._readEgrad1(egrad1_in_path)
        assert grad.shape == self.xyz_bohr.shape, f"the grad shape from BDF output .egrad1 file: {egrad1_in_path} is {grad.shape}, different with the initial molecular shape {self.xyz_bohr.shape}"

        if self.verbosity > 1:
            print("Finished calculating gradient %d"%index, flush=True)
        # we do not recommend going to such high verbosity, except for serial runs
        # in parallel runs, the printout of different processes will mess up with each other
        if self.verbosity > 4:
            print("Energy: %.12f Hartree"%energy)
            print("Gradients in Hartree/Bohr:")
            n_atoms = len(self.atoms)
            for iatom in range(n_atoms):
                print("%5s %20.12f %20.12f %20.12f"%(self.atoms[iatom], grad[iatom,0], grad[iatom,1], grad[iatom,2]))
            tend = time.time()
            print("Total time: %.2f sec"%(tend-tstart)) # type: ignore
            print("")

        return grad.reshape((self.xyz_bohr.size,))

    def calcHessian_ORCA(
        self,
        method: str,
        delta: float,
        total_cores: Union[int, None] = None,
        inp: Union[str, Path] = ...,
        encoding: str = "utf-8",
        tempdir: Union[Path, str] = "~/tmp",
        task_name: str = "",
        if_exists: str = "overwrite",
        config_name: str = "",
    ) -> np.ndarray:
        """
        The core parameter for parallelization is specified in the inp file
        """
        # ========== check params
        _ = getConfig("ORCA", config_name)

        inp = getAbsPath(inp)
        if not inp.is_file():
            raise FileNotFoundError(f"input .inp file: {inp} not exists or not a file")
        tempdir = getAbsPath(tempdir)
        if not tempdir.exists():
            os.makedirs(tempdir, exist_ok=True)

        task_name = task_name if task_name else inp.stem

        inp_str = inp.read_text(encoding=encoding)
        # ========== get "core"
        match = re.search(r"^\s*!.*?PAL(\d+)|^\s*%\s*pal\s*nprocs\s*(\d+)", inp_str, re.MULTILINE | re.IGNORECASE)
        if match is None:
            raise ValueError(f"inp file {inp} does not contain parallel information like 'PAL' or '%pal nprocs'")
        core = int(match.group(1) if match.group(1) else match.group(2))

        # ========== make sure gradient is calculated in ORCA
        if re.search(r"^\s*!.*?EnGrad", inp_str, re.MULTILINE | re.IGNORECASE) is None:
            raise ValueError(f"inp file {inp} does not contain parameter 'EnGrad', cannot calculate gradient in ORCA")

        if self.verbosity > 1:
            print("Start calculating numerical Hessian (ORCA)...")
            print("Parameters:")
            print(" - Method: %s"%method)
            print(" - Step length: %e Bohr"%delta)
            print(" - Number of cores used in the calculation: %d"%core)
            # print(" - Maximum memory per core: %s"%mem) # TODO Read config file in advance and get information
            print("")
            tstart = time.time()

        # ========== interface with O1NH
        return self._O1NH(
            grad_func=self._calcGrad_ORCA,
            method=method,
            delta=delta,
            core=core,
            total_cores=total_cores,
            task_name=task_name,
            if_exists=if_exists,
            **{
                "inp": inp,
                "encoding": encoding,
                "tempdir": tempdir,
                "config_name": config_name,
            }
        )

    def _calcGrad_ORCA(
        self,
        x_bohr: np.ndarray,
        index: int,
        core: int,
        inp: Union[str, Path],
        encoding: str = "utf-8",
        tempdir: Union[str, Path] = "~/tmp",
        task_name: str = "",
        config_name: str = "",
    ) -> np.ndarray:
        # ========== check params
        config = getConfig("ORCA", config_name)
        assert 0 < core and isinstance(core, int)
        inp = getAbsPath(inp)
        if not inp.is_file():
            raise FileNotFoundError(f"input .inp file: {inp} not exists or not a file")
        tempdir = getAbsPath(tempdir)
        if not tempdir.exists():
            os.makedirs(tempdir, exist_ok=True)

        task_name = task_name if task_name else inp.stem

        # ========== make sure the input x is valid
        x_bohr = np.array(x_bohr)
        assert x_bohr.size == self.xyz_bohr.size, f"the input size of x is {x_bohr.size}, different with the initial molecular size {self.xyz_bohr.size}"

        # ========== generate filename and path for ORCA files
        # print(Path("."))
        cwd = getAbsPath(".")
        suffix = str(index).zfill(len(str(x_bohr.size * 2))) # use index and x.size to generate a suffix with proper length
        task_name = f"{task_name}_{suffix}"     # name for current task instance
        tempdir = tempdir / task_name           # make tempdir for current task instance
        xyz_out_path = getAbsPath(f"{task_name}.xyz")
        inp_out_path = getAbsPath(f"{task_name}.inp")
        sh_out_path = getAbsPath(f"{task_name}.sh")
        engrad_in_path = getAbsPath(f"{task_name}.engrad")

        # ========== generate new .inp file for ORCA
        inp_str = inp.read_text(encoding)

        # remove comments in inp file
        inp_str = re.sub(r"#.*?#", "", inp_str, flags=re.MULTILINE)
        inp_str = re.sub(r"#.*?$\n", "", inp_str, flags=re.MULTILINE)

        # make sure to calculate gradient in ORCA
        if re.search(r"^\s*!.*?EnGrad", inp_str, re.MULTILINE | re.IGNORECASE) is None:
            raise ValueError(f"inp file {inp} does not contain parameter 'EnGrad', cannot calculate gradient in ORCA")

        # find "pal" and replace with current core
        pattern1 = r"(^\s*!.*?PAL)(\d+)"
        pattern2 = r"(^\s*%\s*pal\s*nprocs\s*)(\d+)"
        if re.search(pattern1, inp_str, re.MULTILINE | re.IGNORECASE) is None and re.search(pattern2, inp_str, re.MULTILINE | re.IGNORECASE) is None:
            raise ValueError(f"inp file {inp} does not contain parallel information like 'PAL' or '%pal nprocs'")
        inp_str = re.sub(pattern1, rf"\g<1>{core}", inp_str, flags=re.MULTILINE | re.IGNORECASE) # replace PAL
        inp_str = re.sub(pattern2, rf"\g<1>{core}", inp_str, flags=re.MULTILINE | re.IGNORECASE) # replace nprocs
        # print(inp_str)

        # find "unit", line like: ! xxxxxx Bohrs
        useBohr = True if re.search(r"^\s*!.*?Bohrs", inp_str, re.MULTILINE | re.IGNORECASE) else False

        # find and replace the .xyz file line
        pattern = r"(^\s*\*\s*xyzfile\s*\d+\s*\d+\s*)(.+\.xyz)"
        if re.search(pattern, inp_str, re.MULTILINE | re.IGNORECASE) is None:
            raise ValueError(f"inp file {inp} does not contain molecular coordinate information, coordinates must be specified by xyzfile format")
        inp_str = re.sub(pattern, rf"\g<1>{xyz_out_path.name}", inp_str, flags=re.MULTILINE | re.IGNORECASE)
        # print(inp_str)

        # output file
        inp_out_path.write_text(inp_str, encoding=encoding)

        # ========== deal with .gbw files
        gbw_str_list = []
        home_dir = inp.parent # find gwb files at the folder of inp file
        # 1. file with the same name of inp file will be copy as task_name.gbw
        gbw_in_path = getAbsPath(home_dir / f"{inp.stem}.gbw")
        gbw_out_path = getAbsPath(tempdir / f"{task_name}.gbw")
        if gbw_in_path.is_file():
            gbw_str_list.append(f"cp {gbw_in_path} {gbw_out_path}")

        # 2. %moinp "file.xxx" and "xxx.gbw" will be copied
        files2copy = set()
        files2copy.update(re.findall(r'"(.*?\.gbw)"', inp_str, flags=re.MULTILINE | re.IGNORECASE))
        files2copy.update(re.findall(r'^\s*%moinp\s*"(.*?)"', inp_str, flags=re.MULTILINE | re.IGNORECASE))
        # print(files2copy)
        for file in files2copy:
            path = getAbsPath(home_dir / file)
            if not path.is_file():
                raise FileNotFoundError(f"gbw file: {path} not exists.")
            gbw_str_list.append(f"cp {path} {tempdir}")

        gbw_str = "\n".join(gbw_str_list)

        # ========== generate new .xyz file for ORCA
        self._writeXYZ(x_bohr.reshape(x_bohr.size // 3, 3), self.atoms, xyz_out_path, useBohr)

        # print(f"{tempdir}")
        # ========== generate new .sh file to run ORCA
        sh_out_path.write_text(config["bash"] + \
            dedent(f"""
            rm -rf {tempdir}
            mkdir {tempdir}
            cp {inp_out_path} {tempdir}
            cp {xyz_out_path} {tempdir}
            {{gbw_str}}

            cd {tempdir}
            {config["path"]} {inp_out_path.name} >& {task_name}.out

            rm -f *.inp *.xyz *.tmp*
            cp * {cwd}

            rm -rf {tempdir}
            """
        ).format(gbw_str=gbw_str), "utf-8")
        # cd to tempdir, calculate, del non-result file, copy the rest result file back

        # ========== calculate
        os.system(f"bash {sh_out_path}")

        # ========== read result
        energy, grad = self._readEngrad(engrad_in_path)
        assert grad.size == self.xyz_bohr.size, f"the grad size from ORCA output .engrad file: {engrad_in_path} is {grad.size}, different with the initial molecular shape {self.xyz_bohr.size}"
        if self.verbosity > 1:
            print("Finished calculating gradient %d"%index, flush=True)
        # we do not recommend going to such high verbosity, except for serial runs
        # in parallel runs, the printout of different processes will mess up with each other
        if self.verbosity > 4:
            print("Energy: %.12f Hartree"%energy)
            print("Gradients in Hartree/Bohr:")
            n_atoms = len(self.atoms)
            for iatom in range(n_atoms):
                print("%5s %20.12f %20.12f %20.12f"%(self.atoms[iatom], grad[iatom,0], grad[iatom,1], grad[iatom,2]))
            tend = time.time()
            print("Total time: %.2f sec"%(tend-tstart)) # type: ignore
            print("")
        return grad.reshape((self.xyz_bohr.size,))

    # def toHessian(self):
