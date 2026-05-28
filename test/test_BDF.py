import os
import shutil
from textwrap import dedent
from typing import Union
import unittest
import numpy as np
from pathlib import Path
from O1NumHess_QC import O1NumHess_QC
from O1NumHess_QC.constants import bohr2angstrom, angstrom2bohr


class TestBDF(unittest.TestCase):
    def setUp(self):
        print("\n" + "="*50)
        print(f"Starting test: {self._testMethodName}")
        print("="*50)

    def tearDown(self):
        print("-"*50)
        print(f"Finished test: {self._testMethodName}")
        print("-"*50 + "\n")

    def test_unitConvert(self):
        self.assertAlmostEqual(bohr2angstrom * angstrom2bohr, 1)

    def test_readAndWriteXYZ(self):
        path = Path("_testR&W.xyz")
        xyz_bohr = np.random.random((12, 3))
        atoms = ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"]
        # write in Bohr, read in Angstrom and been converted to bohr, manually convert unit
        O1NumHess_QC._writeXYZ(xyz_bohr, atoms, path, useBohr=True)
        _, xyz_bohr_, _atoms = O1NumHess_QC._readXYZ(path, unit="angstrom")
        np.testing.assert_array_almost_equal(xyz_bohr_ * bohr2angstrom, xyz_bohr)
        self.assertListEqual(atoms, list(_atoms))
        os.remove(path)

    def test_total_cores(self):
        xyz = self._generateXYZ("_test_total_cores.xyz")
        inp = self._generateInp("_test_total_cores.inp", xyz)
        qc = O1NumHess_QC(xyz)

        with self.assertRaises(ValueError):
            qc.calcHessian_BDF("single", 1e-3, core=2, mem="2G", total_cores=999, inp=inp)
        with self.assertRaises(ValueError):
            qc.calcHessian_BDF("single", 1e-3, core=2, mem="2G", total_cores=-1, inp=inp)
        # with self.assertWarns(RuntimeWarning):
        #     qc.calcHessian_BDF("single", 1e-3, core=2, mem="2G", total_cores=None, inp=inp)
        with self.assertRaises(TypeError):
            qc.calcHessian_BDF("single", 1e-3, core=2, mem="2G", total_cores=1.2, inp=inp) # type: ignore
        os.remove(xyz)
        os.remove(inp)

    def test_BDF(self):
        cur_dir = Path(".").absolute()
        test_dir = Path("./_test_BDF").absolute()
        os.makedirs(test_dir, exist_ok=True)

        test_xyz = self._generateXYZ("test.xyz")
        test_inp = self._generateInp("test.inp", test_xyz)

        try:
            os.chdir(test_dir)
            qc = O1NumHess_QC(test_xyz)
            eng_ = -232.108204353561
            grad_: np.ndarray = np.array([
                [ 0.0042475274,  0.0060127667, -0.0000039839],
                [-0.0031002205,  0.0066626468, -0.0000030526],
                [-0.0073204325,  0.0006716961,  0.0000008478],
                [-0.0042366843, -0.0060243853, -0.0000048821],
                [ 0.0030982019, -0.0066557785,  0.0000061277],
                [ 0.0072992591, -0.0006745076, -0.0000023993],
                [-0.0042945127, -0.0061028118, -0.0000007873],
                [-0.0074275957,  0.0006603016,  0.0000007442],
                [-0.0031308833,  0.0067766069,  0.0000040330],
                [ 0.0043069792,  0.0060984600, -0.0000008635],
                [ 0.0031303714, -0.0067593045,  0.0000035547],
                [ 0.0074279899, -0.0006656904,  0.0000006613]
            ])

            index = 0
            grad = qc._calcGrad_BDF(qc.xyz_bohr, index, 4, "4G", test_inp)
            eng, _ = qc._readEgrad1(test_dir / f"test_{index:02}.egrad1")
            self.assertAlmostEqual(eng_, eng)
            np.testing.assert_array_almost_equal(grad_.reshape((grad_.size,)), grad)
        finally:
            os.chdir(cur_dir)
            os.remove(test_xyz)
            os.remove(test_inp)
            shutil.rmtree(test_dir)

    def _generateInp(self, path: Union[str, Path], xyz:Path):
        test_inp = Path(path).absolute()
        test_inp.write_text(dedent(f"""
            $COMPASS
            Title
            benzene grad
            Basis
            cc-pvdz
            Geometry
            file={xyz.name}
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
            """).strip()
        )
        return test_inp

    def test_BDF_Hessian(self):
        """run the follow to test this function only:

        python3 -m unittest test.test_BDF.TestBDF.test_BDF_Hessian -v
        """
        cur_dir = Path(".").absolute()
        test_dir = Path("./test_BDF").absolute()
        os.makedirs(test_dir, exist_ok=True)

        test_xyz = Path("test.xyz").absolute()
        test_xyz.write_text(dedent("""
            30

            N          0.17587861      2.64192585     -0.74008024
            N         -1.70113520      1.95307406      0.29442613
            N         -1.52213246      0.06730898      2.67581683
            N          0.78234041      0.16500941      2.35762192
            N          1.83755534      1.45928805      0.60589215
            N          1.17203815     -0.41281381     -2.11223940
            N         -1.09911487     -0.40330148     -1.80637088
            H         -1.34229828     -0.72439480      3.27043067
            H         -2.39529705      0.05626338      2.17303445
            H          2.80261119      0.31392716      2.01149064
            H          0.75758469      3.06489127     -1.44477901
            H          2.32245487     -1.63199297     -0.88120634
            H          0.07234767     -2.34052168      1.66430046
            H         -0.26359205     -3.64195683      0.52962171
            H          1.40782189     -3.12219205      0.81703794
            H         -1.76286378      3.25811629     -1.39041926
            H         -2.00482926     -0.01508636     -2.03290503
            H          1.98190680     -0.04093196     -2.57978698
            C          0.25417482     -1.78886486     -0.38738589
            C          0.37870648     -2.78286464      0.71779411
            C         -0.59314857      1.40652191      0.89586304
            C          0.59179597      1.81307204      0.27207023
            C          1.83181266      0.64237368      1.66247459
            C         -1.07189437     -1.29472597     -0.72830772
            C         -0.44955962      0.52655683      1.98290680
            C          1.30963092     -1.31859980     -1.08939537
            C         -0.03495267      0.11590949     -2.51070496
            C         -1.18828026      2.68083282     -0.68548930
            O         -0.14321041      0.95778657     -3.38743764
            O         -2.10635163     -1.60461060     -0.15427362
            """).lstrip()
        )
        test_inp = Path("test.inp").absolute()
        test_inp.write_text(dedent(f"""
            $COMPASS
            Title
            H2O2 grad
            Basis
            sto-3g
            Geometry
            file={test_xyz.name}
            End geometry
            nosym
            norotate
            $END

            $xuanyuan
            $end

            $scf
            RHF
            $end

            $resp
            geom
            $end
            """).strip()
        )

        try:
            os.chdir(test_dir)
            qc = O1NumHess_QC(test_xyz)
            qc.setVerbosity(10)
            hessian = qc.calcHessian_BDF(
                method = "o1numhess",
                #method = "single",
                delta = 1e-3,
                total_cores = 10,
                core = 1,
                dmax = 1,
                mem = "4G",
                inp = test_inp,
                tempdir = "~/tmp",
                config_name = "BDF",
                task_name="test_BDF_Hessian",
                keep_log=True,
                # if_exists="ask",
            )
            print(hessian)
        finally:
            os.chdir(cur_dir)
            os.remove(test_xyz)
            os.remove(test_inp)
            shutil.rmtree(test_dir)

    def _generateXYZ(self, path: Union[str, Path]):
        test_xyz = Path(path).absolute()
        test_xyz.write_text(dedent("""
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
            """).lstrip()
        )

        return test_xyz

if __name__ == "__main__":
    unittest.main()
