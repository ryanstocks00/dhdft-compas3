import sys
from pathlib import Path

inputs_path = Path(__file__).parent.parent
sys.path.insert(0, str(inputs_path))
import common

for xyz in common.get_all_graphene_isomers():
    if xyz.carbons == 24 and xyz.hydrogens == 14:
        print(f"Generating ORCA input for {xyz.name}")
        input_path = Path(__file__).parent / "orca_inputs"
        input_path.mkdir(parents=True, exist_ok=True)
        with open(input_path / f"{xyz.name}.inp", "w") as f:
            f.write(
                f"""! RKS revDSD-PBEP86-D4/2021 defgrid3 RIJK RI NOPOP NOMULLIKEN NOLOEWDIN NOMAYER NoFrozencore NoUseSym

%basis
  Basis "def2-QZVPP"
  AuxJK "def2-QZVPP/C"
  AuxC "def2-QZVPP/C"
end

%method
  ScalHFX 0.69
  ScalDFX 0.31
  ScalDFC 0.4038
  ScalLDAC 0.4038

  D4S6 0.4612
  D4S8 0.0
  D4A1 0.44
  D4A2 3.60
end

%mp2
  PS 0.5979
  PT 0.0571
end

%pal
    nprocs 104
end

*XYZFILE 0 1 {xyz.xyz_path.resolve()}
"""
            )
