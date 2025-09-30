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

%pal
    nprocs 104
end

"""
            )
            f.write(f"* xyz 0 1\n")
            with open(xyz.xyz_path, "r") as xyz_file:
                lines = xyz_file.readlines()[2:]
                f.writelines(lines)
            f.write(f"*\n")
