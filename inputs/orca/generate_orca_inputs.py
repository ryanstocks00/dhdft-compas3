import sys
from pathlib import Path

inputs_path = Path(__file__).parent.parent
sys.path.insert(0, str(inputs_path))
import common

def rmdir(dir):
    if not dir.exists():
        return
    for item in dir.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    dir.rmdir()

input_path = Path(__file__).parent / "orca_inputs"
rmdir(input_path)

for xyz in common.get_all_graphene_isomers():
    if xyz.carbons == 24 and xyz.hydrogens == 14:
        print(f"Generating ORCA input for {xyz.name}")
        for basis_name,bases in common.basis_combos.items():
            input_path = Path(__file__).parent / "orca_inputs" / basis_name
            input_path.mkdir(parents=True, exist_ok=True)

            primary_basis = bases[0]
            scf_aux_basis = None
            ri_aux_basis = None
            if len(bases) > 2:
                scf_aux_basis = bases[1]
            if len(bases) > 1:
                ri_aux_basis = bases[-1]
            ri_text = ""
            if scf_aux_basis is not None:
                ri_text += f" RIJK"
            if scf_aux_basis is None and ri_aux_basis is None:
                ri_text += f" NORI"
            with open(input_path / f"{xyz.name}_{basis_name}.inp", "w") as f:
                f.write(
                    f"""! RKS revDSD-PBEP86-D4/2021 defgrid3 {ri_text} NOPOP NoFrozencore NoUseSym

%basis
  Basis "{primary_basis}"
  { '' if scf_aux_basis is None else f'AuxJK "{scf_aux_basis}"' }
  { '' if ri_aux_basis is None else f'AuxC "{ri_aux_basis}"'}
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
