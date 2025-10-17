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

for calc in common.orca_calculations:
    primary_basis = calc.primary_basis
    scf_aux_basis = calc.scf_aux_basis
    ri_aux_basis = calc.ri_aux_basis

    nori = False
    nprocs = 104
    ri_text = ""
    if scf_aux_basis is not None:
        ri_text += f" RIJK"
    if scf_aux_basis is None and ri_aux_basis is None:
        nori = True
        nprocs = 10
        ri_text += f" NORI"
    with open(input_path / calc.input_filepath(), "w") as f:
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
"""
            + """MaxCore 43860
Q1Opt 1
"""
            if nori
            else ""
            + """
end

%pal
nprocs {nprocs}
end

*XYZFILE 0 1 {calc.isomer.xyz_path.resolve()}
"""
        )
