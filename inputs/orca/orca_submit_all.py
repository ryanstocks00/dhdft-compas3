import subprocess, re, sys
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.as_posix())
import common

MAX_CONCURRENT = 30
job_ids, jobid_pat = [], re.compile(r"(\d[\w\.\-]*)")

idx = 0
for i, calc in enumerate(common.orca_calculations):

    if calc.basis_id not in ["qz_nori", "qz_jkjk"]:
        continue
    if calc.basis_id in ["qz_nori"] and calc.isomer.id != 1:
        continue

    inp, out = calc.input_filepath(), calc.output_filepath()
    dep = (
        f"-W depend=afterany:{job_ids[idx - MAX_CONCURRENT]}"
        if idx >= MAX_CONCURRENT
        else ""
    )
    idx += 1
    cmd = [
        "qsub",
        "-v",
        f"OUTPUT_FOLDER={out.parent.resolve()},INPUT_FOLDER={inp.parent.resolve()},"
        f"INPUT_FILE={inp.stem},OUTPUT_FILE={out.name}",
    ]
    if dep:
        cmd += dep.split()

    if calc.basis_id == "qz_nori":
        cmd.append("orca_pbs_superlong.sh")
    elif calc.isomer.carbons > 24 or (
        calc.scf_aux_basis is None and calc.ri_aux_basis is None
    ):
        cmd.append("orca_pbs_long.sh")
    else:
        cmd.append("orca_pbs.sh")
    print(" ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    m = jobid_pat.search(res.stdout)
    job_ids.append(m.group(1) if m else "unknown")
