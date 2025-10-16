import subprocess, re, sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())
import common

MAX_CONCURRENT = 20
job_ids, jobid_pat = [], re.compile(r"(\d[\w\.\-]*)")

for i, calc in enumerate(common.orca_calculations):
    inp, out = calc.input_filepath(), calc.output_filepath()
    dep = f"-W depend=afterany:{job_ids[i - MAX_CONCURRENT]}" if i >= MAX_CONCURRENT else ""
    cmd = [
                    "qsub", "-v",
                            f"OUTPUT_FOLDER={out.parent.resolve()},INPUT_FOLDER={inp.parent.resolve()},"
                                    f"INPUT_FILE={inp.stem},OUTPUT_FILE={out.name}",
                                        ]
    if dep: 
        cmd += dep.split()
    cmd.append("orca_pbs.sh")
    print(" ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    m = jobid_pat.search(res.stdout)
    job_ids.append(m.group(1) if m else "unknown")
