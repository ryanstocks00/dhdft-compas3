import sys
from pathlib import Path
import json

inputs_path = Path(__file__).parent.parent
sys.path.insert(0, str(inputs_path))
import common

input_path = Path(__file__).parent / "exess_inputs"
input_path.mkdir(parents=True, exist_ok=True)

batch_set = "PAH335"

for batch in common.exess_pah335_batches:
    print(
        f"Batch {batch.initial_index}-{batch.final_index}: {len(batch.isomers)} isomers"
    )
    isomers = batch.isomers
    json_to_write = {
        "driver": "Energy",
        "model": {
            "method": "RestrictedKSDFT",
            "basis": "def2-QZVPP",
            "aux_basis": "def2-QZVPP-RIFIT",
            "force_cartesian_basis_sets": False,
            "standard_orientation": "None",
        },
        "system": {
            "max_gpu_memory_mb": 54000,
        },
        "keywords": {
            "scf": {
                "convergence_threshold": 1.0e-10,
                "density_threshold": 1.0e-10,
                "use_ri": True,
                "compress_ri_b": True,
            },
            "log": {"console": {"level": "Info"}},
            "ks_dft": {
                "functional": "revDSD-PBEP86-D4(noFC)",
                "method": "BatchDense",
                "use_C_opt": False,
                "grid": {
                    "default_grid": "ULTRAFINE",
                    "octree": {"max_size": 2048},
                },
                "batches_per_batch": 30,
            },
        },
        "topologies": [],
    }
    for xyz in isomers:
        print(xyz)
        json_to_write["topologies"].append({"xyz": str(xyz.xyz_path.resolve())})
    with open(batch.input_file_path(), "w") as f:
        json.dump(json_to_write, f, indent=4)
