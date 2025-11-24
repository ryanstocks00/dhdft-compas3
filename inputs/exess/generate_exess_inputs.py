    print(json_re0
import sys
from pathlib import Path
import json

inputs_path = Path(__file__).parent.parent
sys.path.insert(0, str(inputs_path))
import common

input_path = Path(__file__).parent / "exess_inputs"
input_path.mkdir(parents=True, exist_ok=True)

batch_set = "BENCHMARKING"

if batch_set == "PAH335":
    batches = common.exess_pah335_batches + common.exess_pah335_pbe_batches
elif batch_set == "COMPAS-3":
    batches = common.exess_batches
elif batch_set == "BOTH":
    batches = common.exess_pah335_batches + common.exess_batches + common.exess_pah335_pbe_batches
elif batch_set == "BENCHMARKING":
    # batches = common.exess_svwn_batches
    batches = common.exess_gga_batches
else:
    raise ValueError(f"Unknown batch set: {batch_set}")

for batch in batches:
    print(
        f"Batch {batch.initial_index}-{batch.final_index}: {len(batch.isomers)} isomers"
    )
    isomers = batch.isomers
    json_to_write = {
        "driver": "Energy",
        "model": {
            "method": "RestrictedKSDFT",
            "basis": batch.basis,
            "aux_basis": batch.aux_basis,
            "force_cartesian_basis_sets": False,
            "standard_orientation": "None",
        },
        "system": {
            "max_gpu_memory_mb": 40000,
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
                "functional": batch.functional_name,
                "method": "GauXC",
                "use_C_opt": True,
                "grid": {
                    "default_grid": "ULTRAFINE",
                    "pruning_scheme": "ROBUST",
                    "octree": {"max_size": 2048, "combine_small_children": True},
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
