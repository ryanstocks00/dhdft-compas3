import sys
from pathlib import Path
import json

inputs_path = Path(__file__).parent.parent
sys.path.insert(0, str(inputs_path))
import common

BATCH_SIZE = 20

graphene_isomers = common.get_all_graphene_isomers()
selected_isomers = list(
    filter(lambda x: x.carbons == 24 and x.hydrogens == 14, graphene_isomers)
)

input_path = Path(__file__).parent / "exess_inputs"
input_path.mkdir(parents=True, exist_ok=True)

for i in range(0, len(selected_isomers), BATCH_SIZE):
    batch = selected_isomers[i : i + BATCH_SIZE]
    print(f"Batch {i//BATCH_SIZE + 1}: {len(batch)} isomers")
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
            "max_gpu_memory_mb": 50000,
        },
        "keywords": {
            "scf": {
                "convergence_threshold": 1.0e-10,
                "density_threshold": 1.0e-10,
                "use_ri": True,
            },
            "log": {"console": {"level": "Verbose"}},
            "ks_dft": {
                "functional": "revDSD-PBEP86-D4(noFC)",
                "method": "BatchDense",
                "grid": {
                    "default_grid": "ULTRAFINE",
                    "octree": {"max_size": 2048},
                },
                "batches_per_batch": 50,
            },
        },
        "topologies": [],
    }
    for xyz in batch:
        print(xyz)
        json_to_write["topologies"].append({"xyz": str(xyz.xyz_path.resolve())})
    with open(input_path / f"isomers_{i}-{i + len(batch) - 1}.json", "w") as f:
        json.dump(json_to_write, f, indent=4)
