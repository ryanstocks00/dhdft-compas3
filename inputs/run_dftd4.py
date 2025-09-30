from pathlib import Path
import common
import subprocess


output_dir = Path(__file__).parent.parent / "output" / "dftd4"
output_dir.mkdir(parents=True, exist_ok=True)

for xyz in common.get_all_graphene_isomers():
    print(f"Running dftd4 for {xyz}")
    out = subprocess.run(
        [
            "dftd4",
            str(xyz.xyz_path),
            "--param",
            "0.4612",
            "0",
            "0.44",
            "3.60",
            "--json",
            str(output_dir / f"{xyz.name}.json"),
        ],
        capture_output=True,
        text=True,
    )
