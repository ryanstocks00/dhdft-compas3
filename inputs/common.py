from pathlib import Path
import re


def get_all_xyzs(folder: Path) -> list[Path]:
    """Get all .xyz files in a folder."""
    assert folder.exists(), f"Folder {folder} does not exist."
    return list(sorted(folder.glob("*.xyz")))


class GrapheneIsomer:
    def __init__(self, xyz_path: Path, optimizer):
        prefix = "compas3x_" if optimizer == "xTB" else "compas3D_"
        self.xyz_path = xyz_path
        self.name = prefix + xyz_path.stem
        self.optimizer = optimizer

        m = re.match(r"compas3._hc_c(\d+)h(\d+)_0pent_(\d+)", self.name)
        if m:
            self.carbons = int(m.group(1))
            self.hydrogens = int(m.group(2))
            self.id = int(m.group(3))
        else:
            raise ValueError(f"Filename {self.name} does not match expected pattern.")

    def __repr__(self):
        return f"GrapheneIsomer(name={self.name}, carbons={self.carbons}, hydrogens={self.hydrogens}, id={self.id}, optimizer={self.optimizer})"


def get_all_graphene_isomers(optimizer: str) -> list[GrapheneIsomer]:
    """Get all GrapheneIsomer objects from .xyz files in a folder."""

    if optimizer == "xTB":
        folder = Path(__file__).parent / "compas3x-xyzs"
        tar_path = folder.parent / "compas-3x.tar.gz"
    elif optimizer == "DFT":
        folder = Path(__file__).parent / "compas3D-xyzs"
        tar_path = folder.parent / "compas-3d.tar.gz"
    else:
        raise ValueError(f"Optimizer {optimizer} not recognized. Use 'xTB' or 'DFT'.")

    if not folder.exists():
        # try unzipping compas-3x.tar.gz from the parent directory
        import tarfile
        import shutil
        import sys

        print(
            f"Folder {folder} does not exist. Attempting to extract from {tar_path}... (this may take a while)",
        )
        parent_folder = folder.parent
        if not tar_path.exists():
            print(f"Error: {tar_path} does not exist.", file=sys.stderr)
            sys.exit(1)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=parent_folder)
        if not folder.exists():
            print(
                f"Error: {folder} does not exist after extracting {tar_path}.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Successfully extracted {tar_path} to {parent_folder}.")

    xyz_files = get_all_xyzs(folder)
    graphenes = [GrapheneIsomer(x, optimizer) for x in xyz_files if x.name[0] != "."]
    graphenes.sort(key=lambda g: (g.carbons, g.hydrogens, g.id))
    return graphenes


xtb_graphene_isomers = get_all_graphene_isomers("xTB")
dft_graphene_isomers = get_all_graphene_isomers("DFT")

all_graphene_isomers = xtb_graphene_isomers + dft_graphene_isomers


basis_combos = {
    "qz_rijk": ["def2-QZVPP", "def2-QZVPP/JK", "def2-QZVPP/C"],
    "qz_riri": ["def2-QZVPP", "def2-QZVPP/C", "def2-QZVPP/C"],
    "qz_jkjk": ["def2-QZVPP", "def2-QZVPP/JK", "def2-QZVPP/JK"],
    "qz_ri": ["def2-QZVPP", "def2-QZVPP/C"],
    "qz_nori": ["def2-QZVPP"],
    "tz_rijk": ["def2-TZVP", "def2/JK", "def2-TZVP/C"],
    "tz_riri": ["def2-TZVP", "def2-TZVP/C", "def2-TZVP/C"],
    "tz_jkjk": ["def2-TZVP", "def2/JK", "def2/JK"],
    "tz_ri": ["def2-TZVP", "def2-TZVP/C"],
    "tz_nori": ["def2-TZVP"],
    "dz_rijk": ["def2-SVP", "def2/JK", "def2-SVP/C"],
    "dz_riri": ["def2-SVP", "def2-SVP/C", "def2-SVP/C"],
    "dz_jkjk": ["def2-SVP", "def2/JK", "def2/JK"],
    "dz_ri": ["def2-SVP", "def2-SVP/C"],
    "dz_nori": ["def2-SVP"],
}


class ORCACalculationToPerform:
    def __init__(self, isomer: GrapheneIsomer, basis_id: str):
        self.isomer = isomer
        self.basis_id = basis_id

        if basis_id not in basis_combos:
            raise ValueError(f"Basis combo {basis_id} not recognized.")

        self.primary_basis = basis_combos[basis_id][0]
        self.scf_aux_basis = None
        self.ri_aux_basis = None
        if len(basis_combos[basis_id]) > 2:
            self.scf_aux_basis = basis_combos[basis_id][1]
        if len(basis_combos[basis_id]) > 1:
            self.ri_aux_basis = basis_combos[basis_id][-1]

        self.input_filename = f"{isomer.name}_{self.basis_id}.inp"
        self.output_filename = f"{isomer.name}_{self.basis_id}.out"

    def input_filepath(self) -> Path:
        input_path = Path(__file__).parent / "orca" / "orca_inputs" / self.basis_id
        input_path.mkdir(parents=True, exist_ok=True)
        return input_path / self.input_filename

    def output_filepath(self) -> Path:
        output_path = Path(__file__).parent.parent / "outputs" / "orca"
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / self.output_filename

    def __repr__(self):
        return f"ORCACalculationToPerform(isomer={self.isomer.name}, basis_combo={self.basis_id})"


orca_calculations: list[ORCACalculationToPerform] = []

for isomer in all_graphene_isomers:
    if isomer.carbons == 24 and isomer.hydrogens == 14:
        for basis_name in basis_combos.keys():
            orca_calculations.append(ORCACalculationToPerform(isomer, basis_name))


class EXESSCalculationBatch:
    def __init__(self, initial_index: int):
        self.initial_index = initial_index
        self.final_index = initial_index - 1
        self.isomers: list[GrapheneIsomer] = []

    def add_isomer(self, isomer: GrapheneIsomer):
        self.isomers.append(isomer)
        self.final_index = self.initial_index + len(self.isomers) - 1

    def name(self) -> str:
        return f"isomers_{self.initial_index}-{self.final_index}"

    def input_file_path(self) -> Path:
        input_path = Path(__file__).parent / "exess" / "exess_inputs"
        input_path.mkdir(parents=True, exist_ok=True)
        return input_path / f"{self.name()}.json"


selected_isomers = list(
    filter(lambda x: x.carbons <= 32 or x.id < 40, all_graphene_isomers)
)

exess_batches: list[EXESSCalculationBatch] = []

BATCH_SIZE = 20

for i in range(0, len(selected_isomers), BATCH_SIZE):
    batch = selected_isomers[i : i + BATCH_SIZE]
    EXESS_batch = EXESSCalculationBatch(initial_index=i)
    for isomer in batch:
        EXESS_batch.add_isomer(isomer)
    exess_batches.append(EXESS_batch)
