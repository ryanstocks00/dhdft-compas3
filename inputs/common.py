from pathlib import Path
import re
import subprocess


def get_all_xyzs(folder: Path) -> list[Path]:
    """Get all .xyz files in a folder."""
    assert folder.exists(), f"Folder {folder} does not exist."
    return list(sorted(folder.glob("*.xyz")))


class GrapheneIsomer:
    def __init__(self, xyz_path: Path, optimizer, check_xyz: bool = False):
        prefixes = {
            "xTB": "compas3x_",
            "DFT": "compas3D_",
            "G4(MP2)": "PAH335_",
        }
        prefix = prefixes.get(optimizer, "")
        self.xyz_path = xyz_path
        self.name = prefix + xyz_path.stem
        self.optimizer = optimizer

        if optimizer in ["xTB", "DFT"]:
            m = re.match(r"compas3._hc_c(\d+)h(\d+)_0pent_(\d+)", self.name)
            if m:
                self.carbons = int(m.group(1))
                self.hydrogens = int(m.group(2))
                self.id = int(m.group(3))
            else:
                raise ValueError(
                    f"Filename {self.name} does not match expected pattern."
                )
        elif optimizer == "G4(MP2)":
            m = re.match(r"PAH335_C(\d+)H(\d+)_pah(\d+)", self.name)
            if m:
                self.carbons = int(m.group(1))
                self.hydrogens = int(m.group(2))
                self.id = int(m.group(3))
            else:
                raise ValueError(
                    f"Filename {self.name} does not match expected pattern."
                )
        if check_xyz:
            with open(self.xyz_path, "r") as f:
                lines = f.readlines()
                num_atoms = int(lines[0].strip())
                expected_num_atoms = self.carbons + self.hydrogens
                if num_atoms != expected_num_atoms:
                    raise ValueError(
                        f"Number of atoms in {self.name} ({num_atoms}) does not match expected ({expected_num_atoms})."
                    )
                n_carbons = sum(1 for line in lines[2:] if line.strip().startswith("C"))
                n_hydrogens = sum(
                    1 for line in lines[2:] if line.strip().startswith("H")
                )
                if n_carbons != self.carbons or n_hydrogens != self.hydrogens:
                    raise ValueError(
                        f"Number of C/H atoms in {self.name} ({n_carbons} C, {n_hydrogens} H) does not match expected ({self.carbons} C, {self.hydrogens} H)."
                    )

    def __repr__(self):
        return f"GrapheneIsomer(name={self.name}, carbons={self.carbons}, hydrogens={self.hydrogens}, id={self.id}, optimizer={self.optimizer})"


def clone_compas_repo(
    gitlab_url="https://gitlab.com/porannegroup/compas.git", cache_dir=None
):
    """Clone the COMPAS-3 repository from GitLab to .compas_cache if it doesn't exist."""
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / ".compas_cache"
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    repo_url = gitlab_url if gitlab_url.endswith(".git") else gitlab_url + ".git"
    repo_dir = cache_dir / "compas"

    if repo_dir.exists() and (repo_dir / ".git").exists():
        subprocess.run(["git", "pull"], cwd=repo_dir, check=False, capture_output=True)
        return repo_dir

    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", "main", repo_url, str(repo_dir)],
        check=True,
    )
    return repo_dir


def get_all_graphene_isomers(optimizer: str) -> list[GrapheneIsomer]:
    """Get all GrapheneIsomer objects from .xyz files in a folder."""

    if optimizer == "xTB":
        folder = Path(__file__).parent / "compas3x-xyzs"
        # Ensure COMPAS repo is cloned before accessing tar.gz
        clone_compas_repo()
        tar_path = (
            Path(__file__).parent.parent
            / ".compas_cache"
            / "compas"
            / "COMPAS-3"
            / "compas-3x.tar.gz"
        )
    elif optimizer == "DFT":
        folder = Path(__file__).parent / "compas3D-xyzs"
        # Ensure COMPAS repo is cloned before accessing tar.gz
        clone_compas_repo()
        tar_path = (
            Path(__file__).parent.parent
            / ".compas_cache"
            / "compas"
            / "COMPAS-3"
            / "compas-3D.tar.gz"
        )
    elif optimizer == "G4(MP2)":
        folder = Path(__file__).parent / "PAH335_Structures"
        tar_path = folder.parent / "PAH335_Structures.tar.gz"
    else:
        raise ValueError(f"Optimizer {optimizer} not recognized. Use 'xTB' or 'DFT'.")

    if not folder.exists():
        # try unzipping compas-3x.tar.gz from .compas_cache
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


g4mp2_pahs = get_all_graphene_isomers("G4(MP2)")
xtb_graphene_isomers = get_all_graphene_isomers("xTB")
dft_graphene_isomers = get_all_graphene_isomers("DFT")

all_compas3_graphene_isomers = xtb_graphene_isomers + dft_graphene_isomers
all_structures = all_compas3_graphene_isomers + g4mp2_pahs


basis_combos = {
    "qz_rijk": ["def2-QZVPP", "def2/JK", "def2-QZVPP/C"],
    "qz_riri": ["def2-QZVPP", "def2-QZVPP/C", "def2-QZVPP/C"],
    "qz_jkjk": ["def2-QZVPP", "def2/JK", "def2/JK"],
    "qz_ri": ["def2-QZVPP", "def2-QZVPP/C"],
    # "qz_nori": ["def2-QZVPP"], # Too expensive, >24 hrs with ORCA
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

# Load isomers from size_profiling.json
size_profiling_path = Path(__file__).parent.parent / "inputs" / "size_profiling.json"
size_profiling_isomer_names = set()
if size_profiling_path.exists():
    import json

    with open(size_profiling_path, "r") as f:
        size_profiling_data = json.load(f)
    for topo in size_profiling_data.get("topologies", []):
        xyz_path = Path(topo["xyz"])
        # Extract name like 'hc_c16h10_0pent_1' from path
        isomer_name = xyz_path.stem
        size_profiling_isomer_names.add(isomer_name)

# Create ORCA calculations for all isomers in size_profiling.json
for isomer in all_compas3_graphene_isomers:
    # Match by xyz_path stem (without prefix)
    isomer_base_name = isomer.xyz_path.stem
    if isomer_base_name in size_profiling_isomer_names:
        for basis_name in basis_combos.keys():
            orca_calculations.append(ORCACalculationToPerform(isomer, basis_name))


def basis_to_aux_basis(basis: str) -> str:
    """Get the auxiliary basis for a given primary basis."""
    aux_basis_map = {
        "def2-QZVPP": "def2-QZVPP-RIFIT",
        "def2-TZVP": "def2-TZVP-RIFIT",
        "def2-SVP": "def2-SVP-RIFIT",
    }
    if basis not in aux_basis_map:
        raise ValueError(f"Auxiliary basis for {basis} not found.")
    return aux_basis_map[basis]


functional_to_name = {
    "revDSD-PBEP86-D4": "revDSD-PBEP86-D4(noFC)",
    "PBE0": "PBE0",
    "SVWN5": "SVWN5",
    "PBE": "PBE",
    "BLYP": "BLYP",
    "revPBE": "GGA_X_PBE_R+GGA_C_PBE",
    "BP86": "GGA_X_B88+GGA_C_P86",
    "BPW91": "GGA_X_B88+GGA_C_PW91",
    "B97-D": "GGA_XC_B97_D",
    "HCTH407": "GGA_XC_HCTH_407",
    "TPSS": "MGGA_C_TPSS+MGGA_X_TPSS",
    "MN15L": "MGGA_X_MN15_L+MGGA_C_MN15_L",
    "SCAN": "MGGA_X_SCAN+MGGA_C_SCAN",
    "rSCAN": "MGGA_X_RSCAN+MGGA_C_RSCAN",
    "r2SCAN": "MGGA_X_R2SCAN+MGGA_C_R2SCAN",
    "revTPSS": "MGGA_X_REVTPSS+MGGA_C_REVTPSS",
    "t-HCTH": "MGGA_X_TAU_HCTH+GGA_C_TAU_HCTH",
    "M06-L": "MGGA_X_M06_L+MGGA_C_M06_L",
    "M11-L": "MGGA_X_M11_L+MGGA_C_M11_L",
}


class EXESSCalculationBatch:
    def __init__(
        self,
        initial_index: int,
        prefix: str = "isomers",
        basis: str = "def2-QZVPP",
        functional: str = "revDSD-PBEP86-D4",
    ):
        self.initial_index = initial_index
        self.basis = basis
        self.aux_basis = basis_to_aux_basis(basis)
        self.functional = functional
        self.functional_name = functional_to_name[functional]
        self.final_index = initial_index - 1
        self.isomers: list[GrapheneIsomer] = []
        self.prefix = prefix

    def add_isomer(self, isomer: GrapheneIsomer):
        self.isomers.append(isomer)
        self.final_index = self.initial_index + len(self.isomers) - 1

    def name(self) -> str:
        return f"{self.prefix}_{self.initial_index}-{self.final_index}"

    def input_file_path(self) -> Path:
        input_path = (
            Path(__file__).parent
            / "exess"
            / "exess_inputs"
            / self.functional
            / self.basis
        )
        input_path.mkdir(parents=True, exist_ok=True)
        return input_path / f"{self.name()}.json"

    def output_file_path(self) -> Path:
        output_path = (
            Path(__file__).parent.parent
            / "outputs"
            / "exess"
            / self.functional
            / self.basis
        )
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / f"{self.name()}.out"


# selected_isomers = list(
# filter(lambda x: x.carbons <= 32 or x.id < 40, all_graphene_isomers)
# )
selected_isomers = list(all_compas3_graphene_isomers)

exess_batches: list[EXESSCalculationBatch] = []

BATCH_SIZE = 20

for i in range(0, len(selected_isomers), BATCH_SIZE):
    batch = selected_isomers[i : i + BATCH_SIZE]
    EXESS_batch = EXESSCalculationBatch(initial_index=i)
    for isomer in batch:
        EXESS_batch.add_isomer(isomer)
    exess_batches.append(EXESS_batch)

exess_svwn_batches: list[EXESSCalculationBatch] = []
for i in range(0, len(xtb_graphene_isomers), BATCH_SIZE):
    batch = xtb_graphene_isomers[i : i + BATCH_SIZE]
    EXESS_batch = EXESSCalculationBatch(
        initial_index=i, basis="def2-TZVP", functional="SVWN5"
    )
    for isomer in batch:
        EXESS_batch.add_isomer(isomer)
    exess_svwn_batches.append(EXESS_batch)

exess_gga_batches: list[EXESSCalculationBatch] = []
for gga in ["PBE", "BLYP", "revPBE", "BP86", "BPW91", "B97-D", "HCTH407"]:
    for i in range(0, len(xtb_graphene_isomers), BATCH_SIZE):
        batch = xtb_graphene_isomers[i : i + BATCH_SIZE]
        EXESS_batch = EXESSCalculationBatch(
            initial_index=i, basis="def2-TZVP", functional=gga
        )
        for isomer in batch:
            EXESS_batch.add_isomer(isomer)
        exess_gga_batches.append(EXESS_batch)

exess_pah335_batches = []
for i in range(0, len(g4mp2_pahs), BATCH_SIZE):
    batch = g4mp2_pahs[i : i + BATCH_SIZE]
    EXESS_batch = EXESSCalculationBatch(initial_index=i, prefix="PAH335")
    for isomer in batch:
        EXESS_batch.add_isomer(isomer)
    exess_pah335_batches.append(EXESS_batch)

exess_pah335_pbe_batches = []
for i in range(0, len(g4mp2_pahs), BATCH_SIZE):
    batch = g4mp2_pahs[i : i + BATCH_SIZE]
    EXESS_batch_pbe_qz = EXESSCalculationBatch(
        initial_index=i, prefix="PAH335", functional="PBE0", basis="def2-QZVPP"
    )
    EXESS_batch_pbe_tz = EXESSCalculationBatch(
        initial_index=i, prefix="PAH335", functional="PBE0", basis="def2-TZVP"
    )
    for isomer in batch:
        EXESS_batch_pbe_qz.add_isomer(isomer)
        EXESS_batch_pbe_tz.add_isomer(isomer)
    exess_pah335_pbe_batches.append(EXESS_batch_pbe_qz)
    exess_pah335_pbe_batches.append(EXESS_batch_pbe_tz)

exess_mgga_batches: list[EXESSCalculationBatch] = []
for mgga in ["TPSS", "MN15L", "SCAN", "rSCAN", "r2SCAN", "revTPSS", "t-HCTH", "M06-L", "M11-L"]:
    for i in range(0, len(xtb_graphene_isomers), BATCH_SIZE):
        batch = xtb_graphene_isomers[i : i + BATCH_SIZE]
        EXESS_batch = EXESSCalculationBatch(
            initial_index=i, basis="def2-TZVP", functional=mgga
        )
        for isomer in batch:
            EXESS_batch.add_isomer(isomer)
        exess_mgga_batches.append(EXESS_batch)
