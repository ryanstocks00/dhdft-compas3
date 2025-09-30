from pathlib import Path
import re


def get_all_xyzs(folder: Path) -> list[Path]:
    """Get all .xyz files in a folder."""
    assert folder.exists(), f"Folder {folder} does not exist."
    return list(sorted(folder.glob("*.xyz")))


class GrapheneIsomer:
    def __init__(self, xyz_path: Path):
        self.xyz_path = xyz_path
        self.name = xyz_path.stem

        m = re.match(r"hc_c(\d+)h(\d+)_0pent_(\d+)", self.name)
        if m:
            self.carbons = int(m.group(1))
            self.hydrogens = int(m.group(2))
            self.id = int(m.group(3))
        else:
            raise ValueError(f"Filename {self.name} does not match expected pattern.")

    def __repr__(self):
        return f"GrapheneIsomer(name={self.name}, carbons={self.carbons}, hydrogens={self.hydrogens}, id={self.id})"


def get_all_graphene_isomers(
    folder: Path = Path(__file__).parent / "compas3x-xyzs",
) -> list[GrapheneIsomer]:
    """Get all GrapheneIsomer objects from .xyz files in a folder."""
    if not folder.exists():
        # try unzipping compas-3x.tar.gz from the parent directory
        import tarfile
        import shutil
        import sys
        parent_folder = folder.parent
        tar_path = parent_folder / "compas-3x.tar.gz"
        if not tar_path.exists():
            print(f"Error: {tar_path} does not exist.", file=sys.stderr)
            sys.exit(1)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=parent_folder)
        if not folder.exists():
            print(f"Error: {folder} does not exist after extracting {tar_path}.", file=sys.stderr)
            sys.exit(1)



    xyz_files = get_all_xyzs(folder)
    graphenes = [GrapheneIsomer(x) for x in xyz_files]
    graphenes.sort(key=lambda g: g.id)
    return graphenes


if __name__ == "__main__":
    xyz_files = get_all_graphene_isomers()
    for xyz in xyz_files:
        print(f"Loaded isomer: {xyz}")
