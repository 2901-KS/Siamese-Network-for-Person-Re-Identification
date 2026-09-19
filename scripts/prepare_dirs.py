from pathlib import Path

for path in [
    Path("outputs/figures"),
    Path("outputs/results"),
    Path("checkpoints"),
]:
    path.mkdir(parents=True, exist_ok=True)

print("Project directories are ready.")
