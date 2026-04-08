from pathlib import Path

local_root = Path("/home/benjamin/Documents/CSMF")
container_root = Path("/workspace/repo/CSMF")
output_file = local_root / "docs" / "file_index.md"

# Ensure docs/ exists
output_file.parent.mkdir(parents=True, exist_ok=True)

# Directories to exclude
exclude_dirs = {
    local_root / ".git",
    #local_root / ".gitignore",
    local_root / "CSMF-ENV",
    local_root / ".pytest_cache",
    local_root / "docs" , 
    
}

def is_excluded(p: Path) -> bool:
    return any(exc in p.parents or p == exc for exc in exclude_dirs)

# Gather all files (excluding the output file itself and excluded dirs)
files = sorted(
    p for p in local_root.rglob("*")
    if p.is_file() and p != output_file and not is_excluded(p)
)

# Pre-include the output file itself
all_files = sorted(
    [output_file] + files,
    key=lambda p: str(p.relative_to(local_root))
)

# Remap to container paths
container_paths = [
    container_root / p.relative_to(local_root) for p in all_files
]

lines = ["# File Index\n",
         "## Files to Modify\n",
         "N/A\n",
         "\n## DO NOT Modify\n"]
for cp in container_paths:
    lines.append(f"- `{cp}`\n")

output_file.write_text("".join(lines))
print(f"Written {len(container_paths)} files to {output_file}")