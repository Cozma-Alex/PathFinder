import os
import shutil
import subprocess

# Ensure directories exist
os.makedirs("dist", exist_ok=True)
os.makedirs("build", exist_ok=True)

# Create an empty trained_models directory if it doesn't exist
# This will be available for users to add their models
models_dir = "dist/trained_models"
os.makedirs(models_dir, exist_ok=True)

# Run PyInstaller
print("Building application with PyInstaller...")
result = subprocess.run(["pyinstaller", "--clean", "pathfinder.spec"], 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE, 
                       text=True)

print(result.stdout)
if result.stderr:
    print("Errors/Warnings:")
    print(result.stderr)

if result.returncode != 0:
    print("Build failed!")
    exit(1)

print("\nBuild completed successfully!")
print(f"\nExecutable located at: {os.path.abspath('dist/PathFinder')}")