import subprocess
from pathlib import Path
import shutil

def test_training_creates_model(tmp_path):
    """
    Create a temporary training folder structure with one person’s image,
    run the training command, and verify that a model file is created.
    """
    # Create temporary training directory: poi/TestPerson/
    train_dir = tmp_path / "poi"
    train_dir.mkdir(parents=True)

    person_dir = tmp_path / "poi" / "TestPerson"
    person_dir.mkdir(parents=True)

    # Copy the realistic face image from the tests folder
    src_image_path = Path(__file__).parent / "thispersondoesntexist.jpeg"
    dest_image_path = person_dir / "thispersondoesntexist.jpeg"
    shutil.copy(src_image_path, dest_image_path)

    # Set a path for the output model file (updated to the correct filename)
    model_save_path = tmp_path / "trained_knn_model.clf"
    print("tmp_path:", tmp_path)
    print("train_dir:", train_dir)
    print("person_dir:", person_dir)
    print("img path:", dest_image_path)
    print("model_save_path:", model_save_path)

    # Determine project root (assuming main.py is in the project root)
    project_root = Path(__file__).parent.parent.resolve()

    # Build the command to run training mode using absolute path to main.py
    cmd = [
        "python", str(project_root / "main.py"),
        "--train",
        "--train_dir", str(tmp_path / "poi"),
        "--model_save_path", str(model_save_path)
    ]

    # Run the command with the working directory set to the project root
    result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)

    # Assert that the process exited successfully
    assert result.returncode == 0, f"Training failed: {result.stderr}"
    # Assert that the model file was created
    assert model_save_path.exists(), "Model file was not created."
