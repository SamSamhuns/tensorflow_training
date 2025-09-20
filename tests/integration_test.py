import subprocess
import tempfile
import os
import pytest
from pathlib import Path


@pytest.mark.integration
def test_train_runs_successfully():
    """Integration test for train.py end-to-end training run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # point model/log dirs to temp
        cfg_path = "config/train_image_clsf.yaml"
        run_id = "pytest_test_run"

        cmd = [
            "python",
            "train.py",
            "--cfg", cfg_path,
            "--id", run_id,
            "-o",  # overrides to make test fast
            "save_dir=" + tmpdir,
            "trainer.epochs=1",
            "data.num_train_samples=8",
            "data.num_val_samples=4",
            "data.train_bsize=4",
            "data.val_bsize=2",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # debug on failure
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)

        assert result.returncode == 0, "train.py crashed"

        # check that model was saved
        model_dir = Path(tmpdir) / "Image_Classification" / run_id / "models"
        assert model_dir.exists(), "Model directory not created"
        saved = list(model_dir.glob("**/*"))
        assert saved, "No model artifacts saved"
