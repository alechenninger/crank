"""CLI integration tests."""

from click.testing import CliRunner

from crank.cli import main


def test_rank_demo_table() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["rank", "--demo"])
    assert result.exit_code == 0
    assert "prod-eu-pci" in result.output
    assert "RANK" in result.output


def test_rank_demo_json() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["rank", "--demo", "--format", "json"])
    assert result.exit_code == 0
    assert '"scoring_mode"' in result.output
    assert '"base_score"' in result.output


def test_train_command(tmp_path) -> None:
    runner = CliRunner()
    dataset = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "examples"
        / "training_dataset.jsonl"
    )
    out = tmp_path / "model.joblib"
    result = runner.invoke(main, ["train", "--dataset", str(dataset), "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()


def test_rank_requires_clusters_without_demo() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["rank"])
    assert result.exit_code != 0
