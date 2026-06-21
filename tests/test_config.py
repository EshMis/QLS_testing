from qls_testing.core.config import load_config


def test_dotted_cli_overrides_are_typed_and_validated():
    config = load_config(
        "configs/default.yaml",
        ["time.dt=0.05", "linearization.settings.order=3", "output.save_plot=false"],
    )
    assert config.time.dt == 0.05
    assert config.linearization.settings["order"] == 3
    assert config.output.save_plot is False

