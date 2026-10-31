"""Shared helpers for the scripts/CLI-layer tests (not a test module).

Everything here serves ``@pytest.mark.cli`` modules: standing in for
ConfigHandler and the batch logger, and staging a synthetic night tree on disk.
Builders only -- no assertions.
"""

from pathlib import Path

from astropy.io import fits

# Repo root: tests/regression/_scripts.py -> tests/regression -> tests -> repo.
REPO_ROOT = str(Path(__file__).resolve().parents[2])


class _FakeConfig:
    """Stand-in for ConfigHandler in the exit-code tests (no real file read)."""

    def __init__(self, path, overrides=None):
        # Every masters stage on, as in the shipped config, unless overridden.
        stages = dict.fromkeys(("bias", "dark", "flat", "order_trace", "thar"), True)
        self.stages = (overrides or {}).get("STAGES", stages)

    def get_params(self, keys):
        if keys == ["STAGES"]:
            return self.stages
        return {"KPF_DATA_INPUT": "/in", "log_dir": "/l"}


class _NoLogDirConfig(_FakeConfig):
    """A config with a resolvable data input but no configured log_dir."""

    def get_params(self, keys):
        return {"KPF_DATA_INPUT": "/in"}


def _stub_batch_log(monkeypatch, mod, tmp_path):
    """Stub ``mod.setup_batch_logging``: no real handlers, a .log path in tmp_path.

    The driver only reports the path, so nothing is created there. Returns it.
    """
    fake_log = str(tmp_path / "kpf_batch_x_20240405T000000.log")
    label = mod.__name__.rsplit(".", 1)[-1]
    monkeypatch.setattr(
        mod, "setup_batch_logging", lambda *a, **k: (f"{label}_x", fake_log)
    )
    return fake_log


def write_l0_tree(
    data_input,
    datecode,
    seconds,
    *,
    obj="10700",
    imtype="Object",
    targname=None,
    exptime=60.0,
    elapsed=60.0,
    extra_cards=None,
):
    """Write one PRIMARY-only L0 frame under ``{data_input}/L0/{datecode}``.

    Enough header for a mini-db header scan and nothing more; returns the obs_id.
    ``targname`` defaults to ``obj`` (calibration frames set them independently).
    ``extra_cards`` adds instrument cards past the mini-db set, for the analysis
    scripts that read their own (the ThAr lamp cards, say).
    """
    l0_dir = Path(data_input) / "L0" / datecode
    l0_dir.mkdir(parents=True, exist_ok=True)
    obs_id = f"KP.{datecode}.{seconds:05d}.00"
    header = fits.Header(
        {
            "OBJECT": obj,
            "IMTYPE": imtype,
            "TARGNAME": obj if targname is None else targname,
            "EXPTIME": exptime,
            "ELAPSED": elapsed,
            "GREEN": "YES",
            "RED": "YES",
            # A bias/dark must record every shutter closed.
            **(
                {k: "closed" for k in ("SCISEL", "SKYSEL", "SIMCALSH")}
                if imtype in ("Bias", "Dark")
                else {}
            ),
            **(extra_cards or {}),
        }
    )
    fits.PrimaryHDU(header=header).writeto(l0_dir / f"{obs_id}.fits")
    return obs_id


def add_junk_obs_id(data_input, obs_id):
    """Append obs_id to the WMKO junk list under ``{data_input}/vNext/reference/``."""
    ref = Path(data_input) / "vNext" / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    junk_csv = ref / "junk_obs.csv"
    if not junk_csv.exists():
        junk_csv.write_text("observation_id\n")
    with junk_csv.open("a") as fh:
        fh.write(f"{obs_id}\n")
