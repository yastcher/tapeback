"""Regression: the transcription bench measured the developer's configuration.

`scripts/bench_transcribe.py` built a plain `Settings(...)` for each grid point, so a
`TAPEBACK_*` variable in the shell or a line in `.env` reached the measurement: a
developer with `TAPEBACK_BEAM_SIZE=1` got a table of beam 1 labelled as the default.
And `Transcriber` reuses a channel from the resume cache: a second run of the same
grid timed a cache hit, and the first one wrote into the developer's cache.
"""

from tests.fixtures import load_script

bench = load_script("bench_transcribe")


def test_a_grid_point_ignores_the_developers_environment(monkeypatch):
    monkeypatch.setenv("TAPEBACK_BEAM_SIZE", "1")
    monkeypatch.setenv("TAPEBACK_CHUNK_LENGTH", "2")

    settings = bench.grid_settings(
        model="large-v3-turbo", compute="int8_float16", device="cpu", language="ru", hotwords=""
    )

    assert settings.beam_size == 4
    assert settings.chunk_length == 30


def test_a_grid_point_neither_reads_nor_writes_the_resume_cache():
    settings = bench.grid_settings(
        model="large-v3-turbo", compute="int8_float16", device="cpu", language="ru", hotwords=""
    )

    assert settings.resume_cache is False
    assert settings.run_log is False
