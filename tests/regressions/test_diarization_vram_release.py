"""Regression: diarization kept its VRAM after it finished.

The pipeline gave the GPU back after transcription but not after diarization. The
pyannote pipeline became garbage once the stage was over, yet torch's caching
allocator kept its memory reserved for the rest of the process. `tapeback start`
exits soon after; the tray does not. An idle installed tapeback was seen holding
2.8 GB of a 4 GB card, and the next recording's Whisper would find less than
`min_free_vram_mib` free and fall back to the CPU.

After diarization the pyannote pipeline is released and the CUDA cache is emptied
before the stage returns.
"""

import shutil
import weakref
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from tapeback.pipeline import process_mono_file, process_stereo_file
from tapeback.settings import Settings
from tests.fixtures import (
    create_mono_wav,
    create_stereo_wav_segments,
    mock_pyannote_annotation,
    mock_whisper_transcribe,
    requires_pyannote,
)


class _FakePipeline:
    """pyannote's pipeline, built fresh by every from_pretrained call: nothing but
    tapeback holds it, so a weakref tells when tapeback lets go."""

    def to(self, _device):
        return self

    def __call__(self, _audio, **_kwargs):
        return mock_pyannote_annotation([(0.0, 1.0, "SPEAKER_00")])


@requires_pyannote
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
@pytest.mark.parametrize("source", ["stereo", "mono"])
def test_diarization_gives_its_vram_back(tmp_path, source):
    pipelines: list[weakref.ref] = []
    # At each empty_cache: how many pipelines were built so far, and are all of them gone.
    cache_emptied: list[tuple[int, bool]] = []

    def from_pretrained(*_args, **_kwargs):
        pipeline = _FakePipeline()
        pipelines.append(weakref.ref(pipeline))
        return pipeline

    def empty_cache():
        cache_emptied.append((len(pipelines), all(ref() is None for ref in pipelines)))

    settings = Settings(vault_path=tmp_path, hf_token=SecretStr("hf_fake"), device="cpu")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    audio = tmp_path / f"{source}.wav"
    if source == "stereo":
        create_stereo_wav_segments(audio, 48000, [(1.0, 0.8, 0.003), (1.0, 0.003, 0.8)])
    else:
        create_mono_wav(audio, duration=2.0)

    with (
        patch(
            "tapeback.transcriber.WhisperModel",
            return_value=mock_whisper_transcribe([(0.0, 1.0, "Speech.")]),
        ),
        patch("pyannote.audio.Pipeline.from_pretrained", side_effect=from_pretrained),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.empty_cache", side_effect=empty_cache),
    ):
        process = process_stereo_file if source == "stereo" else process_mono_file
        process(audio, output_dir, settings, diarize=True)

    assert len(pipelines) == 1
    assert (1, True) in cache_emptied
