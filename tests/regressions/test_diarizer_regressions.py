"""Regression tests for diarizer bugs."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from meetrec.settings import Settings


def test_diarizer_passes_token_correctly(tmp_vault):
    """Pipeline.from_pretrained must receive 'token' kwarg, not 'use_auth_token'.

    Bug: pyannote 4.x changed API from use_auth_token to token.
    Old code passed use_auth_token which caused TypeError.
    """
    from meetrec.diarizer import Diarizer

    settings = Settings(vault_path=tmp_vault, hf_token="hf_test_123", device="cpu")

    with patch("pyannote.audio.Pipeline") as mock_pipeline_cls:
        mock_pipeline_cls.from_pretrained.return_value = MagicMock()
        Diarizer(settings)

    _args, kwargs = mock_pipeline_cls.from_pretrained.call_args
    assert "use_auth_token" not in kwargs
    assert kwargs["token"] == "hf_test_123"


def test_diarize_handles_diarize_output(tmp_vault):
    """Diarizer.diarize should handle DiarizeOutput (pyannote 4.x) wrapping Annotation.

    Bug: pyannote 4.x returns DiarizeOutput instead of Annotation.
    Code called .itertracks() directly which raised AttributeError.
    """
    from meetrec.diarizer import Diarizer

    settings = Settings(vault_path=tmp_vault, hf_token="hf_fake_token", device="cpu")

    mock_turn = MagicMock()
    mock_turn.start = 0.0
    mock_turn.end = 3.0

    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = [(mock_turn, None, "SPEAKER_00")]

    # Simulate DiarizeOutput: no itertracks, but has speaker_diarization
    mock_diarize_output = MagicMock(spec=[])
    mock_diarize_output.speaker_diarization = mock_annotation

    with patch("pyannote.audio.Pipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_pipeline.return_value = mock_diarize_output

        diarizer = Diarizer(settings)
        result = diarizer.diarize(Path("/fake/audio.wav"))

    assert len(result) == 1
    assert result[0].speaker == "SPEAKER_00"


def test_diarize_cuda_init_fallback(tmp_vault, capsys):
    """Diarizer should fall back to CPU when CUDA is not available at init.

    Bug: Diarizer crashed if torch.device('cuda') failed.
    """
    from meetrec.diarizer import Diarizer

    settings = Settings(vault_path=tmp_vault, hf_token="hf_fake_token", device="cuda")

    with patch("pyannote.audio.Pipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.side_effect = RuntimeError("CUDA not available")

        Diarizer(settings)

    captured = capsys.readouterr()
    assert "CUDA not available for diarization" in captured.err


def test_diarize_cuda_oom_fallback(tmp_vault, capsys):
    """Diarizer.diarize should fall back to CPU on CUDA OOM during inference.

    Bug: GPU with 3.6 GiB couldn't hold both Whisper and pyannote models.
    Pipeline loaded on CUDA but crashed with OutOfMemoryError during inference.
    """
    from meetrec.diarizer import Diarizer

    settings = Settings(vault_path=tmp_vault, hf_token="hf_fake_token", device="cuda")

    mock_turn = MagicMock()
    mock_turn.start = 0.0
    mock_turn.end = 3.0
    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = [(mock_turn, None, "SPEAKER_00")]

    call_count = 0

    def pipeline_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("CUDA out of memory.")
        return mock_annotation

    with patch("pyannote.audio.Pipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_pipeline.side_effect = pipeline_call

        diarizer = Diarizer(settings)
        result = diarizer.diarize(Path("/fake/audio.wav"))

    assert len(result) == 1
    assert result[0].speaker == "SPEAKER_00"
    mock_pipeline.to.assert_called()
    captured = capsys.readouterr()
    assert "CUDA out of memory" in captured.err
