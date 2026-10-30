"""Regression: the spectral profile stopped one frame short.

`_speaker_spectral_profile` started its frames at `range(0, len(chunk) - n_fft, hop)`,
which leaves out the last start, `len(chunk) - n_fft`. A segment exactly one FFT long
got no frame at all: its profile was silent, and that speaker could never be merged.
"""

import numpy as np

from tapeback.models import DiarizationSegment
from tapeback.speaker_merge import _speaker_spectral_profile


def test_a_segment_one_fft_long_has_a_profile():
    # 400 Hz at 16 kHz is bin 51 of a 2048-point FFT, the 40th of the 100-4000 Hz band.
    tone = np.sin(2 * np.pi * 400 * np.arange(2048) / 16000).astype(np.float32)
    segments = [DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.0)]

    profile = _speaker_spectral_profile(tone, 16000, segments, "SPEAKER_00")

    assert int(np.argmax(profile)) == 39
