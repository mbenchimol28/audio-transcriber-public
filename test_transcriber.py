import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from transcriber import Engine, Transcript, Word, identify_speakers, make_srt, save_batch, youtube_url


class TranscriberTests(unittest.TestCase):
    def test_speakers_follow_the_strongest_overlap_and_keep_unmatched_words(self):
        words = (Word(0, 1, "Olá"), Word(1, 2, " novamente"), Word(3, 4, "Oi"), Word(5, 6, "Fim"))
        turns = [
            SimpleNamespace(start=2, end=4.8, speaker=3),
            SimpleNamespace(start=0, end=2.8, speaker=7),
        ]
        marked = identify_speakers(Transcript("Olá novamente. Oi. Fim.", words), turns)
        self.assertEqual(
            marked.text, "Falante 1: Olá novamente\n\nFalante 2: Oi\n\nFalante não identificado: Fim"
        )
        self.assertIn("Falante 2: Oi", make_srt(marked.words))

    def test_youtube_url_accepts_a_video_and_removes_playlist(self):
        expected = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
        self.assertEqual(youtube_url("https://youtu.be/jNQXAC9IVRw?list=123"), expected)
        with self.assertRaises(ValueError):
            youtube_url("https://www.youtube.com.example.com/watch?v=jNQXAC9IVRw")
        with self.assertRaises(ValueError):
            youtube_url("https://www.youtube.com/playlist?list=123")

    def test_srt_numbering_and_millisecond_format(self):
        self.assertEqual(
            make_srt((Word(0.125, 0.25, "Olá"), Word(60, 60, "mundo"))),
            "1\n00:00:00,125 --> 00:00:00,250\nOlá\n\n"
            "2\n00:01:00,000 --> 00:01:00,001\nmundo\n\n",
        )

    def test_batch_preserves_existing_output_and_pairs_filenames(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            existing = directory / "voz.txt"
            existing.write_text("original", encoding="utf-8")
            result = Transcript("Olá", (Word(0, 1, "Olá"),))
            self.assertEqual(save_batch(directory, "voz", result, True), "voz (2).txt")
            self.assertEqual(existing.read_text(encoding="utf-8"), "original")
            self.assertEqual((directory / "voz (2).txt").read_text(encoding="utf-8"), "Olá")
            self.assertIn("Olá", (directory / "voz (2).srt").read_text(encoding="utf-8"))
            self.assertEqual(len(list(directory.iterdir())), 3)

    def test_gpu_error_retries_same_item_on_cpu_and_resets_partial_result(self):
        engine = Engine()
        engine.device = "cuda"
        devices, resets = [], []
        transcript = Transcript("Olá", ())

        def transcribe(*_):
            devices.append(engine.device)
            if engine.device == "cuda":
                raise RuntimeError("CUDA out of memory")
            return transcript

        with patch.object(engine, "_transcribe", side_effect=transcribe):
            result = engine.transcribe(
                Path("voz.wav"), "pt", False, None, None, lambda: resets.append(True)
            )
        self.assertEqual(result, transcript)
        self.assertEqual(devices, ["cuda", "cpu"])
        self.assertEqual(resets, [True])


if __name__ == "__main__":
    unittest.main()
