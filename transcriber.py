"""Local inference, media inputs and exports for the desktop app."""

from contextlib import contextmanager, nullcontext
from ctypes import WinDLL
from dataclasses import dataclass
from itertools import groupby
from os import add_dll_directory, environ, fsync, get_exec_path, pathsep
from pathlib import Path
import re
from shutil import which
from sysconfig import get_path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import monotonic
from urllib.parse import parse_qs, urlsplit


MEDIA_EXTENSIONS = frozenset(".wav .mp3 .m4a .flac .ogg .mp4 .mkv .mov .webm".split())
WHISPER_MODEL = "large-v3"
_gpu_dll_directories = []


@dataclass(frozen=True, slots=True)
class Word:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass(frozen=True)
class Transcript:
    text: str
    words: tuple[Word, ...]


def identify_speakers(transcript, turns):
    turns = sorted(turns, key=lambda turn: turn.start)
    names = {}
    for turn in turns:
        names.setdefault(turn.speaker, f"Falante {len(names) + 1}")

    words, active = [], []
    index = 0
    for word in transcript.words:
        start, end = word.start, word.end
        active = [turn for turn in active if turn.end > start]
        cutoff = max(start + 0.001, end)
        while index < len(turns) and turns[index].start < cutoff:
            turn = turns[index]
            if turn.end > start:
                active.append(turn)
            index += 1
        speaker = "Falante não identificado"
        best_overlap = 0
        for turn in active:
            overlap = min(end, turn.end) - max(start, turn.start)
            if overlap > best_overlap or (
                best_overlap == 0 and start == end and turn.start <= start < turn.end
            ):
                speaker = names[turn.speaker]
                best_overlap = max(overlap, 0.001)
        words.append(Word(start, end, word.text, speaker))

    paragraphs = [
        f"{speaker}: {''.join(word.text for word in group).strip()}"
        for speaker, group in groupby(words, key=lambda word: word.speaker)
    ]
    return Transcript("\n\n".join(paragraphs) if words else transcript.text, tuple(words))


def youtube_url(value):
    try:
        url = urlsplit(value.strip())
    except ValueError:
        url = None
    video_id = ""
    if url and url.scheme in {"http", "https"}:
        if url.netloc.lower() == "youtu.be":
            video_id = url.path.removeprefix("/")
        elif url.netloc.lower() in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
            if url.path == "/watch":
                video_id = parse_qs(url.query).get("v", [""])[0]
            elif url.path.startswith(("/shorts/", "/live/")):
                video_id = url.path.split("/", 2)[-1]
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Cole o link de um único vídeo do YouTube.")
    return f"https://www.youtube.com/watch?v={video_id}"


@contextmanager
def local_media(source, status):
    if isinstance(source, Path):
        yield source
        return

    from yt_dlp import YoutubeDL

    url = youtube_url(source)
    runtime = next((name for name in ("deno", "node") if which(name)), None)
    if runtime is None:
        raise OSError("Instale Deno ou Node.js para baixar áudio do YouTube.")
    with TemporaryDirectory(prefix="transcritor-") as directory:
        status("Baixando áudio do YouTube…")
        options = {
            "format": "bestaudio",
            "noplaylist": True,
            "quiet": True,
            "noprogress": True,
            "no_warnings": True,
            "socket_timeout": 20,
            "retries": 2,
            "js_runtimes": {runtime: {}},
            "outtmpl": str(Path(directory) / "audio.%(ext)s"),
        }
        with YoutubeDL(options) as downloader:
            result = downloader.extract_info(url, download=True)
            path = Path(downloader.prepare_filename(result))
        if not path.is_file():
            raise OSError("O download não gerou um arquivo de áudio.")
        yield path


def preferred_device():
    try:
        from ctranslate2 import get_cuda_device_count, get_supported_compute_types

        if get_cuda_device_count() and "int8_float16" in get_supported_compute_types("cuda"):
            if not _gpu_dll_directories:
                packages = Path(__file__).with_name("nvidia")
                if not packages.is_dir():
                    packages = Path(get_path("purelib")) / "nvidia"
                directories = [
                    str(packages / name / "bin")
                    for name in ("cublas", "cuda_nvrtc", "cuda_runtime", "cudnn")
                    if (packages / name / "bin").is_dir()
                ]
                for directory in directories:
                    _gpu_dll_directories.append(add_dll_directory(directory))
                if directories:
                    environ["PATH"] = pathsep.join([*directories, environ.get("PATH", "")])
            for library in ("cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll"):
                found = next(
                    (folder / library for entry in get_exec_path()
                     if (folder := Path(entry)).is_absolute() and (folder / library).is_file()),
                    library,
                )
                WinDLL(str(found))
            return "cuda"
    except (ImportError, OSError, RuntimeError, ValueError):
        pass
    return "cpu"


class Engine:
    def __init__(self):
        self.model = None
        self.model_device = None
        self.device = "cpu"
        self.diarizer = None
        self.speaker_count = None

    def _load(self, status):
        from faster_whisper import WhisperModel

        if self.model_device != self.device:
            status(f"Preparando Whisper {WHISPER_MODEL} (pode baixar os pesos no primeiro uso)…")
            self.model = self.model_device = None
            precision = "int8_float16" if self.device == "cuda" else "int8"
            self.model = WhisperModel(WHISPER_MODEL, device=self.device, compute_type=precision)
            self.model_device = self.device
        return self.model

    def transcribe(
        self, path, language, timestamps, status, update, reset, *,
        diarization=False, speaker_count=0,
    ):
        audio = str(path)
        if diarization:
            from faster_whisper.audio import decode_audio

            status("Preparando áudio para transcrição e diarização…")
            audio = decode_audio(audio, sampling_rate=16000)
        result = self._recognize(audio, language, timestamps or diarization, status, update, reset)
        if diarization and result.words:
            result = identify_speakers(result, self._speaker_turns(audio, speaker_count, status, update))
        return result if timestamps else Transcript(result.text, ())

    def _recognize(self, audio, language, timestamps, status, update, reset):
        try:
            return self._transcribe(audio, language, timestamps, status, update)
        except (RuntimeError, OSError) as error:
            gpu_error = any(
                token in str(error).lower()
                for token in ("cuda", "cublas", "cudnn", "out of memory", "failed to allocate")
            )
            if self.device != "cuda" or not gpu_error:
                raise

        self.device = "cpu"
        self.model = self.model_device = None
        reset()
        return self._transcribe(audio, language, timestamps, status, update)

    def _speaker_turns(self, audio, speaker_count, status, update):
        if self.diarizer is None or self.speaker_count != speaker_count:
            from diarization import load_diarizer

            status("Preparando os modelos de diarização do Hugging Face…")
            self.diarizer = load_diarizer(speaker_count)
            self.speaker_count = speaker_count
        status("Identificando mudanças de falante…")
        last_update = 0

        def progress(done, total):
            nonlocal last_update
            now = monotonic()
            if total and now - last_update >= 0.15:
                update("", min(99, 100 * done // total))
                last_update = now
            return 0

        return self.diarizer.process(audio, progress).sort_by_start_time()

    def _transcribe(self, audio, language, timestamps, status, update):
        model = self._load(status)
        status("Transcrevendo…")
        segments, info = model.transcribe(
            audio, language=language, word_timestamps=timestamps, vad_filter=True
        )
        lines, words, pending = [], [], []
        last_update = 0
        progress = 0
        for segment in segments:
            line = segment.text.strip()
            if line:
                lines.append(line)
                pending.append(line)
            if timestamps:
                words.extend(Word(w.start, w.end, w.word) for w in segment.words or () if w.word.strip())
            if info.duration:
                progress = min(99, max(progress, int(100 * segment.end / info.duration)))
            now = monotonic()
            if now - last_update >= 0.15:
                update("\n".join(pending), progress)
                pending.clear()
                last_update = now
        update("\n".join(pending), 100)
        return Transcript("\n".join(lines), tuple(words))


def timestamp(seconds):
    minutes, millis = divmod(max(0, round(seconds * 1000)), 60_000)
    hours, minute = divmod(minutes, 60)
    second, millis = divmod(millis, 1000)
    return f"{hours:02}:{minute:02}:{second:02},{millis:03}"


def iter_srt(words):
    number = 0
    for word in words:
        text = word.text.strip()
        if not text:
            continue
        number += 1
        if word.speaker:
            text = f"{word.speaker}: {text}"
        end = max(word.end, max(0, word.start) + 0.001)
        yield f"{number}\n{timestamp(word.start)} --> {timestamp(end)}\n{text}\n\n"


def make_srt(words):
    return "".join(iter_srt(words))


@contextmanager
def _stage_export(directory, content):
    staged = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=directory, delete=False
        ) as output:
            staged = Path(output.name)
            if isinstance(content, str):
                output.write(content)
            else:
                output.writelines(content)
            output.flush()
            fsync(output.fileno())
        yield staged
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def save_export(path, content, *, replace, source=None):
    if source is not None:
        try:
            if path.samefile(source):
                raise FileExistsError("O destino é a mídia original. Escolha outro arquivo.")
        except FileNotFoundError:
            pass

    with _stage_export(path.parent, content) as staged:
        if replace:
            staged.replace(path)
        else:
            staged.rename(path)  # Windows rename fails if the destination already exists.


def save_batch(directory, stem, transcript, timestamps):
    subtitles = _stage_export(directory, iter_srt(transcript.words)) if timestamps else nullcontext()
    # Complete both writes before publishing either file.
    with _stage_export(directory, transcript.text) as staged_text, subtitles as staged_srt:
        suffix = 1
        while True:
            name = stem if suffix == 1 else f"{stem} ({suffix})"
            suffix += 1
            text_file = directory / f"{name}.txt"
            srt_file = directory / f"{name}.srt"
            if text_file.exists() or srt_file.exists():
                continue
            try:
                staged_text.rename(text_file)
            except FileExistsError:
                continue
            try:
                if staged_srt is not None:
                    staged_srt.rename(srt_file)
            except OSError as error:
                text_file.rename(staged_text)
                if not isinstance(error, FileExistsError):
                    raise
            else:
                return text_file.name
