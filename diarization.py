"""Public ONNX models for local speaker diarization."""

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
import sherpa_onnx as sherpa


def model_file(repo, name):
    try:
        return hf_hub_download(repo, name, local_files_only=True, token=False)
    except LocalEntryNotFoundError:
        return hf_hub_download(repo, name, token=False)


def load_diarizer(speaker_count):
    segmentation = model_file("csukuangfj/sherpa-onnx-pyannote-segmentation-3-0", "model.int8.onnx")
    embedding = model_file(
        "csukuangfj/speaker-embedding-models",
        "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
    )
    config = sherpa.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa.OfflineSpeakerSegmentationPyannoteModelConfig(model=segmentation),
            num_threads=2,
        ),
        embedding=sherpa.SpeakerEmbeddingExtractorConfig(model=embedding, num_threads=2),
        clustering=sherpa.FastClusteringConfig(num_clusters=speaker_count or -1, threshold=0.6),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("A configuração do modelo de diarização é inválida.")
    return sherpa.OfflineSpeakerDiarization(config)
