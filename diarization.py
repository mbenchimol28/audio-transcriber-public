"""Public ONNX models for local speaker diarization."""

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
import sherpa_onnx as sherpa
from local_token import load_token


def model_file(repo, name, token):
    endpoint = "https://huggingface.co"
    try:
        return hf_hub_download(repo, name, local_files_only=True, token=False, endpoint=endpoint)
    except LocalEntryNotFoundError:
        try:
            return hf_hub_download(repo, name, token=token or False, endpoint=endpoint)
        except HfHubHTTPError as error:
            code = error.response.status_code if error.response is not None else None
            if code in (401, 403):
                raise RuntimeError("O Hugging Face recusou o download. Confira o token de leitura salvo.") from None
            raise RuntimeError("Não foi possível baixar o modelo de diarização do Hugging Face.") from None


def load_diarizer(speaker_count):
    try:
        token = load_token()
    except (OSError, UnicodeError):
        raise RuntimeError("Não foi possível ler o token local do Hugging Face. Salve um novo token.") from None
    segmentation = model_file("csukuangfj/sherpa-onnx-pyannote-segmentation-3-0", "model.int8.onnx", token)
    embedding = model_file(
        "csukuangfj/speaker-embedding-models",
        "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        token,
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
