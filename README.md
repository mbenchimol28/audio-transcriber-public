# Audio Transcriber

**Transcreva áudio e vídeo no Windows, com identificação de falantes e legendas por palavra.**

O processamento roda no computador com Whisper Large-v3. Abra uma gravação, cole o link de um vídeo público do YouTube ou monte uma fila de arquivos. A interface e os resultados estão disponíveis em português.

[Baixar código-fonte](https://github.com/mbenchimol28/audio-transcriber-public/releases/latest) · [Executar no Windows](#executar-no-windows) · [Identificar falantes](#identificar-falantes) · [Gerar um executável](#gerar-um-executável)

![Janela do Audio Transcriber com fila de arquivos, diarização ativada e uma transcrição de demonstração com dois falantes](docs/interface.png)

*Interface do aplicativo com dados de demonstração.*

| O que você pode fazer | Como funciona |
| --- | --- |
| Transcrever uma gravação | Abra um arquivo local ou cole o link de um vídeo público do YouTube. |
| Processar uma fila | Adicione arquivos individualmente ou pelo primeiro nível de uma pasta. Escolha onde salvar os resultados. |
| Separar falantes | Use a detecção automática ou indique o número de vozes. Os rótulos aparecem no resultado e nos arquivos exportados. |
| Exportar a transcrição | Salve em TXT ou habilite SRT para obter marcações de tempo por palavra. |

## Executar no Windows

Requer **Windows 10 ou 11 x64** e **Python 3.12 x64**. No PowerShell:

```powershell
git clone https://github.com/mbenchimol28/audio-transcriber-public.git
cd audio-transcriber-public
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Se você baixou o ZIP na aba [Releases](https://github.com/mbenchimol28/audio-transcriber-public/releases), extraia o arquivo, abra a pasta extraída no PowerShell e comece pelo comando que cria o ambiente virtual.

O primeiro uso baixa o [Whisper Large-v3](https://huggingface.co/Systran/faster-whisper-large-v3). Depois que os modelos estiverem em cache, você pode transcrever arquivos locais offline.

- **GPU NVIDIA.** Com CUDA 12, cuDNN 9 e driver compatível, o app usa a GPU. Se a aceleração estiver indisponível ou falhar, a transcrição continua na CPU. [Requisitos de GPU](https://github.com/SYSTRAN/faster-whisper#gpu)
- **YouTube.** Instale Deno 2.3+ ou Node.js 22+ e deixe o executável no PATH. A mídia temporária é removida ao terminar. [Requisitos do yt-dlp](https://github.com/yt-dlp/yt-dlp#dependencies)

## Identificar falantes

1. Marque **Identificar falantes (diarização local)**.
2. Escolha **Automático** ou informe o número de vozes. No processamento em lote, um número fixo vale para todos os arquivos.
3. Inicie a transcrição. A numeração de falantes recomeça a cada arquivo.

A diarização roda na CPU e baixa aproximadamente 41 MB de modelos públicos na primeira execução. Vozes sobrepostas ou parecidas podem receber rótulos imprecisos. [Segmentação pyannote](https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0) · [Modelo 3D-Speaker](https://huggingface.co/csukuangfj/speaker-embedding-models/blob/main/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx)

O token de leitura do Hugging Face é opcional. Se quiser usá-lo, preencha o campo e clique em **Salvar token**. O Windows protege o arquivo `%LOCALAPPDATA%\Transcritor\huggingface.dpapi` com a conta do usuário. Para apagar o token salvo, limpe o campo e clique em **Salvar token** novamente.

## Gerar um executável

Com o ambiente virtual já preparado, instale as dependências de empacotamento e execute:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1
```

Abra `dist\Transcritor\Transcritor.exe`. Para mover o aplicativo, copie a pasta `Transcritor` inteira, incluindo `_internal`.

Para compilar sem empacotar as bibliotecas NVIDIA, acrescente `-CpuOnly` ao comando do script. O resultado fica em `dist\portable\Transcritor`.

## Privacidade e desenvolvimento

A transcrição e a diarização são locais. O aplicativo desativa a telemetria do Hugging Face e do Deno. A conexão externa é usada para baixar modelos ou obter áudio de um link do YouTube fornecido por você.

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

O GitHub executa a análise estática de Python pelo CodeQL neste repositório. [Execuções do GitHub Actions](https://github.com/mbenchimol28/audio-transcriber-public/actions)

## Licença

Distribuído sob a [licença MIT](LICENSE). Os modelos Whisper, pyannote e 3D-Speaker e as bibliotecas usadas seguem as licenças dos respectivos projetos.
