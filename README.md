# Transcritor

Aplicativo para Windows 10/11 x64 que transcreve áudio e vídeo no computador. A interface em português aceita uma mídia por vez, vídeo público do YouTube ou uma fila de arquivos. Exporta TXT e, quando selecionado, SRT com uma entrada por palavra.

## Abrir o executável

Gere o executável seguindo a seção [Gerar o executável](#gerar-o-executável). A pasta `build` contém arquivos intermediários de compilação. O executável dentro dela depende da montagem final em `dist`.

Abra `dist\Transcritor\Transcritor.exe`. Ao copiar o programa para outro local, leve a pasta `Transcritor` inteira, incluindo `_internal`. O pacote já inclui o Python, CUDA 12 e cuDNN 9. A aceleração requer uma GPU NVIDIA e um driver compatível.

Para usar os modelos já armazenados em `.venv\hf`, inicie pelo PowerShell na pasta do projeto.

```powershell
$env:HF_HOME = Join-Path $PWD '.venv\hf'
.\dist\Transcritor\Transcritor.exe
```

## Identificar falantes

Marque **Identificar falantes (diarização local)** antes de transcrever. Escolha **Automático** ou informe a quantidade de vozes, se souber. No lote, um número fixo vale para todos os arquivos.

Os rótulos aparecem no resultado, no TXT e no SRT quando a exportação de legendas estiver habilitada. A numeração recomeça a cada arquivo. Quando uma palavra fica fora dos trechos reconhecidos como fala, o aplicativo a marca como **Falante não identificado**. Vozes sobrepostas ou muito parecidas podem receber rótulos imprecisos.

O áudio é analisado no computador, em CPU. No primeiro uso, o aplicativo baixa aproximadamente 41 MB de pesos públicos. O cache permite usos posteriores offline com arquivos locais. [Segmentação pyannote 3.0](https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0) · [Modelo de voz 3D-Speaker](https://huggingface.co/csukuangfj/speaker-embedding-models/blob/main/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx)

Em **Identificar falantes**, você pode preencher **Token do Hugging Face** e clicar em **Salvar token**. O campo é opcional para os modelos públicos. Use um token de leitura. Ele é armazenado em `%LOCALAPPDATA%\Transcritor\huggingface.dpapi`, protegido pela conta do Windows, e reaparece mascarado ao reabrir o aplicativo. Para removê-lo, limpe o campo e clique em **Salvar token**. Cada pessoa deve usar seu próprio token.

O aplicativo desativa a telemetria do Hugging Face e do Deno. A transcrição e a diarização processam a mídia localmente. As conexões externas são usadas para baixar modelos ou para obter áudio quando você fornece um link do YouTube. O token salvo é usado apenas nas solicitações de modelos de diarização ao Hugging Face.

## Executar a partir do código-fonte

Instale o Python 3.12 x64 e execute estes comandos no PowerShell, na pasta do projeto.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Se o ambiente virtual já estiver instalado, execute apenas a última linha. Para reaproveitar os modelos armazenados em `.venv\hf`, rode primeiro `$env:HF_HOME = Join-Path $PWD '.venv\hf'`.

- **Whisper Large-v3.** O aplicativo usa este modelo tanto na GPU quanto na CPU. O primeiro uso baixa os pesos. Arquivos locais podem ser transcritos offline depois que o modelo estiver em cache. [Modelo no Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v3)
- **GPU NVIDIA.** A transcrição usa primeiro a GPU com precisão `int8_float16`. Se CUDA estiver indisponível ou falhar durante uma tarefa, o aplicativo usa o mesmo modelo em CPU com `int8` e mostra a troca na interface. Uma nova tarefa verifica a GPU novamente. A diarização roda na CPU. [Requisitos do faster-whisper](https://github.com/SYSTRAN/faster-whisper#gpu)
- **YouTube.** Instale Deno 2.3 ou superior, ou Node.js 22 ou superior, e deixe o executável no PATH. Cole um link individual, clique em **Usar link** e depois em **Transcrever**. A mídia temporária é removida quando a tarefa termina. [Requisitos do yt-dlp](https://github.com/yt-dlp/yt-dlp#dependencies)
- **Lotes.** Adicione arquivos ou escolha uma pasta. A fila inclui as mídias no primeiro nível. **Iniciar lote** pede o destino e grava os resultados conforme avança.

## Gerar o executável

Com as dependências de execução instaladas no ambiente virtual, instale as dependências de empacotamento e rode o script. As bibliotecas NVIDIA desse ambiente também habilitam a GPU ao rodar pelo código-fonte.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1
```

Para gerar o pacote portátil sem as bibliotecas NVIDIA, acrescente `-CpuOnly` à última linha. O executável fica em `dist\portable\Transcritor\Transcritor.exe`. Distribua a pasta inteira.

## Verificar

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

Para comparar com outro transcritor, use a mesma máquina, mídia, modelo e configuração de SRT. Meça a primeira execução separadamente das execuções com o modelo carregado.
