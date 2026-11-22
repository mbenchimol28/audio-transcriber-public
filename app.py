"""Windows desktop entry point."""

from dataclasses import dataclass
import faulthandler
import logging
from os import environ
from pathlib import Path
import sys


def configure_diagnostics():
    folder = Path(environ.get("LOCALAPPDATA", Path.home())) / "Transcritor"
    folder.mkdir(parents=True, exist_ok=True)
    stream = (folder / "transcritor.log").open("a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream
    logging.basicConfig(
        level=logging.INFO, stream=stream, format="%(asctime)s %(levelname)s %(message)s"
    )
    faulthandler.enable(file=stream)
    logging.info("Iniciando %s", sys.executable)
    return stream


from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPalette, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSpinBox,
    QVBoxLayout, QWidget,
)

from transcriber import (
    Engine, MEDIA_EXTENSIONS, Transcript, WHISPER_MODEL, local_media, make_srt, preferred_device,
    save_batch, save_export, youtube_url,
)


MEDIA_FILTER = "Áudio e vídeo (" + " ".join(f"*{ext}" for ext in sorted(MEDIA_EXTENSIONS)) + ")"
LANGUAGES = {
    "Português": "pt", "Detectar automaticamente": None, "Inglês": "en",
    "Espanhol": "es", "Francês": "fr", "Alemão": "de", "Italiano": "it",
    "Japonês": "ja", "Chinês": "zh",
}


@dataclass(frozen=True)
class Job:
    sources: tuple[Path | str, ...]
    folder: Path | None
    language: str | None
    timestamps: bool
    diarization: bool
    speaker_count: int


class Worker(QObject):
    device = Signal(str)
    status = Signal(str)
    update = Signal(str, int)
    reset = Signal()
    log = Signal(str)
    finished = Signal(object, str)

    def __init__(self):
        super().__init__()
        self.engine = Engine()

    @Slot()
    def initialize(self):
        self.engine.device = preferred_device()
        self.device.emit(self.engine.device)

    @Slot(object)
    def run(self, job):
        if self.engine.device == "cpu":
            self.initialize()
        failures = 0
        for index, source in enumerate(job.sources, 1):
            name = source.stem if isinstance(source, Path) else source.rsplit("=", 1)[-1]
            label = source.name if isinstance(source, Path) else f"YouTube · {name}"
            prefix = f"{index}/{len(job.sources)} · " if job.folder else ""

            def status(message):
                logging.info("%s%s — %s", prefix, label, message)
                self.status.emit(f"{prefix}{label} — {message}")

            def update(text, progress):
                self.update.emit(text if not job.folder else "", progress)

            def reset():
                self.reset.emit()
                self.device.emit("cpu")
                self.log.emit(f"{label} · GPU indisponível, repetindo em CPU.")

            try:
                with local_media(source, status) as path:
                    result = self.engine.transcribe(
                        path, job.language, job.timestamps, status, update, reset,
                        diarization=job.diarization,
                        speaker_count=job.speaker_count,
                    )
                if not job.folder:
                    self.finished.emit(result, name)
                    return
                exported = save_batch(job.folder, name, result, job.timestamps)
                self.log.emit(f"{label} · Salvo como {exported}")
            except Exception as error:
                logging.exception("Falha em %s", label)
                detail = str(error).strip() or type(error).__name__
                if not job.folder:
                    self.log.emit(detail)
                    self.finished.emit(None, "Falha na transcrição. Confira o resultado.")
                    return
                failures += 1
                self.log.emit(f"{label} · Falha: {detail}")
        self.finished.emit(
            None, f"Lote concluído · {len(job.sources) - failures} salvo(s) · {failures} com falha"
        )


def label(text, style=None):
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    if style:
        widget.setObjectName(style)
    return widget


def button(text, action, style="secondary"):
    widget = QPushButton(text)
    widget.setProperty("role", style)
    widget.clicked.connect(action)
    return widget


def card(title):
    panel = QFrame()
    panel.setObjectName("card")
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(9)
    header = QHBoxLayout()
    header.addWidget(label(title, "sectionTitle"))
    header.addStretch()
    layout.addLayout(header)
    return panel, layout, header


class Window(QMainWindow):
    start = Signal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Transcritor")
        self.resize(1040, 870)
        self.setMinimumSize(440, 560)
        self.source = None
        self.paths = {}
        self.result = None
        self.result_name = ""
        self.busy = False
        self.initializing = True
        self.is_batch = False
        self.used_cpu_fallback = False
        self.stacked = None
        self._build()

        self.thread = QThread(self)
        self.worker = Worker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.initialize)
        self.thread.finished.connect(self.worker.deleteLater)
        self.start.connect(self.worker.run)
        self.worker.device.connect(self.show_device)
        self.worker.status.connect(self.show_status)
        self.worker.update.connect(self.show_update)
        self.worker.reset.connect(self.reset_progress)
        self.worker.log.connect(self.output.appendPlainText)
        self.worker.finished.connect(self.finish)
        self.thread.start()
        self.sync_buttons()

    def _build(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.setCentralWidget(scroll)
        content = QWidget()
        content.setObjectName("content")
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(22, 18, 22, 22)
        layout.setSpacing(12)

        header = QHBoxLayout()
        header.addWidget(label("Transcritor", "title"))
        header.addStretch()
        self.device_label = label("Detectando dispositivo…", "device")
        header.addWidget(self.device_label)
        layout.addLayout(header)

        settings = QFrame()
        settings.setObjectName("card")
        controls = QGridLayout(settings)
        controls.setContentsMargins(16, 10, 16, 12)
        controls.setVerticalSpacing(4)
        self.language = QComboBox()
        for name, code in LANGUAGES.items():
            self.language.addItem(name, code)
        language_label = QLabel("&Idioma")
        language_label.setBuddy(self.language)
        controls.addWidget(language_label, 0, 0)
        controls.addWidget(self.language, 1, 0)
        controls.addWidget(label("Modelo"), 0, 1)
        controls.addWidget(label(f"Whisper {WHISPER_MODEL}"), 1, 1)
        controls.setColumnStretch(0, 1)
        controls.setColumnStretch(1, 1)
        layout.addWidget(settings)

        self.panels = QGridLayout()
        self.panels.setSpacing(12)
        layout.addLayout(self.panels)
        self.single_panel, single, _ = card("Arquivo ou YouTube")
        self.source_label = label("Nenhuma mídia selecionada", "muted")
        self.source_label.setWordWrap(True)
        single.addWidget(self.source_label)
        self.browse = button("Escolher arquivo", self.choose_file)
        single.addWidget(self.browse)
        self.url = QLineEdit()
        self.url.setPlaceholderText("Cole o link de um vídeo do YouTube")
        self.url.setAccessibleName("Link do YouTube")
        self.url.textChanged.connect(self.sync_buttons)
        self.url.returnPressed.connect(self.choose_link)
        self.use_link = button("Usar link", self.choose_link)
        url_row = QHBoxLayout()
        url_row.addWidget(self.url, 1)
        url_row.addWidget(self.use_link)
        single.addLayout(url_row)
        single.addStretch()
        self.transcribe = button("Transcrever", self.start_single, "primary")
        single.addWidget(self.transcribe)

        self.batch_panel, batch, _ = card("Lote")
        self.queue = QListWidget()
        self.queue.setAccessibleName("Arquivos da fila")
        self.queue.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.queue.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        self.queue.setMinimumHeight(58)
        self.queue.setMaximumHeight(92)
        batch.addWidget(self.queue)
        batch_options = QHBoxLayout()
        self.add_files = button("Adicionar arquivos", self.choose_files)
        self.add_folder = button("Adicionar pasta", self.choose_folder)
        batch_options.addWidget(self.add_files, 1)
        batch_options.addWidget(self.add_folder, 1)
        batch.addLayout(batch_options)
        batch_actions = QHBoxLayout()
        self.queue_count = label("Fila vazia", "muted")
        batch_actions.addWidget(self.queue_count)
        batch_actions.addStretch()
        self.clear = button("Limpar lista", self.clear_queue, "quiet")
        batch_actions.addWidget(self.clear)
        batch.addLayout(batch_actions)
        self.batch_start = button("Iniciar lote", self.start_batch, "primary")
        batch.addWidget(self.batch_start)
        self.reflow(1040)

        status = QFrame()
        status.setObjectName("card")
        status_layout = QVBoxLayout(status)
        status_layout.setContentsMargins(16, 10, 16, 10)
        status_layout.setSpacing(6)
        self.timestamps = QCheckBox("Gerar legendas SRT com marcação por palavra")
        status_layout.addWidget(self.timestamps)
        self.diarization = QCheckBox("Identificar falantes (diarização local)")
        self.diarization.setToolTip("Organiza os trechos por voz e baixa os modelos no primeiro uso.")
        status_layout.addWidget(self.diarization)
        self.speaker_settings = QWidget()
        speaker_row = QHBoxLayout(self.speaker_settings)
        speaker_row.setContentsMargins(0, 0, 0, 0)
        speaker_label = QLabel("&Número de falantes")
        self.speaker_count = QSpinBox()
        self.speaker_count.setRange(0, 99)
        self.speaker_count.setSpecialValueText("Automático")
        self.speaker_count.setToolTip("No lote, um número fixo vale para todos os arquivos.")
        speaker_label.setBuddy(self.speaker_count)
        speaker_row.addWidget(speaker_label)
        speaker_row.addWidget(self.speaker_count)
        speaker_row.addStretch()
        self.speaker_settings.hide()
        self.diarization.toggled.connect(self.speaker_settings.setVisible)
        status_layout.addWidget(self.speaker_settings)
        self.status = label("Pronto para transcrever", "muted")
        self.status.setWordWrap(True)
        status_layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setAccessibleName("Progresso da transcrição")
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(5)
        self.progress.reset()
        status_layout.addWidget(self.progress)
        layout.addWidget(status)

        result, result_layout, export = card("Resultado")
        self.save_text = button("Salvar TXT", lambda: self.export("txt"), "export")
        self.save_srt = button("Salvar SRT", lambda: self.export("srt"), "export")
        export.addWidget(self.save_text)
        export.addWidget(self.save_srt)
        self.output = QPlainTextEdit()
        self.output.setAccessibleName("Resultado da transcrição")
        self.output.setPlaceholderText("A transcrição e os resultados do lote aparecem aqui.")
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(140)
        result_layout.addWidget(self.output, 1)
        layout.addWidget(result, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "batch_panel"):
            self.reflow(event.size().width())

    def reflow(self, width):
        stacked = width < 790
        if stacked == self.stacked:
            return
        self.stacked = stacked
        for panel in (self.single_panel, self.batch_panel):
            self.panels.removeWidget(panel)
        self.panels.addWidget(self.single_panel, 0, 0)
        self.panels.addWidget(self.batch_panel, 1 if stacked else 0, 0 if stacked else 1)
        self.panels.setColumnStretch(0, 1)
        self.panels.setColumnStretch(1, 0 if stacked else 1)

    @Slot(str)
    def show_device(self, device):
        self.device_label.setText("GPU NVIDIA" if device == "cuda" else "CPU · modo alternativo")
        if self.initializing:
            self.initializing = False
            self.sync_buttons()

    def sync_buttons(self):
        idle = not self.busy
        for widget in (
            self.language, self.browse, self.url, self.add_files, self.add_folder,
            self.timestamps, self.diarization, self.speaker_count,
        ):
            widget.setEnabled(idle)
        ready = idle and not self.initializing
        self.use_link.setEnabled(idle and bool(self.url.text().strip()))
        self.transcribe.setEnabled(ready and self.source is not None)
        self.clear.setEnabled(idle and bool(self.paths))
        self.batch_start.setEnabled(ready and bool(self.paths))
        self.save_text.setEnabled(idle and self.result is not None)
        self.save_srt.setEnabled(idle and self.result is not None and bool(self.result.words))

    def choose_file(self):
        name, _ = QFileDialog.getOpenFileName(self, "Escolher mídia", filter=MEDIA_FILTER)
        if name:
            self.source = Path(name)
            self.source_label.setText(self.source.name)
            self.source_label.setToolTip(name)
            self.sync_buttons()

    def choose_link(self):
        if not self.use_link.isEnabled():
            return
        try:
            self.source = youtube_url(self.url.text())
        except ValueError as error:
            QMessageBox.warning(self, "Link inválido", str(error))
            return
        self.source_label.setText("YouTube · " + self.source.rsplit("=", 1)[-1])
        self.source_label.setToolTip(self.source)
        self.sync_buttons()

    def choose_files(self):
        names, _ = QFileDialog.getOpenFileNames(self, "Adicionar mídias", filter=MEDIA_FILTER)
        self.add_paths(map(Path, names))

    def choose_folder(self):
        name = QFileDialog.getExistingDirectory(self, "Adicionar mídias de uma pasta")
        if name:
            try:
                self.add_paths(sorted(path for path in Path(name).iterdir() if path.is_file()))
            except OSError as error:
                QMessageBox.warning(self, "Erro ao abrir pasta", str(error))

    def add_paths(self, paths):
        for path in paths:
            if path.suffix.lower() not in MEDIA_EXTENSIONS:
                continue
            key = path.resolve()
            if key not in self.paths:
                self.paths[key] = path
                item = QListWidgetItem(path.name)
                item.setToolTip(str(path))
                self.queue.addItem(item)
        self.queue_count.setText(f"{len(self.paths)} na fila" if self.paths else "Fila vazia")
        self.sync_buttons()

    def clear_queue(self):
        self.paths.clear()
        self.queue.clear()
        self.queue_count.setText("Fila vazia")
        self.sync_buttons()

    def start_single(self):
        self.run((self.source,), None)

    def start_batch(self):
        folder = QFileDialog.getExistingDirectory(self, "Escolher pasta para salvar o lote")
        if folder:
            self.run(tuple(self.paths.values()), Path(folder))

    def run(self, sources, folder):
        self.busy = True
        self.is_batch = folder is not None
        self.used_cpu_fallback = False
        self.result = None
        self.output.clear()
        self.show_status("Preparando transcrição…")
        self.sync_buttons()
        self.start.emit(Job(
            sources, folder, self.language.currentData(),
            self.timestamps.isChecked(), self.diarization.isChecked(), self.speaker_count.value(),
        ))

    @Slot(str)
    def show_status(self, message):
        self.status.setText(message)
        self.progress.setRange(0, 0)

    @Slot(str, int)
    def show_update(self, text, progress):
        if text:
            cursor = QTextCursor(self.output.document())
            cursor.movePosition(QTextCursor.MoveOperation.End)
            if not self.output.document().isEmpty():
                cursor.insertText("\n")
            cursor.insertText(text)
        self.progress.setRange(0, 100)
        self.progress.setValue(progress)

    @Slot()
    def reset_progress(self):
        self.used_cpu_fallback = True
        if not self.is_batch:
            self.output.clear()

    @Slot(object, str)
    def finish(self, result, detail):
        self.busy = False
        self.result = result
        self.result_name = detail if isinstance(result, Transcript) else ""
        if result is not None:
            self.output.setPlainText(result.text or "Nenhuma fala detectada.")
            detail = "✓ Transcrição concluída"
            if self.used_cpu_fallback:
                detail += " · CPU após falha da GPU"
            self.status.setText(detail)
        else:
            self.status.setText(detail)
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if self.is_batch or result is not None else 0)
        self.sync_buttons()

    def export(self, extension):
        content = self.result.text if extension == "txt" else make_srt(self.result.words)
        name, _ = QFileDialog.getSaveFileName(
            self, "Salvar transcrição", f"{self.result_name}.{extension}",
            f"Arquivo {extension.upper()} (*.{extension})",
        )
        if not name:
            return
        try:
            save_export(Path(name), content, replace=True)
        except OSError as error:
            QMessageBox.warning(self, "Erro ao salvar arquivo", str(error))
        else:
            self.status.setText(f"Arquivo salvo · {Path(name).name}")

    def closeEvent(self, event):
        if self.busy:
            QMessageBox.information(self, "Transcrição em andamento", "Aguarde a tarefa terminar para fechar.")
            event.ignore()
            return
        self.thread.quit()
        self.thread.wait()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    palette = QPalette()
    for role, color in {
        "Window": "#101522", "WindowText": "#edf2ff", "Base": "#121a2a",
        "AlternateBase": "#1b2334", "Text": "#edf2ff", "Button": "#303a50",
        "ButtonText": "#edf2ff", "Highlight": "#5549a0", "HighlightedText": "#ffffff",
        "PlaceholderText": "#9aa7c0", "Light": "#596780", "Dark": "#101522",
    }.items():
        palette.setColor(getattr(QPalette.ColorRole, role), QColor(color))
    app.setPalette(palette)
    app.setStyleSheet(Path(__file__).with_name("theme.qss").read_text(encoding="utf-8"))
    window = Window()
    window.show()
    return app.exec()


if __name__ == "__main__":
    diagnostics = configure_diagnostics()
    raise SystemExit(main())
