"""Day 2: GPU transcription with faster-whisper (word timestamps)."""

from clipador.pipeline.base import Stage, StageContext


class TranscribeStage(Stage):
    name = "transcribe"
    needs_gpu = True

    def run(self, ctx: StageContext) -> None:
        raise NotImplementedError("transcrição chega no dia 2")
