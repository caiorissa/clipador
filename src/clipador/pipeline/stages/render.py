"""Day 4: render 9:16 clips (blur / facecam layouts, ASS word-by-word captions, NVENC)."""

from clipador.pipeline.base import Stage, StageContext


class RenderStage(Stage):
    name = "render"
    needs_gpu = True  # NVENC uses GPU memory; never alongside Whisper/Ollama

    def run(self, ctx: StageContext) -> None:
        raise NotImplementedError("renderização chega no dia 4")
