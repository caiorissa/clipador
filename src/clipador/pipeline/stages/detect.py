"""Day 2: score windows from audio energy, transcript and (optional) chat peaks."""

from clipador.pipeline.base import Stage, StageContext


class DetectStage(Stage):
    name = "detect"
    needs_gpu = False

    def run(self, ctx: StageContext) -> None:
        raise NotImplementedError("detecção de momentos chega no dia 2")
