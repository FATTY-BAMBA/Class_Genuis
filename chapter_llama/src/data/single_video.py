from pathlib import Path
import json
import codecs

from chapter_llama.src.data.chapters import sec_to_hms
from chapter_llama.tools.extract.asr_faster_whisper import ASRProcessor


class SingleVideo:
    """
    A simplified implementation of the src.data.chapters.Chapters interface for single video inference.

    Mimics ChaptersASR but for a single video file. Provides the methods required by PromptASR
    to generate chapter timestamps and titles. Inference-only.
    """

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.video_ids = [video_path.stem]
        assert video_path.exists(), f"Video file {video_path} not found"
        self.asr, self.duration = get_asr(video_path, overwrite=True)

    def __len__(self):
        return len(self.video_ids)

    def __iter__(self):
        return iter(self.video_ids)

    def __contains__(self, vid_id):
        return vid_id in self.video_ids

    def get_duration(self, vid_id, hms=False):
        assert vid_id == self.video_ids[0], f"Invalid video ID: {vid_id}"
        if hms:
            return sec_to_hms(self.duration)
        return self.duration

    def get_asr(self, vid_id):
        assert vid_id == self.video_ids[0], f"Invalid video ID: {vid_id}"
        return self.asr


def get_asr(video_path: Path, overwrite=False):
    output_dir = Path(f"outputs/inference/{video_path.stem}")
    asr_output = output_dir / "asr.txt"
    duration_output = output_dir / "duration.txt"
    metrics_output = output_dir / "asr_metrics.json"

    if asr_output.exists() and duration_output.exists() and not overwrite:
        # Read ASR lines
        asr = asr_output.read_text(encoding="utf-8").splitlines()
        asr = "\n".join(asr) + "\n"

        # Read duration (expect one line)
        duration_lines = duration_output.read_text(encoding="utf-8").splitlines()
        assert len(duration_lines) == 1, f"Duration is not a list of length 1: {duration_lines}"
        duration = float(duration_lines[0])
        assert duration > 0, f"Duration is not positive: {duration}"
        return asr, duration

    print(f"\n=== 🎙️ Processing ASR for {video_path} ===")
    asr_processor = ASRProcessor()
    asr, duration = asr_processor.get_asr(video_path)
    print(f"=== ✅ ASR processing complete for {video_path} ===\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    with codecs.open(asr_output, 'w', encoding='utf-8') as f:
        f.write(asr)

    duration_output.write_text(str(duration), encoding="utf-8")

    # Persist coverage metrics if the processor provided them
    metrics = getattr(asr_processor, "last_metrics", None)
    if metrics:
        try:
            with open(metrics_output, "w", encoding="utf-8") as mf:
                json.dump(metrics, mf, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return asr, duration
