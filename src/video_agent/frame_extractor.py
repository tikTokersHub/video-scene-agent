from pathlib import Path

import cv2

def extract_frames(
        video_path:Path,
        output_dir:Path,
        sample_every_n: int = 8,
) -> list[Path]:
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found on {video_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Video cannot be opened {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    saved_paths: list[Path] = []
    frame_idx = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
         
        if frame_idx % sample_every_n == 0:
            frame_path = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)

            saved_paths.append(frame_path)
            saved_count += 1

        frame_idx += 1

    cap.release()
    
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Saved frames: {saved_count}")
    print(f"Output dir: {output_dir}")

    return saved_paths