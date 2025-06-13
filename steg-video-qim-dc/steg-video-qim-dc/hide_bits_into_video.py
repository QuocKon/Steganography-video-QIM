import sys
import numpy as np
import cv2
from scipy.fftpack import dct, idct
import ffmpeg
import os
import tempfile

class QIM:
    def __init__(self, delta):
        self.delta = delta

    def embed(self, x, m):
        """
        Embed binary message m into vector x using QIM.
        """
        x = x.astype(float)
        d = self.delta
        y = np.round(x / d) * d + (-1)**(m + 1) * d / 4.0
        return y

def apply_dct(frame):
    """
    Apply 2D DCT to the Y channel of the frame.
    """
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0].astype(float)
    dct_y = dct(dct(y.T, norm='ortho').T, norm='ortho')
    return dct_y, yuv

def apply_idct(dct_y, yuv):
    """
    Apply inverse DCT to reconstruct the frame.
    """
    y = idct(idct(dct_y.T, norm='ortho').T, norm='ortho')
    yuv[:, :, 0] = np.clip(y, 0, 255).astype(np.uint8)
    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return frame

def embed_message_to_video(input_video, output_video, bit_file, delta=200):
    """
    Embed binary message from bit file into video, preserving audio.
    """
    qim = QIM(delta)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError("Cannot open input video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with open(bit_file, 'r') as f:
        bit_string = f.read().strip()
    if not bit_string:
        raise ValueError(f"File {bit_file} is empty or contains invalid data.")
    if not all(bit in '01' for bit in bit_string):
        raise ValueError(f"File {bit_file} contains invalid characters. Only 0 and 1 are allowed.")
    
    msg = np.array([int(bit) for bit in bit_string], dtype=int)
    msg_len = len(msg)

    coeffs_per_frame = 40 * 40
    total_capacity = frame_count * coeffs_per_frame
    if total_capacity < msg_len:
        raise ValueError(f"Video too short! Required at least {msg_len / coeffs_per_frame:.2f} frames, but only {frame_count} available.")

    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_file, fourcc, fps, (width, height))

    msg_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        dct_y, yuv = apply_dct(frame)
        coeffs = dct_y[2:42, 2:42].flatten()
        coeffs_len = len(coeffs)

        if msg_idx < msg_len:
            embed_len = min(coeffs_len, msg_len - msg_idx)
            m = msg[msg_idx:msg_idx + embed_len]
            if len(m) < coeffs_len:
                m = np.pad(m, (0, coeffs_len - len(m)), mode='constant')
            coeffs_embedded = qim.embed(coeffs, m)
            dct_y[2:42, 2:42] = coeffs_embedded.reshape(40, 40)
            msg_idx += embed_len

        frame_out = apply_idct(dct_y, yuv)
        out.write(frame_out)

    cap.release()
    out.release()

    try:
        input_stream = ffmpeg.input(input_video)
        video_stream = ffmpeg.input(temp_file).video
        audio_stream = input_stream.audio
        output_stream = ffmpeg.output(
            video_stream, audio_stream, output_video,
            vcodec='libx264', acodec='aac',
            video_bitrate='5000k', crf=8, strict='experimental'
        )
        ffmpeg.run(output_stream)
        os.remove(temp_file)
    except ffmpeg.Error as e:
        print(f"Error during audio merging: {e.stderr.decode()}")
        raise

    print(f"Message has been embedded into {output_video}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 hide_bits_into_video.py <input_video> <output_video> <bit_file.txt>")
        sys.exit(1)

    input_video = sys.argv[1]
    output_video = sys.argv[2]
    bit_file = sys.argv[3]

    if not os.path.exists(input_video):
        raise FileNotFoundError(f"File {input_video} does not exist.")
    if not os.path.exists(bit_file):
        raise FileNotFoundError(f"File {bit_file} does not exist.")

    embed_message_to_video(input_video, output_video, bit_file, delta=250)

if __name__ == "__main__":
    main()
