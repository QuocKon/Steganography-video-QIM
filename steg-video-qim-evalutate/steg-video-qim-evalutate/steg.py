import numpy as np
import cv2
from scipy.fftpack import dct, idct
import ffmpeg
import os
import tempfile
import sys
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_video_quality(original_video, processed_video):
    cap1 = cv2.VideoCapture(original_video)
    cap2 = cv2.VideoCapture(processed_video)

    psnr_total = 0
    ssim_total = 0
    frame_count = 0

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        frame_count += 1
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        psnr_val = calculate_psnr(gray1, gray2)
        ssim_val = ssim(gray1, gray2)

        psnr_total += psnr_val
        ssim_total += ssim_val

    cap1.release()
    cap2.release()

    if frame_count == 0:
        return None, None
    return psnr_total / frame_count, ssim_total / frame_count

class QIM:
    def __init__(self, delta):
        self.delta = delta

    def embed(self, x, m):
        x = x.astype(float)
        d = self.delta
        y = np.round(x / d) * d + (-1)**(m + 1) * d / 4.0
        return y

    def detect(self, z):
        z = z.flatten().astype(float)
        d = self.delta
        z0 = np.round(z / d) * d - d / 4.0
        z1 = np.round(z / d) * d + d / 4.0
        d0 = np.abs(z - z0)
        d1 = np.abs(z - z1)
        m_detected = (d1 <= d0).astype(int)
        return m_detected

def apply_dct(frame):
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0].astype(float)
    dct_y = dct(dct(y.T, norm='ortho').T, norm='ortho')
    return dct_y, yuv

def apply_idct(dct_y, yuv):
    y = idct(idct(dct_y.T, norm='ortho').T, norm='ortho')
    yuv[:, :, 0] = np.clip(y, 0, 255).astype(np.uint8)
    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return frame

def embed_message_to_video(input_video, output_video, message, delta=200,
                           codec='mp4v', container_format='mp4',
                           ffmpeg_vcodec='libx264', ffmpeg_crf=23,
                           x265_lossless_val=None):
    qim = QIM(delta)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError("Cannot open input video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    msg_orig = np.array([int(b) for b in ''.join(format(ord(c), '08b') for c in message)])
    msg = np.repeat(msg_orig, 3)
    msg_len = len(msg)

    coeffs_per_frame = 40 * 40
    total_capacity = frame_count * coeffs_per_frame
    if total_capacity < msg_len:
        raise ValueError(f"Video too short! Need at least {msg_len / coeffs_per_frame:.2f} frames but only have {frame_count}.")

    temp_file = tempfile.NamedTemporaryFile(suffix='.' + container_format, delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(temp_file, fourcc, fps, (width, height))

    msg_idx = 0
    frame_idx = 0
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
        frame_idx += 1

    cap.release()
    out.release()

    try:
        input_stream = ffmpeg.input(input_video)
        video_stream = ffmpeg.input(temp_file).video
        audio_stream = input_stream.audio

        ext = os.path.splitext(output_video)[1].lower()
        if ext == '.mkv':
            container_out = 'matroska'
        elif ext in ['.mp4', '.avi', '.mov']:
            container_out = ext[1:]
        else:
            container_out = None

        kwargs = {
            'vcodec': ffmpeg_vcodec,
            'acodec': 'copy',
            'crf': ffmpeg_crf,
            'strict': 'experimental'
        }
        if container_out is not None:
            kwargs['format'] = container_out

        # Nếu encode H.265 và truyền tham số lossless thì thêm x265-params
        if ffmpeg_vcodec == 'libx265' and x265_lossless_val is not None:
            kwargs['x265-params'] = f'lossless={x265_lossless_val}'

        output_stream = ffmpeg.output(
            video_stream, audio_stream, output_video,
            **kwargs
        )
        ffmpeg.run(output_stream)
        os.remove(temp_file)
    except ffmpeg.Error as e:
        if e.stderr:
            print(f"Error merging audio: {e.stderr.decode()}")
        else:
            print(f"Error merging audio: {str(e)}")
        raise

    print(f"Message embedded into {output_video} with codec={codec}, ffmpeg_vcodec={ffmpeg_vcodec}, crf={ffmpeg_crf}, x265_lossless={x265_lossless_val}")

def extract_message_from_video(input_video, msg_len, delta=200):
    qim = QIM(delta)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError("Cannot open input video.")

    extracted_msg = []
    bits_needed = msg_len * 3
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or len(extracted_msg) >= bits_needed:
            break

        dct_y, _ = apply_dct(frame)
        coeffs = dct_y[2:42, 2:42].flatten()
        m_detected = qim.detect(coeffs)
        extracted_msg.extend(m_detected[:min(len(m_detected), bits_needed - len(extracted_msg))])
        frame_idx += 1

    cap.release()

    extracted_msg = extracted_msg[:bits_needed]
    extracted_msg_orig = []
    for i in range(0, len(extracted_msg), 3):
        bits = extracted_msg[i:i+3]
        if len(bits) >= 3:
            extracted_msg_orig.append(1 if sum(bits) >= 2 else 0)
        else:
            extracted_msg_orig.append(bits[0] if bits else 0)

    msg_str = ''
    for i in range(0, len(extracted_msg_orig), 8):
        byte = extracted_msg_orig[i:i+8]
        if len(byte) == 8:
            try:
                msg_str += chr(int(''.join(map(str, byte)), 2))
            except ValueError:
                msg_str += '?'
    return msg_str

def quality_description(psnr, ssim_val):
    if psnr is None or ssim_val is None:
        return "Cannot evaluate quality (no frames found)."
    
    if psnr > 40:
        psnr_desc = "Excellent (almost indistinguishable from original video)"
    elif psnr > 30:
        psnr_desc = "Good (minor differences, hard to notice)"
    else:
        psnr_desc = "Degraded quality (noticeable differences)"
    
    if ssim_val > 0.95:
        ssim_desc = "Excellent (structurally almost identical)"
    elif ssim_val > 0.8:
        ssim_desc = "Fair (only slight structural differences)"
    else:
        ssim_desc = "Significant degradation (clear structural differences)"
    
    return f"PSNR: {psnr_desc}; SSIM: {ssim_desc}"

def main():
    input_video = "example.mp4"
    message = "Hello World"
    delta = 250

    if not os.path.exists(input_video):
        print(f"File {input_video} does not exist.")
        sys.exit(1)

    print("Select output video codec option:")
    print("1: FFV1 (lossless)")
    print("2: H.265 (HEVC) - set lossless parameter (0-3)")
    print("3: H.264 - enter CRF value")

    choice = input("Enter option (1/2/3): ").strip()

    if choice == '1':
        codec = 'FFV1'
        container = 'avi'
        ffmpeg_vcodec = 'ffv1'
        ffmpeg_crf = 0
        x265_lossless_val = None
        output_file = f'output1_ffv1.{container}'

    elif choice == '2':
        codec = 'XVID'  
        container = 'mp4'  
        ffmpeg_vcodec = 'libx265'
        try:
            x265_lossless_val = int(input("Enter lossless parameter for H.265 (0=lossy, 1,2,3=lossless variants): ").strip())
            if x265_lossless_val not in [0,1,2,3]:
                print("Invalid value, defaulting to 0 (lossy).")
                x265_lossless_val = 0
        except:
            print("Invalid input, defaulting to 0 (lossy).")
            x265_lossless_val = 0
        ffmpeg_crf = 0 if x265_lossless_val != 0 else 28
        output_file = f'output2_h265_lossless{x265_lossless_val}.{container}'

    elif choice == '3':
        codec = 'mp4v'
        container = 'mp4'
        ffmpeg_vcodec = 'libx264'
        x265_lossless_val = None
        try:
            ffmpeg_crf = int(input("Enter CRF value for H.264 (0-51): ").strip())
            if not (0 <= ffmpeg_crf <= 51):
                raise ValueError
        except ValueError:
            print("Invalid CRF, defaulting to 23.")
            ffmpeg_crf = 23
        output_file = f'output3_h264_crf{ffmpeg_crf}.{container}'

    else:
        print("Invalid option.")
        sys.exit(1)

    print(f"\nEmbedding message with codec={codec}, ffmpeg_vcodec={ffmpeg_vcodec}, crf={ffmpeg_crf}, x265_lossless={x265_lossless_val} ...")
    embed_message_to_video(input_video, output_file, message, delta,
                           codec=codec,
                           container_format=container,
                           ffmpeg_vcodec=ffmpeg_vcodec,
                           ffmpeg_crf=ffmpeg_crf,
                           x265_lossless_val=x265_lossless_val)

    msg_len = len(message) * 8
    extracted = extract_message_from_video(output_file, msg_len, delta)
    print(f"Extracted message: {extracted}")

    psnr_val, ssim_val = calculate_video_quality(input_video, output_file)
    print("Video quality after embedding and encoding:")
    print(f"Average PSNR: {psnr_val:.2f} dB")
    print(f"Average SSIM: {ssim_val:.4f}")
    print(quality_description(psnr_val, ssim_val))
    


if __name__ == "__main__":
    main()

