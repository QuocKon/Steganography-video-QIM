import sys
import numpy as np
import cv2
from scipy.fftpack import dct
import os

class QIM:
    def __init__(self, delta):
        self.delta = delta

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
    return dct_y

def extract_bits_from_video(input_video, bit_count, delta):
    qim = QIM(delta)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError("Cannot open input video.")

    extracted_bits = []
    while cap.isOpened() and len(extracted_bits) < bit_count:
        ret, frame = cap.read()
        if not ret:
            break

        dct_y = apply_dct(frame)
        coeffs = dct_y[2:42, 2:42].flatten()

        detected = qim.detect(coeffs)
        bits_needed = bit_count - len(extracted_bits)
        extracted_bits.extend(detected[:bits_needed])

    cap.release()
    return ''.join(str(bit) for bit in extracted_bits)

def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_bits_from_video.py <input_video> <output_txt_file>")
        sys.exit(1)

    input_video = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(input_video):
        raise FileNotFoundError(f"File '{input_video}' not found.")

    # 👇 Yêu cầu người dùng nhập thông tin thủ công
    try:
        delta = int(input("Enter delta (e.g., 250): "))
        message_length = int(input("Enter message length (number of characters): "))
        repeat = int(input("Enter number of repeat rounds: "))
    except ValueError:
        print("[✗] Invalid input. Please enter integer values.")
        sys.exit(1)

    bit_count = message_length * 8 * repeat
    print(f"[*] Extracting {bit_count} bits with delta={delta}...")

    bitstring = extract_bits_from_video(input_video, bit_count, delta)

    with open(output_file, 'w') as f:
        f.write(bitstring)

    print(f"[✓] Bit string saved to: {output_file}")

if __name__ == "__main__":
    main()
