import cv2
import numpy as np
import os

# ======================== CONFIGURATION ========================
FRAME_FOLDER = "frames"
TARGET_FRAME_INDEX = 100
SECRET_DATA = "Hello World"  # Hidden message
BLOCK_SIZE = 8
BIT_PLANES = [2, 3]
COLOR_CHANNEL = 0 
POSITION_FILE = "positions.txt"  # Position file 

# ======================== MAIN FUNCTION ========================
def embed_secret():
    # Embedded into frames
    print(f"[1/1] Embedding the secret into frame {TARGET_FRAME_INDEX}...")
    frame_paths = sorted([f for f in os.listdir(FRAME_FOLDER) if f.endswith('.png')])
    target_frame_path = os.path.join(FRAME_FOLDER, frame_paths[TARGET_FRAME_INDEX])
    
    # hide message in frame
    embed_data_to_frame(target_frame_path, BLOCK_SIZE, SECRET_DATA.encode('utf-8'), BIT_PLANES, COLOR_CHANNEL)
    
    print(f"Secret embedded successfully in {target_frame_path}")
    print(f"Positions saved to {POSITION_FILE}")

# ======================== HELPER FUNCTIONS ========================
def embed_data_to_frame(frame_path, block_size, secret_data, bit_planes, color_channel):
    """Nhung du lieu vao frame chon"""
    frame = cv2.imread(frame_path)
    binary_data = ''.join(format(byte, '08b') for byte in secret_data)
    data_index = 0
    positions = []
    
    for bit in bit_planes:
        channel_data = frame[:, :, color_channel].astype(np.int16)
        bit_plane = (channel_data >> bit) & 1
        height, width = bit_plane.shape
        
        for i in range(0, height - block_size, block_size):
            for j in range(0, width - block_size, block_size):
                block = bit_plane[i:i+block_size, j:j+block_size]
                complexity = np.sum(block) / block.size
                
                if 0.3 < complexity < 0.7 and data_index < len(binary_data):
                    center = block_size // 2
                    block[center, center] = int(binary_data[data_index])
                    positions.append((i + center, j + center, bit))  # Lưu vi tri bitplanes
                    data_index += 1
        
        # Cap nhat kenh mau
        updated_channel = ((channel_data & ~(1 << bit)) | (bit_plane << bit)).clip(0, 255)
        frame[:, :, color_channel] = updated_channel.astype(np.uint8)
    
    # Luu frame da chinh sua
    cv2.imwrite(frame_path, frame)
    
    # Luu vi tri da giau 
    with open(POSITION_FILE, 'w') as f:
        for pos in positions:
            f.write(f"{pos[0]} {pos[1]} {pos[2]}\n")  # Ghi x, y va bit plane

if __name__ == "__main__":
    embed_secret()
