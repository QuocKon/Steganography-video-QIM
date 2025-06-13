import sys
import numpy as np
import os

def message_to_bits(message, output_file):
    """
    Convert message to binary bit string and save to a single line in file.
    """
    msg_orig = np.array([int(b) for b in ''.join(format(ord(c), '08b') for c in message)])
    msg = np.repeat(msg_orig, 3)
    bit_string = ''.join(map(str, msg))
    with open(output_file, 'w') as f:
        f.write(bit_string)
    print(f"File {output_file} has been created containing binary bit string.")

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 message_to_bits.py <message> <output_file.txt>")
        sys.exit(1)

    message = sys.argv[1]
    output_file = sys.argv[2]

    if os.path.exists(output_file):
        raise FileExistsError(f"File {output_file} already exists. Please choose a different name.")

    message_to_bits(message, output_file)

if __name__ == "__main__":
    main()
