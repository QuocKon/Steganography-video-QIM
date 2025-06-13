import sys
import numpy as np
import os

def message_to_bits_from_file(input_file, output_file):
    """
    Read message from input file, convert to binary bit string (with repetition),
    and save to a single line in output file.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        message = f.read()
    
    msg_orig = np.array([int(b) for b in ''.join(format(ord(c), '08b') for c in message)])
    msg = np.repeat(msg_orig, 3)
    bit_string = ''.join(map(str, msg))
    
    with open(output_file, 'w') as f:
        f.write(bit_string)
    
    print(f"File '{output_file}' has been created containing the encoded bit string.")

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 message_to_bits.py <input_message_file.txt> <output_bit_file.txt>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"❌ Input file '{input_file}' does not exist.")
        sys.exit(1)

    if os.path.exists(output_file):
        raise FileExistsError(f"❌ Output file '{output_file}' already exists. Please choose a different name.")

    message_to_bits_from_file(input_file, output_file)

if __name__ == "__main__":
    main()

