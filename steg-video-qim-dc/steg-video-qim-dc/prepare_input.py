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
    print(f"[✓] File {output_file} has been created containing binary bit string.")
    return bit_string

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 message_to_bits.py <message.txt>")
        sys.exit(1)

    message_file = sys.argv[1]
    bits_output_file = "secret_bits.txt"
    metadata_output_file = "metadata.txt"

    # Configuration (matching prepare_message.py)
    delta = 250
    repeat = 3
    coeff_block = "[2:42, 2:42]"

    try:
        # Check if output files already exist
        if os.path.exists(bits_output_file):
            raise FileExistsError(f"File {bits_output_file} already exists. Please delete or rename it.")
        if os.path.exists(metadata_output_file):
            raise FileExistsError(f"File {metadata_output_file} already exists. Please delete or rename it.")

        # Read message
        with open(message_file, "r", encoding="utf-8") as f:
            message = f.read().strip()

        # Validate message
        message_length = len(message)
        if message_length == 0:
            raise ValueError("Message is empty.")
        for char in message:
            if ord(char) > 127:
                raise ValueError(f"Non-ASCII character '{char}' found. Only ASCII characters are supported.")

        # Convert message to bits and save to secret_bits.txt
        bit_string = message_to_bits(message, bits_output_file)
        bit_count = len(bit_string)

        # Write metadata to file
        with open(metadata_output_file, "w") as f:
            f.write(f"""# Metadata for extraction
delta={delta}
message_length={message_length}
repeat={repeat}
bit_count={bit_count}
coeff_block={coeff_block}
""")

        print(f"[✓] Message read from '{message_file}'")
        print(f"[✓] Generated '{bits_output_file}' with {bit_count} bits")
        print(f"[✓] Created '{metadata_output_file}' with metadata")

    except FileNotFoundError:
        print(f"[✗] File '{message_file}' not found.")
    except FileExistsError as e:
        print(f"[✗] Error: {str(e)}")
    except ValueError as e:
        print(f"[✗] Error: {str(e)}")
    except Exception as e:
        print(f"[✗] Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
