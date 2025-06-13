import sys

def majority_vote(bits):
    """
    Perform majority voting for each group of 3 bits.
    """
    result = []
    for i in range(0, len(bits), 3):
        group = bits[i:i+3]
        if len(group) < 3:
            result.append(group[0])
        else:
            result.append('1' if group.count('1') >= 2 else '0')
    return ''.join(result)

def bits_to_message(bitstring):
    """
    Convert a bit string to a message after majority voting.
    """
    message = ''
    for i in range(0, len(bitstring), 8):
        byte = bitstring[i:i+8]
        if len(byte) == 8:
            try:
                message += chr(int(byte, 2))
            except ValueError:
                message += '?'
    return message

def main():
    if len(sys.argv) < 2:
        print("Usage: python bits_to_message.py <bitfile.txt>")
        sys.exit(1)

    input_file = sys.argv[1]

    with open(input_file, 'r') as f:
        bitstring = f.read().strip()

    voted_bits = majority_vote(bitstring)
    message = bits_to_message(voted_bits)

    print(f"Extracted message: {message}")

if __name__ == "__main__":
    main()
