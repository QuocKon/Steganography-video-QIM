import sys

def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        content1 = f1.read().strip()
        content2 = f2.read().strip()

        if content1 == content2:
            print("✅ The two files are identical.")
        else:
            print("❌ The two files are different.")
            # Optional: show where they differ
            for i, (a, b) in enumerate(zip(content1, content2)):
                if a != b:
                    print(f"Difference at position {i}: '{a}' vs '{b}'")
                    break
            if len(content1) != len(content2):
                print(f"Files have different lengths: {len(content1)} vs {len(content2)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_files.py <file1.txt> <file2.txt>")
        sys.exit(1)

    compare_files(sys.argv[1], sys.argv[2])

