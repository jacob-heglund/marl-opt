import argparse

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-v', '--value', default=0.9, type=float, help="Value to print")
args = parser.parse_args()

print(f"hello {args.value}")

