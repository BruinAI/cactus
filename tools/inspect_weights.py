
import struct
import sys
import os

def inspect_weight_file(filename):
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        return

    with open(filename, 'rb') as f:
        header = f.read(80)
        
        offset = 0
        magic = struct.unpack_from('<I', header, offset)[0]
        offset += 4
        
        if magic != 0x54434143:
            print(f"Invalid magic number: {hex(magic)}")
            return

        version = struct.unpack_from('<I', header, offset)[0]
        offset += 4
        
        flags = struct.unpack_from('<I', header, offset)[0]
        offset += 4
        
        alignment = struct.unpack_from('<I', header, offset)[0]
        offset += 4
        
        ndim = struct.unpack_from('<I', header, offset)[0]
        offset += 4
        
        print(f"File: {filename}")
        print(f"Magic: {hex(magic)}")
        print(f"Version: {version}")
        print(f"Alignment: {alignment}")
        print(f"Dimensions: {ndim}")
        
        dims = []
        for i in range(4):
            dim = struct.unpack_from('<Q', header, offset)[0]
            offset += 8
            if i < ndim:
                dims.append(dim)
        
        print(f"Shape: {dims}")
        
        precision = struct.unpack_from('<I', header, offset)[0]
        offset += 4
        print(f"Precision: {precision}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_weights.py <filename>")
    else:
        inspect_weight_file(sys.argv[1])
