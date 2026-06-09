#!/usr/bin/env python3
import struct, zlib, sys

with open(sys.argv[1], 'rb') as f:
    data = f.read()
print(f'File: {len(data)} bytes')

pos = 8
while pos < len(data):
    chunk_len = struct.unpack('>I', data[pos:pos+4])[0]
    chunk_type = data[pos+4:pos+8].decode('ascii')
    chunk_data = data[pos+8:pos+8+chunk_len]
    stored_crc = struct.unpack('>I', data[pos+8+chunk_len:pos+12+chunk_len])[0]
    correct_crc = zlib.crc32(data[pos+4:pos+8+chunk_len]) & 0xffffffff
    ok = "OK" if stored_crc == correct_crc else "BAD"
    print(f'{chunk_type}: len={chunk_len} CRC={ok}')
    if not ok:
        print(f'  stored={stored_crc:08x} correct={correct_crc:08x}')
    pos += 12 + chunk_len
