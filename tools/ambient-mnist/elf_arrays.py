#!/usr/bin/env python3
"""Read the weight / input arrays straight out of the compiled firmware
(apps/my_mnist/bld/gcc/my_mnist.out, an MSP430 ELF) so we can see exactly
which numbers the built binary carries. Pure Python, no tools needed.

Usage: python3 elf_arrays.py path/to/my_mnist.out
"""
import struct, sys

NAMES = ["input", "conv1_w", "conv1_b", "conv2_w", "conv2_b", "fc1_w", "fc1_b", "fc2_w", "fc2_b"]

def symbol_sizes(path, names):
    """Size in bytes of each named symbol (like `nm -S`)."""
    b = open(path, "rb").read()
    e_shoff, = struct.unpack_from("<I", b, 0x20)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", b, 0x2E)
    secs = [struct.unpack_from("<10I", b, e_shoff + i * e_shentsize) for i in range(e_shnum)]
    symtab = next(s for s in secs if s[1] == 2)
    strtab = secs[symtab[6]]
    out = {}
    for i in range(symtab[5] // 16):
        st_name, st_value, st_size, *_ = struct.unpack_from("<IIIBBH", b, symtab[4] + i * 16)
        end = b.index(b"\0", strtab[4] + st_name)
        n = b[strtab[4] + st_name:end].decode()
        if n in names: out[n] = st_size
    shstr = secs[e_shstrndx]
    for s in secs:
        end = b.index(b"\0", shstr[4] + s[0])
        n = b[shstr[4] + s[0]:end].decode()
        if n in names: out[n] = s[5]
    return out

def read_elf_arrays(path, names=NAMES):
    b = open(path, "rb").read()
    assert b[:4] == b"\x7fELF" and b[4] == 1 and b[5] == 1   # 32-bit little endian
    e_shoff, = struct.unpack_from("<I", b, 0x20)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", b, 0x2E)
    secs = []
    for i in range(e_shnum):
        o = e_shoff + i * e_shentsize
        name, typ, flags, addr, off, size, link, info, align, entsize = struct.unpack_from("<10I", b, o)
        secs.append(dict(name=name, type=typ, addr=addr, off=off, size=size, link=link, entsize=entsize))
    symtab = next(s for s in secs if s["type"] == 2)
    strtab = secs[symtab["link"]]
    def cstr(off):
        end = b.index(b"\0", off); return b[off:end].decode()
    want = {}
    for i in range(symtab["size"] // 16):
        o = symtab["off"] + i * 16
        st_name, st_value, st_size, st_info, st_other, st_shndx = struct.unpack_from("<IIIBBH", b, o)
        n = cstr(strtab["off"] + st_name)
        if n in names and st_size:
            sec = secs[st_shndx]
            fo = sec["off"] + (st_value - sec["addr"])
            want[n] = list(struct.unpack_from("<%dh" % (st_size // 2), b, fo))
    return want

if __name__ == "__main__":
    arrs = read_elf_arrays(sys.argv[1])
    for k, v in arrs.items():
        print(k, len(v), v[:12])
