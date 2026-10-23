"""Minimal read-only decoder for this project's flat, Snappy Parquet tables.

Supports PLAIN dictionary pages and PLAIN/RLE_DICTIONARY data pages only.
Unsupported representations raise rather than silently returning data.
"""
import struct
from pathlib import Path


class Compact:
    def __init__(self, data, offset=0):
        self.data, self.pos = data, offset

    def byte(self):
        out = self.data[self.pos]
        self.pos += 1
        return out

    def varint(self):
        out = shift = 0
        while True:
            b = self.byte()
            out |= (b & 127) << shift
            if b < 128:
                return out
            shift += 7

    def integer(self):
        u = self.varint()
        return (u >> 1) ^ -(u & 1)

    def value(self, kind):
        if kind == 1: return True
        if kind == 2: return False
        if kind == 3: return self.byte()
        if kind in (4, 5, 6): return self.integer()
        if kind == 7:
            value = struct.unpack_from('<d', self.data, self.pos)[0]
            self.pos += 8
            return value
        if kind == 8:
            n = self.varint(); out = self.data[self.pos:self.pos+n]
            self.pos += n
            return out
        if kind in (9, 10):
            h = self.byte(); n, t = h >> 4, h & 15
            if n == 15: n = self.varint()
            return [self.value(t) for _ in range(n)]
        if kind == 11:
            n = self.varint()
            if not n: return {}
            types = self.byte()
            return {self.value(types >> 4): self.value(types & 15) for _ in range(n)}
        if kind == 12: return self.structure()
        raise ValueError(('unsupported compact type', kind))

    def structure(self):
        fields, previous = {}, 0
        while True:
            h = self.byte()
            if h == 0: return fields
            delta, kind = h >> 4, h & 15
            key = previous + delta if delta else self.integer()
            fields[key] = self.value(kind)
            previous = key


def snappy(data):
    r = Compact(data); expected = r.varint(); out = bytearray()
    while r.pos < len(data):
        h = r.byte(); kind = h & 3
        if kind == 0:
            n = h >> 2
            if n < 60: n += 1
            else:
                k = n - 59
                n = int.from_bytes(data[r.pos:r.pos+k], 'little') + 1
                r.pos += k
            out.extend(data[r.pos:r.pos+n]); r.pos += n
        else:
            if kind == 1:
                n = 4 + ((h >> 2) & 7)
                offset = ((h & 224) << 3) | r.byte()
            else:
                n = 1 + (h >> 2); k = 2 if kind == 2 else 4
                offset = int.from_bytes(data[r.pos:r.pos+k], 'little'); r.pos += k
            if not 0 < offset <= len(out): raise ValueError('invalid Snappy offset')
            for _ in range(n): out.append(out[-offset])
    if len(out) != expected: raise ValueError('Snappy length mismatch')
    return bytes(out)


def rle(data, width, count):
    r = Compact(data); values = []
    while len(values) < count:
        h = r.varint()
        if h & 1:
            n = (h >> 1) * 8; size = n * width // 8
            packed = int.from_bytes(data[r.pos:r.pos+size], 'little'); r.pos += size
            mask = (1 << width) - 1
            values.extend((packed >> (i * width)) & mask for i in range(n))
        else:
            n = h >> 1; size = (width + 7) // 8
            v = int.from_bytes(data[r.pos:r.pos+size], 'little'); r.pos += size
            values.extend([v] * n)
    return values[:count], r.pos


def plain(data, physical, n):
    if physical == 6:
        values = []; pos = 0
        for _ in range(n):
            size = struct.unpack_from('<I', data, pos)[0]; pos += 4
            values.append(data[pos:pos+size].decode('utf-8')); pos += size
        return values
    formats = {1:'i', 2:'q', 4:'f', 5:'d'}
    if physical not in formats: raise ValueError(('unsupported plain physical', physical))
    return list(struct.unpack_from('<' + str(n) + formats[physical], data))


def read_columns(path, names):
    data = Path(path).read_bytes()
    assert data[:4] == data[-4:] == b'PAR1'
    footer_len = struct.unpack_from('<I', data, len(data)-8)[0]
    meta = Compact(data, len(data)-8-footer_len).structure()
    schema = {s[4].decode():s for s in meta[2][1:]}
    output = {name:[] for name in names}
    for rowgroup in meta[4]:
        for chunk in rowgroup[1]:
            cm = chunk[3]; name = cm[3][0].decode()
            if name not in output: continue
            assert len(cm[3]) == 1 and schema[name].get(3, 0) in (0, 1)
            physical, codec, total = cm[1], cm[4], cm[5]
            pos = min(cm[9], cm.get(11, cm[9])); values=[]; dictionary=None
            while len(values) < total:
                header = Compact(data, pos); ph = header.structure(); pos = header.pos
                payload = data[pos:pos+ph[3]]; pos += ph[3]
                if codec == 1: payload = snappy(payload)
                elif codec != 0: raise ValueError(('unsupported codec',codec))
                assert len(payload) == ph[2]
                if ph[1] == 2:
                    assert ph[7][2] == 0
                    dictionary = plain(payload, physical, ph[7][1]); continue
                if ph[1] != 0: raise ValueError(('unsupported page', ph[1]))
                dh=ph[5]; count=dh[1]; encoding=dh[2]; start=0
                if schema[name].get(3, 0) == 1:
                    assert dh[3] == 3
                    size=struct.unpack_from('<I',payload,0)[0]
                    defined,_=rle(payload[4:4+size],1,count);start=4+size
                else: defined=[1]*count
                n=sum(defined)
                if encoding in (2,8):
                    assert dictionary is not None
                    width=payload[start]
                    indexes,_=rle(payload[start+1:],width,n)
                    present=[dictionary[j] for j in indexes]
                elif encoding==0: present=plain(payload[start:],physical,n)
                else: raise ValueError(('unsupported encoding',encoding))
                it=iter(present);values.extend(next(it) if d else None for d in defined)
            assert len(values)==total==rowgroup[3]
            output[name].extend(values)
    for name, values in output.items(): assert len(values)==meta[3], (name,len(values),meta[3])
    return output


if __name__ == '__main__':
    import sys, collections
    table=read_columns(sys.argv[1],sys.argv[2:])
    for name,values in table.items(): print(name,len(values),collections.Counter(values).most_common(12))
