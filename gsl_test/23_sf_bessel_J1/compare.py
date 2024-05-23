#!/usr/bin/env python3
import os
import math
import re
import sys
import time

import json
import configparser
import subprocess
import struct

import glob

def IsSpecialValue(result):
    return "nan" in result or "inf" in result

def GetInconsistencyBits(n1, n2):
    if IsSpecialValue(n1):
        return 0.0

    if IsSpecialValue(n2):
        n2 = n2.lower().replace("nan", "inf")
        
    if "-" in n1:
        sign1 = 1
    else: 
        sign1 = 0
    if "-" in n2:
        sign2 = 1
    else: 
        sign2 = 0

    b1 = bytearray(struct.pack('d', float(n1)))
    b2 = bytearray(struct.pack('d', float(n2)))
    b1[-1] = b1[-1] & 0x7F
    b2[-1] = b2[-1] & 0x7F
    i1 = int.from_bytes(b1, byteorder='little', signed=False)
    i2 = int.from_bytes(b2, byteorder='little', signed=False)
    if sign1 != sign2:
        return i1 + i2
    else:
        return abs(i1 - i2)

transformIndex = int(sys.argv[1])
dirPrefix = sys.argv[2]
configDirBase = "./workspace/" + dirPrefix + "{:03d}".format(transformIndex)

fileList = sorted(glob.glob(configDirBase + "/run_*.out"))

prevSuccess = False
firstFile = True
values = []
for file in fileList:
    with open(file, "r") as f:
        f.readline()
        values.append(f.readline().strip())

maxInconsistency = 0

for i in range(len(values)):
    for j in range(len(values)):
        if i >= j:
            continue
        inconsistency = GetInconsistencyBits(values[i], values[j])
        print("inc:", values[i], values[j], math.log2(inconsistency + 1))
        if inconsistency < 0:
            inconsistency = (2**64) + float(inconsistency)
        if inconsistency > maxInconsistency:
            maxInconsistency = inconsistency

if math.log2(maxInconsistency + 1) > 5.53:
    exit(1)

exit(0)