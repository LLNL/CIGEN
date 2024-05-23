#!/usr/bin/env python3
from cmath import nan
import os
import math
import subprocess
import time
from enum import Enum
import multiprocessing as mp
import sys
import shutil
import io
import warnings
import config
import struct
import difflib
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')

class OptLevel(Enum):
    O0 = ["OPT_LEVEL=0", "FASTMATH=0"]
    O3_fastmath = ["OPT_LEVEL=3", "FASTMATH=1"]

def HexToFloat(hex, fpId):
    if fpId == "fp32":
        hex = hex.lower().replace("0x", "").replace("l", "").zfill(8)
    else:
        hex = hex.lower().replace("0x", "").replace("l", "").zfill(16)
    if all([a == '0' for a in list(hex)]):
        return 0.0
    if fpId == "fp32":
        return struct.unpack('!f', bytes.fromhex(hex))[0]
    else:
        return struct.unpack('!d', bytes.fromhex(hex))[0]

def FloatToHex(f, fpId):
    if fpId == "fp32":
        return '0x' + hex(struct.unpack('<I', struct.pack('<f', f))[0])[2:].zfill(8)
    else:
        return '0x' + hex(struct.unpack('<Q', struct.pack('<d', f))[0])[2:].zfill(16)

def ReadSetup():
    try:
        with open("./setup.ini", "r") as f:
            lines = f.readlines()
            for line in lines:
                segments = line.strip().split("=")
                if "CC" in segments[0]:
                    print("has CC")
                    config.Compiler = []
                    ccs = segments[1].split(",")
                    for cc in ccs:
                        print("compiler ", cc)
                        config.Compiler.append([cc.strip(), "CC=" + cc.strip()])
    except:
        pass

def IsSpecialValue(result):
    return "nan" in result or "inf" in result

def GetParts(n, fpId):
    if fpId == "fp32":
        b = bytearray(struct.pack('f', n))
        i = int.from_bytes(b, byteorder='little', signed=False)
        exponent = (i >> 23) & 0xFF
        mantissa = i & 0x7FFFFF
        sign = i >> 31
    else:
        b = bytearray(struct.pack('d', n))
        i = int.from_bytes(b, byteorder='little', signed=False)       
        exponent = (i >> 52) & 0x7FF
        mantissa = i & 0xFFFFFFFFFFFFF
        sign = i >> 63
    return exponent, mantissa, sign

def GetFloatFromParts(fpId, sign, exponent, mantissa):
    if fpId == "fp32":
        n = (sign << 31) + (exponent << 23) + mantissa
        b = bytearray(struct.pack('I', n))
        f = struct.unpack('f', b)[0]
    else:
        n = (sign << 63) + (exponent << 52) + mantissa
        b = bytearray(struct.pack('L', n))
        f = struct.unpack('d', b)[0]
    return f

def GetInconsistencyBits(n1, n2, fpId):
    if IsSpecialValue(n1):
        return 0.0
    n1 = config.ConvertFloat(n1)

    if IsSpecialValue(n2):
        n2 = n2.lower().replace("nan", "inf")
    else:
        n2 = config.ConvertFloat(n2)

    if "-" in n1:
        sign1 = 1
    else: 
        sign1 = 0
    if "-" in n2:
        sign2 = 1
    else: 
        sign2 = 0
    if fpId == "fp32":
        b1 = bytearray(struct.pack('f', float(n1)))
        b2 = bytearray(struct.pack('f', float(n2)))
        b1[-1] = b1[-1] & 0x7F
        b2[-1] = b2[-1] & 0x7F
        i1 = int.from_bytes(b1, byteorder='little', signed=False)
        i2 = int.from_bytes(b2, byteorder='little', signed=False)
        if sign1 != sign2:
            return i1 + i2
        else:
            return abs(i1 - i2)
    else:
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

def TraceDiff(tf1, tf2):
    ms = []
    temp = difflib.SequenceMatcher(None, tf1, tf2)
    return 1.0 - temp.ratio()

def BuildExe(driverPath, fpId):
    configTempDir = "./workspace/"
    origPath = os.getcwd()
    for cc in config.Compiler:
        for level in OptLevel:
            outFile = os.path.join(origPath, "build_" + cc[0] + "_" + level.name + ".out")

            command = ["make", "LLVM_PASS=1", cc[1]]
            command.extend(level.value)

            try:
                with open(outFile, "w") as f:
                    subprocess.check_call(command, stdout=f, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                print(str(e.output) + " : exiting")
                exit(0)

            # copy built executable to separate directories
            shutil.copytree(origPath, configTempDir + cc[0] + "_" + level.name, ignore=shutil.ignore_patterns("workspace"), symlinks=True)
    os.system("cp " + os.path.join(driverPath, "input_mutator") + " " + configTempDir)


def TestInput(driverPath, fpId, callCount, inputStr):
    configTempDir = "./" + str(callCount).zfill(8) + "/"
        
    concatStrList = WriteInputToStringList(inputStr)
    processList = []
    pDoneList = []
    for cc in config.Compiler:
        for level in OptLevel:
            outFile = os.path.join(configTempDir, "run_" + cc[0] + "_" + level.name + ".out")
            p = subprocess.Popen(["./test"] + concatStrList, shell=False, stdout=subprocess.PIPE, cwd=configTempDir + cc[0] + "_" + level.name)
            processList.append(p)
            pDoneList.append(False)

    currentTime = time.time()
    allPDone = False
    outputs = []
    while not allPDone:
        allPDone = True
        for i in range(len(processList)):
            p = processList[i]
            time.sleep(0.01)
            if pDoneList[i] == False and (not p.poll() is None):
                outs, _ = p.communicate()
                outStr = outs.decode("utf-8")
                outputs.append(outStr)
                pDoneList[i] = True
            if pDoneList[i] == False:
                allPDone = False
        newTime = time.time()
        if newTime - currentTime > 2.0:
            break

    if not allPDone:
        for i in range(len(processList)):
            p = processList[i]
            p.kill()
        return True, 0, 0.0, 0.0, False, 0.0

    inconsistency = 0.0
    prevOutFile = ""
    if os.path.exists("./compare_num.py"):
        inconsistency = int(subprocess.check_output(["python3", "./compare_num.py", str(callCount)]).decode("utf-8"))
    else:
        minimum = ""
        maximum = ""
        minHex = ""
        maxHex = ""
        minTrace = []
        maxTrace = []
        i = 0
        for cc in config.Compiler:    
            for level in OptLevel:
                f = io.StringIO(outputs[i])
                trace = []
                while True:
                    line = f.readline()
                    if not line.startswith("ins"):
                        break
                    trace.append(line.strip())
                line = line.strip()
                try:
                    rtext = HexToFloat(line, config.FpId)
                    strIO = io.StringIO()
                    print(rtext, file=strIO)
                    resultValue = strIO.getvalue().strip()
                    resultHex = line
                    strIO.close()
                except:
                    print("missing one output?")
                    raise
                if prevOutFile == "":
                    minimum = resultValue
                    minTrace = trace.copy()
                else:
                    maximum = resultValue
                    maxTrace = trace.copy()
                prevOutFile = outFile
                i += 1

        inconsistency = GetInconsistencyBits(minimum, maximum, fpId)
        if inconsistency < 0:
            inconsistency = (2**64) + float(inconsistency)
        
        diff = TraceDiff(minTrace, maxTrace)

    return False, inconsistency, minimum, maximum, False, diff

def ReadInput():
    maxUsed = sys.float_info.max
    with open("input.txt", "r") as f:
        inputLine = f.readline().strip()
        hexArgList = inputLine.split()
        argList = []
        maxUsed = 3.4028235e+38
        fpId = "fp32"
        suffix = "f"
        for arg in hexArgList:
            if len(arg.strip()) > 12:
                maxUsed = sys.float_info.max
                fpId = "fp64"
                suffix = ""
        for arg in hexArgList:
            argList.append(str(HexToFloat(arg.strip(), fpId)) + suffix)
    with open("variable.txt", "r") as f:
        lines = f.readlines()
        varSet = {int(num) for num in lines[0].strip().split()}
        limitSet = []
        for v in varSet:
            limitSet.append([-maxUsed, maxUsed])
        if len(lines) > 1:
            for idx, line in enumerate(lines):
                if idx == 0:
                    continue
                limits = line.strip().split(",")
                limitValues = [-maxUsed, maxUsed]
                for i in range(0, 2):
                    if not "none" in limits[i]:
                        limitValues[i] = float(limits[i])
                limitSet[idx-1] = [limitValues[0], limitValues[1]]
    return argList, varSet, limitSet

def WriteInputToStringList(argList):
    concatStrList = []
    for item in argList:
        concatStrList.append(FloatToHex(float(item), config.FpId))
    return concatStrList

def WriteInput(argList, configTempDir):
    concatStrList = WriteInputToStringList(argList)
    concatStr = " ".join(concatStrList) + "\n"

    for cc in config.Compiler:
        for level in OptLevel:
            inputPath = os.path.join(configTempDir, cc[0] + "_" + level.name, "input.txt")
            with open(inputPath, "w+") as f:
                f.write(concatStr)

def Test():
    config.ProgramCallCount = 0  
    path = "./variable.txt"
    driverPath = os.path.dirname(os.path.realpath(__file__))
    errorOccurred = True
    argList, varSet, limitSet = ReadInput()
    for idx, i in enumerate(varSet):
        if "f" in argList[i].lower():
            fpId = "fp32"
            config.FpId = fpId
        elif "." in argList[i].lower() or "e" in argList[i].lower():
            fpId = "fp64"
            config.FpId = fpId
    workspaceDir = "./workspace/"
    configTempDir = "./" + str(config.ProgramCallCount).zfill(8) + "/"
    shutil.copytree(workspaceDir, configTempDir, symlinks=True)
    while errorOccurred:
        _, resolved, config.Minimum, config.Maximum, errorOccurred, config.DiffInTrace = TestInput(driverPath, config.FpId, config.ProgramCallCount, argList)
    print("inconsistency:", resolved, " (", config.Minimum, ",", config.Maximum, ")")    

if __name__ == '__main__':
    os.system("make clean")
    config.SyncMode = True
    Test()