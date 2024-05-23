import sys
import multiprocessing as mp
from multiprocessing import shared_memory

ResultSuffix = "_civ"

DiffInTrace = 0.0
Minimum = 0
Maximum = 0
ArgList = []
DriverPath = ""
FpId = ""
VarList = []

StartTime = 0.0

RandomSeed = 0

Compiler = [#["nvcc", "CC=nvcc"], 
            ["clang", "CC=clang"], 
            #["gcc", "CC=gcc"],
            #["icc", "CC=icc"],
            ]

# for limiting input range of GetInconsistency()
TotalLimit = [-sys.float_info.max, sys.float_info.max]
RangeLimit = [-sys.float_info.max, sys.float_info.max]
Base = 1.0

OptimizerMode = False

SkipRandom = False

GlobalInputStr = None         # non-volatile but may have write conflict
GlobalMaximum = None        # non-volatile but may have write conflict
GlobalMinimum = None         # non-volatile but may have write conflict
GlobalInconsistency = None   # non-volatile but may have write conflict

TimePeriodCount = None
TimePeriod = 5.0

ProgramCallCount = None

SyncMode = False
SharedMem = None

BucketMode = True

RandMode = False
BinaryMode = False

NoLogMode = False

TimeLimit = 60.0

minDenormalFloat = float("1.4012984643248171e-45")

def InitializeConfig(size):
    print("size:", size)
    global ProgramCallCount, GlobalMaximum, GlobalMinimum, GlobalInconsistency, GlobalInputStr, TimePeriodCount
    ProgramCallCount = mp.Value('i', 0)
    GlobalMaximum = mp.Value('d', 0.0)
    GlobalMinimum = mp.Value('d', 0.0)
    GlobalInconsistency = mp.Value('d', 0.0)
    GlobalInputStr = shared_memory.ShareableList(["x" * 64] * size)
    TimePeriodCount = mp.Value('i', 1)

def SetMinDenormalFloat(fpId):
    global minDenormalFloat
    if fpId == "fp32":
        minDenormalFloat = float("1.4012984643248171e-45")
    else:
        minDenormalFloat = float.fromhex('0x0.0000000000001p-1022')    

def ConvertFloat(x):
    x = x.replace("f", "")
    x = x.replace("F", "")
    return x