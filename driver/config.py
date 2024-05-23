import sys
import multiprocessing as mp
from multiprocessing import shared_memory

ResultSuffix = "_mcmc"      # non-volatile

Minimum = 0                 # volatile
Maximum = 0                 # volatile
ArgList = []                # volatile
DriverPath = ""             # non-volatile
FpId = ""                   # non-volatile
VarList = []                # non-volatile

StartTime = 0.0             # non-volatile

RandomSeed = 0              # non-volatile

Compiler = [#["nvcc", "CC=nvcc"], 
            ["clang", "CC=clang"], 
            #["gcc", "CC=gcc"],
            #["icc", "CC=icc"],
            ]

# for limiting input range of GetInconsistency()
TotalLimit = [-sys.float_info.max, sys.float_info.max] # non-volatile
RangeLimit = [-sys.float_info.max, sys.float_info.max] # volatile with each optimizer run
Base = 1.0                                             # volatile with each optimizer run

OptimizerMode = False       # non-volatile
EnableLogScaling = []

EnableDenormalInduction = False
DenormalInduction = False

# 0: all, 1: discounting NaN, 2: discounting inf and NaN

GlobalInputStr = [None, None, None]         # non-volatile but may have write conflict
GlobalMaximum = [None, None, None]        # non-volatile but may have write conflict
GlobalMinimum = [None, None, None]         # non-volatile but may have write conflict
GlobalInconsistency = [None, None, None]   # non-volatile but may have write conflict

TimePeriodCount = None
TimePeriod = 5.0
TotalTimeLimit = 60.0
HardTimeLimit = 300.0

TrimRangeLimit = 200

ProgramCallCount = None     # non-volatile but may have write conflict

SyncMode = False            # non-volatile
SharedMem = None            # non-volatile

NoLogMode = False
AllLogMode = False
Base2Coverage = False

PlotRandomPtMode = False
ExpName = ""

minDenormalFloat = float("1.4012984643248171e-45") # non-volatile
maxFloat = sys.float_info.max

def InitializeConfig(size):
    print("size:", size)
    global ProgramCallCount, GlobalMaximum, GlobalMinimum, GlobalInconsistency, GlobalInputStr, TimePeriodCount
    ProgramCallCount = mp.Value('i', 0)
    for i in range(3):
        GlobalMaximum[i] = mp.Value('d', 0.0)
        GlobalMinimum[i] = mp.Value('d', 0.0)
        GlobalInconsistency[i] = mp.Value('d', 0.0)
        GlobalInputStr[i] = shared_memory.ShareableList(["x" * 64] * size)
    TimePeriodCount = mp.Value('i', 1)

def SetMinDenormalFloat(fpId):
    global minDenormalFloat, maxFloat
    if fpId == "fp32":
        minDenormalFloat = float("1.4012984643248171e-45")
        maxFloat = 3.40282e+38
    else:
        minDenormalFloat = float.fromhex('0x0.0000000000001p-1022')
        maxFloat = sys.float_info.max

def ConvertFloat(x):
    x = x.replace("f", "")
    x = x.replace("F", "")
    return x