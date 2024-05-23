#!/usr/bin/env python3
from cmath import nan
import os
import math
import subprocess
import time
from posixpath import split
import multiprocessing as mp
from multiprocessing import shared_memory
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
from matplotlib.patches import Rectangle
import csv
import sys
import shutil
import random
from datetime import datetime
import numpy
import sklearn.cluster
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy
import skopt
import warnings
import config
import tester as te
#import tikzplotlib
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')

def MultiplyPow2(i, x):
    try:
        if i > 2046.0:
            return x * math.pow(2, 1023.0) * math.pow(2, 1023.0) * math.pow(2, i - 2046.0)
        elif i > 1023.0:
            return x * math.pow(2, 1023.0) * math.pow(2, i - 1023.0)
        elif i < -2046.0:
            return x * math.pow(2, -1023.0) * math.pow(2, -1023.0) * math.pow(2, i + 2046.0)
        elif i < -1023.0:
            return x * math.pow(2, -1023.0) * math.pow(2, i + 1023.0)
        else:
            return x * math.pow(2, i)
    except OverflowError as e:
        return x
    
resultLogName = ""
resultRangeLogName = ""
resultRangeLogLock = mp.Value('i', 1)
osHomePath = os.path.expanduser('~')

numVariables = 0

# 0 - free, 1 - occupied, 2 - done
WorkerProcess = []
IsWorkerOccupied = []
WorkerX = []
SignSets = []
WorkerXStr = []
WorkerResults = []
TerminateWorkers = None
NumJobs = -1

def InconsistencyParallelWorker(index):
    workspaceDir = "./workspace/"
    configTempDir = "./" + str(index).zfill(8) + "/"
    shutil.copytree(workspaceDir, configTempDir, symlinks=True)

    while TerminateWorkers.value == 0:
        if IsWorkerOccupied[index].value == 1:
            x = []
            for item in WorkerX[index]:
                x.append(item)
            inc, maximum, minimum = GetInconsistencyParallelJob(x, None, index)
            WorkerResults[index][0] = inc
            WorkerResults[index][1] = maximum
            WorkerResults[index][2] = minimum
            IsWorkerOccupied[index].value = 2
        time.sleep(0.001)
    os.system("rm -r " + configTempDir)

def WaitInconsistencyParallelTask(zero_inc, all_inc):
    global SignSets, WorkerXStr
    HasOccupiedSlots = True
    while HasOccupiedSlots:
        HasOccupiedSlots = False
        for i in range(len(IsWorkerOccupied)):
            if IsWorkerOccupied[i].value == 2:
                # process data
                resolved = abs(WorkerResults[i][0])
                maximum = WorkerResults[i][1]
                minimum = WorkerResults[i][2]
                exponentSet = []
                mantissaSet = []
                for n in WorkerX[i]:
                    exponent, mantissa, _ = te.GetParts(n, config.FpId)
                    exponentSet.append(exponent)
                    mantissaSet.append(mantissa)
                inc = (SignSets[i], exponentSet, mantissaSet, WorkerXStr[i], float(minimum), float(maximum), resolved)
                if abs(resolved) > 1.0:
                    all_inc.append(inc)
                else:
                    zero_inc.append(inc)
                IsWorkerOccupied[i].value = 0
            if IsWorkerOccupied[i].value != 0:
                HasOccupiedSlots = True
        if HasOccupiedSlots:
            time.sleep(0.001)
    return zero_inc, all_inc

def AddInconsistencyParallelTask(x, xstr, data, zero_inc, all_inc):
    global SignSets, WorkerXStr
    HasFreeSlots = False
    while HasFreeSlots == False:
        for i in range(len(IsWorkerOccupied)):
            if IsWorkerOccupied[i].value != 1:
                HasFreeSlots = True
            if IsWorkerOccupied[i].value == 2:
                # process data
                resolved = abs(WorkerResults[i][0])
                maximum = WorkerResults[i][1]
                minimum = WorkerResults[i][2]
                exponentSet = []
                mantissaSet = []
                for n in WorkerX[i]:
                    exponent, mantissa, _ = te.GetParts(n, config.FpId)
                    exponentSet.append(exponent)
                    mantissaSet.append(mantissa)
                inc = (SignSets[i], exponentSet, mantissaSet, WorkerXStr[i], float(minimum), float(maximum), resolved)
                if abs(resolved) > 1.0:
                    all_inc.append(inc)
                else:
                    zero_inc.append(inc)
                IsWorkerOccupied[i].value = 0
        if HasFreeSlots == False:
            time.sleep(0.001)

    for i in range(len(IsWorkerOccupied)):
        if IsWorkerOccupied[i].value == 0:
            for j, xd in enumerate(x):
                WorkerX[i][j] = xd
            SignSets[i] = data
            WorkerXStr[i] = xstr
            IsWorkerOccupied[i].value = 1
            return zero_inc, all_inc
    
    raise Exception("no free slots")

def TerminateInconsistencyParallelWorker():
    TerminateWorkers.value = 1
    for i, process in enumerate(WorkerProcess):
        process.join()
        WorkerX[i].shm.close()
        WorkerX[i].shm.unlink()
        WorkerResults[i].shm.close()
        WorkerResults[i].shm.unlink()

def CreateInconsistencyParallelWorkers(size):
    global NumJobs, WorkerProcess, IsWorkerOccupied, WorkerX, WorkerResults, TerminateWorkers, SignSets, WorkerXStr
    if NumJobs == -1:
        NumJobs = mp.cpu_count()
    
    TerminateWorkers = mp.Value('i', 0)

    for i in range(NumJobs):
        IsWorkerOccupied.append(mp.Value('i', 0))
        WorkerX.append(shared_memory.ShareableList(["x" * 8] * size))
        SignSets.append([0] * size)
        WorkerXStr.append([""] * size)
        WorkerResults.append(shared_memory.ShareableList(["x" * 64] * 3))
        p = mp.Process(target=InconsistencyParallelWorker, args=(i,))
        WorkerProcess.append(p)
        p.start()

def GetInconsistency(x):
    inc, maximum, minimum = GetInconsistencyParallelJob(x, None, None)
    if config.DenormalInduction:
        return math.log2(abs(maximum) + 5e-324) + math.log2(abs(minimum) + 5e-324)
    else:
        return inc

def PrintInc(type, currentCallCount, argChanged, logAbsResolved):
    print(type + " " + str(currentCallCount) + ": set input " + str(config.VarList) + " to " + str(argChanged) + " : inconsistency " + str(logAbsResolved) + " (" + str(config.Minimum) + ", " + str(config.Maximum) + ")")    

def GetInconsistencyParallelJob(x, queue, index):
    with config.ProgramCallCount.get_lock():
        currentCallCount = config.ProgramCallCount.value 
        config.ProgramCallCount.value += 1
    if config.OptimizerMode:
        temp = []
        for idx, (i1, i2) in enumerate(zip(x, config.Base)):
            if config.EnableLogScaling[idx]:
                temp.append(MultiplyPow2(i1, i2))
            else:
                temp.append(i1)
        x = temp
    origValue = config.ArgList.copy()
    argChanged = []
    for i, idx in enumerate(config.VarList):
        config.ArgList[idx] = str(x[i]).replace("[", "").replace("]", "")
        argChanged.append(config.ArgList[idx])
    errorOccurred = True
    if index == None:
        dirName = currentCallCount
        workspaceDir = "./workspace/"
        configTempDir = "./" + str(dirName).zfill(8) + "/"
        shutil.copytree(workspaceDir, configTempDir, symlinks=True)
    else:
        dirName = index
        workspaceDir = "./workspace/"
        configTempDir = "./" + str(dirName).zfill(8) + "/"
    #te.WriteInput(config.ArgList, configTempDir)
    while errorOccurred:
        timeout, resolved, config.Minimum, config.Maximum, errorOccurred = te.TestInput(config.DriverPath, config.FpId, dirName, argChanged)
    if index == None:
        os.system("rm -r " + configTempDir)
    config.ArgList = origValue.copy()
    logAbsResolved = math.log2(abs(resolved) + 1)
    if config.OptimizerMode:
        inc = -logAbsResolved  
    else:
        inc = logAbsResolved
    if timeout:
        print("timeout:" + str(currentCallCount) + ": set input " + str(config.VarList) + " to " + str(argChanged))
    else:
        bestAll = logAbsResolved > config.GlobalInconsistency[0].value
        bestNum = not "nan" in config.Minimum and not "nan" in config.Maximum and logAbsResolved > config.GlobalInconsistency[1].value
        bestNormal = not te.IsSpecialValue(config.Minimum) and not te.IsSpecialValue(config.Maximum) and logAbsResolved > config.GlobalInconsistency[2].value

        if bestAll:
            with config.GlobalInconsistency[0].get_lock():
                for index, item in enumerate(argChanged):
                    config.GlobalInputStr[0][index] = item
                config.GlobalMaximum[0].value = float(config.Maximum)
                config.GlobalMinimum[0].value = float(config.Minimum)
                config.GlobalInconsistency[0].value = logAbsResolved
            elapsedTime = time.time() - config.StartTime
            with open(resultLogName, 'a') as f:
                print(elapsedTime, ",", config.GlobalInconsistency[0].value, \
                    ",", config.GlobalMaximum[0].value, ",", config.GlobalMinimum[0].value, \
                    ",", config.GlobalInputStr[0], file=f, sep='')
            PrintInc("all", currentCallCount, argChanged, logAbsResolved)
        if bestNum:
            with config.GlobalInconsistency[1].get_lock():
                for index, item in enumerate(argChanged):
                    config.GlobalInputStr[1][index] = item
                config.GlobalMaximum[1].value = float(config.Maximum)
                config.GlobalMinimum[1].value = float(config.Minimum)
                config.GlobalInconsistency[1].value = logAbsResolved            
            if not bestAll:
                PrintInc("num", currentCallCount, argChanged, logAbsResolved)
        if bestNormal:
            with config.GlobalInconsistency[2].get_lock():
                for index, item in enumerate(argChanged):
                    config.GlobalInputStr[2][index] = item
                config.GlobalMaximum[2].value = float(config.Maximum)
                config.GlobalMinimum[2].value = float(config.Minimum)
                config.GlobalInconsistency[2].value = logAbsResolved            
            if not bestAll and not bestNum:
                PrintInc("nrm", currentCallCount, argChanged, logAbsResolved)

        if not bestAll and not bestNum and not bestNormal and currentCallCount % 50 == 0:
            print(currentCallCount, end="\r")

    currentTimePeriod = config.TimePeriodCount.value
    elapsedTime = time.time() - config.StartTime
    if elapsedTime > currentTimePeriod * config.TimePeriod:
        TimePeriodIncremented = False
        with config.TimePeriodCount.get_lock():
            if config.TimePeriodCount.value == currentTimePeriod:
                config.TimePeriodCount.value = currentTimePeriod + 1
                TimePeriodIncremented = True
        if TimePeriodIncremented:
            with open(resultLogName, 'a') as f:
                print(elapsedTime, ",", config.GlobalInconsistency[0].value, \
                    ",", config.GlobalMaximum[0].value, ",", config.GlobalMinimum[0].value, \
                    ",", config.GlobalInputStr[0], file=f, sep='')
    if queue:
        queue.put([inc, float(config.Maximum), float(config.Minimum)])
    return inc, float(config.Maximum), float(config.Minimum)

def SaveFig(filename):
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    #tikzplotlib.save(filename.replace("pdf", "tex"))

def PlotNumCeil(n):
    abslog = math.log2(abs(n))
    if config.FpId == "fp32":
        abslog += 150
    else:
        abslog += 1075
    return math.copysign(abslog, n)

def Pow2Format(value, pos):
    if config.FpId == "fp32":
        abslog = abs(value) - 150
    else:
        abslog = abs(value) - 1075
    if abslog >= 1024:
        return " "
    pow2Value = math.copysign(math.pow(2.0, abslog), value)
    return "{:.0e}".format(pow2Value)

def PlotPoints(numVariables, all_inc, zero_inc, rangeList):
    font_size = 14
    formatter = ticker.FuncFormatter(Pow2Format)
    if numVariables == 1:
        plt.figure(figsize=(15,5))
        font_size = 20
    else:
        plt.figure(figsize=(10,10))
    plotx = []
    ploty = []
    strength = []
    for j, inc in enumerate(all_inc):
        if float(config.ConvertFloat(inc[3][0])) == 0.0:
            continue
        if numVariables == 1:
            ploty.append(0)
        else:
            if float(config.ConvertFloat(inc[3][1])) == 0.0:
                continue
            yy = PlotNumCeil(float(config.ConvertFloat(inc[3][1])))
            ploty.append(yy)
        xx = PlotNumCeil(float(config.ConvertFloat(inc[3][0])))
        plotx.append(xx)
        strength.append(abs(inc[6]))

    ax = plt.gca()
    ax.xaxis.set_major_formatter(formatter)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    if numVariables == 1:
        plt.ylabel('inconsistency') 
        plt.scatter(plotx, strength, c='green')
    else:
        ax.yaxis.set_major_formatter(formatter)
        strength = [x + 1.0 for x in strength]
        plt.scatter(plotx, ploty, s=strength, c='green')

    for inputRange in rangeList:
        startx = PlotNumCeil(inputRange[0][0])
        endx = PlotNumCeil(inputRange[1][0])
        if startx > endx:
            temp = startx
            startx = endx
            endx = temp
        if numVariables == 1:
            plt.gca().add_patch(Rectangle((startx, 0.0), endx - startx, 64.0, edgecolor='blue', facecolor='none',lw=3))
            print('add patch in', i, ":", startx, endx)
        else:
            starty = PlotNumCeil(inputRange[0][1])
            endy = PlotNumCeil(inputRange[1][1])
            if starty > endy:
                temp = starty
                starty = endy
                endy = temp
            plt.gca().add_patch(Rectangle((startx, starty), endx - startx, endy - starty, edgecolor='blue', facecolor='none',lw=3))
            print('add patch in', i, ":", startx, endx, starty, endy)

    plt.grid() 
    plt.tight_layout()
    plt.show()      
    samplingPointFigName = osHomePath + "/cigen/results/" + config.ResultSuffix + "/" + config.ExpName + "/sample_inc.pdf"
    SaveFig(samplingPointFigName)
    plotx = []
    ploty = []
    strength = []
    for j, inc in enumerate(zero_inc):
        if float(config.ConvertFloat(inc[3][0])) == 0.0:
            continue
        if numVariables == 1:
            ploty.append(0)
        else:
            if float(config.ConvertFloat(inc[3][1])) == 0.0:
                continue
            yy = PlotNumCeil(float(config.ConvertFloat(inc[3][1])))
            ploty.append(yy)
        xx = PlotNumCeil(float(config.ConvertFloat(inc[3][0])))
        plotx.append(xx)
        strength.append(abs(inc[6]))
    ax.xaxis.set_major_formatter(formatter)
    if numVariables == 1:
        plt.scatter(plotx, strength, c='red')  
    else:
        ax.yaxis.set_major_formatter(formatter)
        strength = [x + 1.0 for x in strength]
        plt.scatter(plotx, ploty, s=strength, c='red')
    plt.xticks(numpy.arange(-2048, 2048 + 512, 512))
    if numVariables == 1:
        plt.ylabel('inconsistency') 
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    if numVariables == 2:
        plt.yticks(numpy.arange(-2048, 2048 + 512, 512))
    plt.grid() 
    plt.tight_layout()
    plt.show() 
    samplingPointFigName = osHomePath + "/cigen/results/" + config.ResultSuffix + "/" + config.ExpName + "/sample.pdf"
    SaveFig(samplingPointFigName)
    plt.clf()

def GetRandomFloat(fpId, minexponent, minmantissa, maxexponent, maxmantissa, sign):
    if fpId == "fp32":
        numInExponent = 2**23
    else:
        numInExponent = 2**52
    totalNum = (maxexponent - minexponent) * numInExponent + (maxmantissa - minmantissa)
    if totalNum < 0:
        print("error:", minexponent, minmantissa, maxexponent, maxmantissa, sign)
        input("enter...")
    position = random.randint(1, totalNum - 1)
    newexponent = minexponent
    while position >= numInExponent:
        newexponent = newexponent + 1
        position = position - numInExponent
    newmantissa = position + minmantissa
    if newmantissa >= numInExponent:
        newexponent = newexponent + 1
        newmantissa = newmantissa - numInExponent
    output = te.GetFloatFromParts(fpId, sign, newexponent, newmantissa)
    return output, newexponent, newmantissa

def GetFloatComponents(x):
    ret = te.GetParts(x, config.FpId)
    return ret[2], ret[0], ret[1]

def GetFloatStrComponents(x):
    x = float(x)
    ret = te.GetParts(x, config.FpId)
    return ret[2], ret[0], ret[1]    

def FilterRange(inputRange):
    return False
    if config.FpId == "fp32":
        minimumNormal = 1.175494e-38
        subnormalLimit = 22.99
        bitLimit = 31.5
    else:
        minimumNormal = sys.float_info.min
        subnormalLimit = 51.99
        bitLimit = 63.5
    if (inputRange[2][4] == 0.0 and abs(inputRange[2][5]) < minimumNormal) or \
        (inputRange[2][5] == 0.0 and abs(inputRange[2][4]) < minimumNormal):
        if config.GlobalInconsistency[0].value > subnormalLimit:
            return True
    else:
        if config.GlobalInconsistency[0].value > bitLimit:
            return True

    return False

def ExitControl():
    currentTime = time.time()
    if currentTime - config.StartTime > config.TotalTimeLimit * (2**len(config.ArgList)):
        return True
    if config.HardTimeLimit > 0.0 and currentTime - config.StartTime > config.HardTimeLimit:
        return True
    return False

def de_callback(xk, convergence):
    return ExitControl()

def PrintRangeResultToCSV(inputRange, maxPoint, result):
    if config.DenormalInduction:
        config.DenormalInduction = False
        result = GetInconsistency(maxPoint)
    if abs(result) < abs(inputRange[2][6]):
        result = inputRange[2][6]
        maxPoint = inputRange[2][3]
    with resultRangeLogLock.get_lock():
        with open(resultRangeLogName, "a") as f:
            writer_object = csv.writer(f)
            entry = []
            for i in range(len(inputRange[0])):
                entry.append(inputRange[0][i])
                entry.append(inputRange[1][i])
            for i in range(len(inputRange[0])):
                entry.append(maxPoint[i])
            entry.append(abs(result))
            if len(inputRange) >= 4:
                entry.append(inputRange[3][0])
                entry.append(inputRange[3][1])
            else:
                entry.append(0.0)
                entry.append(0.0)
            writer_object = csv.writer(f)
            writer_object.writerow(entry)                   

def Opt_DifferentialEvolution(argList, fpId, exponentList, rangeList, numInExponent):
    for inputRange in rangeList:
        # filter out ranges that do not need work
        if FilterRange(inputRange):
            continue
        config.Base = inputRange[0]
        config.RangeLimit = inputRange
        print("config.Base = ", config.Base)
        print("limit = ", inputRange[1])
        config.OptimizerMode = True
        bounds = []
        config.EnableLogScaling = []
        if config.EnableDenormalInduction and abs(inputRange[2][6]) < 52.0 and abs(inputRange[2][4]) < 1e-4 and abs(inputRange[2][5]) < 1e-4:
            config.DenormalInduction = True
        for i in range(len(config.Base)):
            currentBounds = []
            i1 = inputRange[0][i]
            i2 = inputRange[1][i]
            if config.NoLogMode or (not config.AllLogMode and abs(math.log2(abs(i2)) - math.log2(abs(i1))) < 10.0):
                bounds.append([i1, i2])
                config.EnableLogScaling.append(False)
            else:
                if i1 < 0.0:
                    currentBounds.append(math.log2(abs(i2)) - math.log2(abs(i1)))
                    currentBounds.append(0.0)
                else:
                    currentBounds.append(0.0)
                    currentBounds.append(math.log2(abs(i2)) - math.log2(abs(i1)))
                bounds.append(currentBounds)
                config.EnableLogScaling.append(True)
        print("bounds:")
        print(bounds)
        result = scipy.optimize.differential_evolution(GetInconsistency, bounds, updating='deferred', workers=-1, popsize=20, maxiter=50, callback=de_callback)
        maxPoint = []
        idx = 0
        for i1, i2 in zip(result.x, config.Base):
            if config.EnableLogScaling[idx]:
                maxPoint.append(MultiplyPow2(i1, i2))
            else:
                maxPoint.append(result.x[idx])
            idx += 1
        #maxPoint = [MultiplyPow2(i1, i2) for i1, i2 in zip(result.x, config.Base)]
        print("max point: ", maxPoint, " inconsistency bits: ", result.fun)    
        PrintRangeResultToCSV(inputRange, maxPoint, result.fun)
        config.OptimizerMode = False     
        config.DenormalInduction = False                    
    
gEdgeValues32 = [(0, 254, (2**23)-1), (1, 254, (2**23)-1), (0, 0, (2**23)-1), (1, 0, (2**23)-1)]
gEdgeValues64 = [(0, 2046, (2**52)-1), (1, 2046, (2**52)-1), (0, 0, (2**52)-1), (1, 0, (2**52)-1)]

gMultiRange32 = [0, 2, 94, 125, 127, 130, 160, 251, 253, 256]
gMultiRange64 = [0, 5, 692, 991, 1020, 1023, 1026, 1056, 1355, 2042, 2045, 2048]

def SortInc(inc):
    return abs(inc[6])

def RandomSampling(maxexponent, numInExponent, varList, exponentList, mantissaList):
    avg_inc = []
    limitMinSign = []
    limitMinExponent = []
    limitMinMantissa = []
    limitMaxSign = []
    limitMaxExponent = []
    limitMaxMantissa = []    
    limitMinAtEdge = []
    limitMaxAtEdge = []
    all_inc = []
    zero_inc = []    
    for idx in range(0, numVariables):
        if config.TotalLimit[idx][0] > -sys.float_info.max:
            sign_, exponent_, mantissa_ = GetFloatComponents(config.TotalLimit[idx][0])
            limitMinAtEdge.append(False)
        else:
            sign_ = 1
            exponent_ = maxexponent - 2
            mantissa_ = numInExponent - 1
            limitMinAtEdge.append(True)
        limitMinSign.append(sign_)
        limitMinExponent.append(exponent_)
        limitMinMantissa.append(mantissa_)
        if config.TotalLimit[idx][1] < sys.float_info.max:
            sign_, exponent_, mantissa_ = GetFloatComponents(config.TotalLimit[idx][1])
            limitMaxAtEdge.append(False)
        else:
            sign_ = 0
            exponent_ = maxexponent - 2
            mantissa_ = numInExponent - 1            
            limitMaxAtEdge.append(True)
        limitMaxSign.append(sign_)
        limitMaxExponent.append(exponent_)
        limitMaxMantissa.append(mantissa_)

    if config.FpId == "fp32":
        multiRange = gMultiRange32.copy()
        baseExp = 127

        if not config.Base2Coverage:
            powBase = 10
            bottomBase = -45
            topBase = 83
            largestPow = 38.53
            jumpPos = 1000
        else:
            powBase = 2
            bottomBase = -149
            topBase = 149 + 128
            largestPow = 128
            jumpPos = 1000
    else:
        multiRange = gMultiRange64.copy()                     
        baseExp = 1023

        if not config.Base2Coverage:
            powBase = 10
            bottomBase = -324
            topBase = 633
            largestPow = 308.25
            jumpPos = 1000
        else:
            powBase = 2
            bottomBase = -1074
            topBase = 1074 + 1024
            largestPow = 1024
            jumpPos = 5000

    numEdgeValues = numVariables
    iter = 512 * (2**(numVariables - 1))
    processList = []
    coverage = [set()] * numVariables
    sample = 0
    currentCoverageIdx = -1
    while True:
        if sample == iter:
            currentCoverageIdx += 1
            # prepare the coverage sets
            for i in range(numVariables):
                c = set()
                for j in range(topBase):
                    if not j in coverage[i]:
                        c.add(j)
                for j in range(jumpPos, jumpPos + topBase):
                    if not j in coverage[i]:
                        c.add(j)
                coverage[i] = c


        # when numEdgeValues == 4: (4^4 - 2^4)+(2^4)*(4-4+1)
        if numEdgeValues > 0 and sample >= (2**numVariables - 2**numVariables) + (2**numVariables)*(numVariables - numEdgeValues + 1):
            numEdgeValues = numEdgeValues - 1
        if numEdgeValues == numVariables:
            useEdgeValues = []
            for idx in range(0, numVariables):
                useEdgeValues.append(True)
        else:
            useEdgeValues = []
            pb = numEdgeValues / numVariables
            for idx in range(0, numVariables):
                useEdgeValues.append(True if random.random() < pb else False)

        inputValues = []
        inputValueStr = []
        signSet = []
        exponentSet = []
        mantissaSet = []

        skipInput = False
        for idx in range(0, numVariables):    
            totalLimit0 = config.TotalLimit[idx][0]
            totalLimit1 = config.TotalLimit[idx][1]
            if config.TotalLimit[idx][0] == 0:
                totalLimit0 = config.minDenormalFloat
            if config.TotalLimit[idx][1] == 0:
                totalLimit1 = -config.minDenormalFloat                    
            if limitMinSign[idx] != limitMaxSign[idx]:
                sign = random.randint(0, 1)
                startRange = 0
                endRange = 0
                if sign == 1:
                    limitMax = math.log2(abs(totalLimit0)) + baseExp
                else:
                    limitMax = math.log2(abs(totalLimit1)) + baseExp                                         
                for i in range(len(multiRange)):
                    if i == len(multiRange) - 1:
                        endRange = i - 1
                        break
                    if limitMax < multiRange[i]:
                        endRange = i
                        break
            else:
                sign = limitMinSign[idx]
                startRange = 0
                endRange = 0
                if totalLimit0 > 0.0:
                    limitMin = math.log2(abs(totalLimit0)) + baseExp
                    limitMax = math.log2(abs(totalLimit1)) + baseExp   
                else:
                    limitMin = math.log2(abs(totalLimit1)) + baseExp
                    limitMax = math.log2(abs(totalLimit0)) + baseExp                       
                for i in range(len(multiRange)):
                    if limitMin > multiRange[i] or (i == 0 and limitMin <= 0):
                        startRange = i
                    if i == len(multiRange) - 1:
                        endRange = i - 1
                        break
                    if limitMax < multiRange[i]:
                        endRange = i
                        break
            if idx == currentCoverageIdx:
                coverageLog = coverage[idx].pop()
                if coverageLog >= jumpPos:
                    outputNum = -1
                    sign = 1
                    coverageLog -= jumpPos
                else:
                    sign = 0
                    outputNum = 1
                while True:
                    powToUse = random.uniform(0, 1) + (coverageLog + bottomBase)
                    if powToUse < largestPow:
                        break
                outputNum *= math.pow(powBase, powToUse)
                if outputNum < config.TotalLimit[idx][0] or \
                   outputNum > config.TotalLimit[idx][1]:
                    skipInput = True
                output = str(outputNum)
                if config.FpId == "fp32":
                    output = str(outputNum) + "f"
            elif useEdgeValues[idx]:
                if sample >= 2**numVariables:
                    edgeIndex = random.randint(0, len(gEdgeValues32) - 1)                     
                else:
                    sampleStr = numpy.base_repr(sample, base=4)
                    if idx >= len(sampleStr):
                        edgeIndex = 0
                    else:
                        edgeIndex = int(sampleStr[idx])
                if edgeIndex == 0 and limitMaxAtEdge:
                    sign = limitMaxSign[idx]
                    expChange = limitMaxExponent[idx]
                    newmantissa = limitMaxMantissa[idx]
                elif edgeIndex == 1 and limitMinAtEdge:
                    sign = limitMinSign[idx]
                    expChange = limitMinExponent[idx]
                    newmantissa = limitMinMantissa[idx]    
                elif edgeIndex >= 2 and limitMinSign[idx] == limitMaxSign[idx]:
                    useEdgeValues[idx] = False      
                elif config.FpId == "fp32":
                    sign = gEdgeValues32[edgeIndex][0]
                    expChange = gEdgeValues32[edgeIndex][1]
                    newmantissa = gEdgeValues32[edgeIndex][2]
                else:
                    sign = gEdgeValues64[edgeIndex][0]
                    expChange = gEdgeValues64[edgeIndex][1]
                    newmantissa = gEdgeValues64[edgeIndex][2]                       
            else:
                while True:
                    inputArea = random.randint(startRange, endRange)
                    f, _, _ = GetRandomFloat(config.FpId, multiRange[inputArea], 0, multiRange[inputArea + 1], 0, sign)
                    randInput = math.log2(abs(f)) + baseExp
                    if randInput < limitMax:
                        break                
                outputNum = f
                output = str(outputNum)
                if config.FpId == "fp32":
                    output = str(outputNum) + "f"
            if expChange >= maxexponent - 1:
                expChange = maxexponent - 2
            if useEdgeValues[idx]:        
                outputValue = te.GetFloatFromParts(config.FpId, sign, expChange, newmantissa)    
                output = str(outputValue)
            else:
                outputValue = float(config.ConvertFloat(output.strip()))
            inputValues.append(outputValue)
            inputValueStr.append(output.strip())
            if currentCoverageIdx == -1:
                coverageBase = int(math.log10(abs(outputValue)) - bottomBase)
                if outputValue < 0:
                    coverageBase += jumpPos
                coverage[idx].add(coverageBase)
            signSet.append(sign)
            exponentSet.append(expChange)
            mantissaSet.append(newmantissa)
            if idx == currentCoverageIdx and coverage[idx] == set():
                currentCoverageIdx += 1

        if not skipInput:
            zero_inc, all_inc = AddInconsistencyParallelTask(inputValues, inputValueStr, signSet, zero_inc, all_inc)
            sample += 1
        if ExitControl():
            break
        if currentCoverageIdx >= numVariables:
            break
    if ExitControl():
        return all_inc, zero_inc

    zero_inc, all_inc = WaitInconsistencyParallelTask(zero_inc, all_inc)

    return all_inc, zero_inc

def FindRangeWithClustering(all_inc, zero_inc, numVariables, numInExponent):
    if len(all_inc) == 0:
        return []

    linkage_arr = []
    minVec = []
    maxVec = []
    all_inc_temp = []
    for inc in all_inc:
        all_inc_temp.append(inc)

    if len(all_inc_temp) == 0:
        return []

    for inc in all_inc_temp:
        inc_vec = [] 
        for i in range(numVariables):
            value_int_rep = inc[1][i] * numInExponent + inc[2][i]
            if inc[0][i] == 1:
                value_int_rep = -value_int_rep
            if config.FpId == "fp32":
                value_float_rep = float(value_int_rep) / 1e5
            else:
                value_float_rep = float(value_int_rep) / 1e10
            inc_vec.append(value_float_rep)
            if len(minVec) == i:
                minVec.append(sys.float_info.max)
                maxVec.append(-sys.float_info.max)
            if value_float_rep < minVec[i]:
                minVec[i] = value_float_rep
            if value_float_rep > maxVec[i]:
                maxVec[i] = value_float_rep
        linkage_arr.append(inc_vec)

    for inc in zero_inc:
        for i in range(numVariables):
            value_int_rep = inc[1][i] * numInExponent + inc[2][i]
            if inc[0][i] == 1:
                value_int_rep = -value_int_rep
            if config.FpId == "fp32":
                value_float_rep = float(value_int_rep) / 1e5
            else:
                value_float_rep = float(value_int_rep) / 1e10
            if value_float_rep < minVec[i]:
                minVec[i] = value_float_rep
            if value_float_rep > maxVec[i]:
                maxVec[i] = value_float_rep        
    
    # normalize input
    for linkage in linkage_arr:
        for i in range(numVariables):
            if maxVec[i] == minVec[i]:
                linkage[i] = 1.0
            else:
                linkage[i] = (linkage[i] - minVec[i]) / (maxVec[i] - minVec[i])
    A = numpy.array(linkage_arr)
    initDist = 0.2 * math.sqrt(numVariables)
    buckets = []
    if len(linkage_arr) > 1:
        while True:
            hac = sklearn.cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=initDist, metric='euclidean', linkage='single')
            labels = hac.fit_predict(A)
            if numVariables == 1 or (len(numpy.unique(labels) ) < math.pow(2, numVariables)):
                break
            initDist = initDist * 1.2   
            if initDist > 0.3:
                break
        for i in range(len(numpy.unique(labels))):
            buckets.append([])
        for i in range(len(labels)):
            buckets[labels[i]].append(all_inc_temp[i])
    elif len(linkage_arr) == 1:
        buckets.append([])
        buckets[0].append(all_inc_temp[0])

    # expand range with zeroes outside the range
    rangeList = []
    totalMaxInc = 0.0
    for bucket in buckets:
        minVec = []
        maxVec = []        
        initialGuess = ()
        maxInc = 0.0
        for inc in bucket:
            if maxInc < abs(inc[6]):
                maxInc = abs(inc[6])
                initialGuess = inc
            for i in range(numVariables):

                value_rep = float(config.ConvertFloat(inc[3][i]))
                if len(minVec) == i:
                    minVec.append(sys.float_info.max)
                    maxVec.append(-sys.float_info.max)
                if value_rep < minVec[i]:     
                    minVec[i] = value_rep
                if value_rep > maxVec[i]:
                    maxVec[i] = value_rep
        minOutsideVec = []
        maxOutsideVec = []
        for i in range(numVariables):
            if initialGuess[0][i] == 0 and config.TotalLimit[i][0] < 0:
                minOutsideVec.append(config.minDenormalFloat)
            else:
                minOutsideVec.append(config.TotalLimit[i][0])
            if initialGuess[0][i] == 1 and config.TotalLimit[i][1] > 0:
                maxOutsideVec.append(-config.minDenormalFloat)
            else:
                maxOutsideVec.append(config.TotalLimit[i][1])
        for inc in zero_inc:
            for i in range(numVariables):
                value_rep = float(config.ConvertFloat(inc[3][i]))
                if value_rep < minVec[i] and value_rep > minOutsideVec[i]:
                    minOutsideVec[i] = value_rep
                if value_rep > maxVec[i] and value_rep < maxOutsideVec[i]:
                    maxOutsideVec[i] = value_rep
        
        for i in range(len(minOutsideVec)):
            if minOutsideVec[i] == 0.0:
                minOutsideVec[i] = math.copysign(config.minDenormalFloat, maxOutsideVec[i])
        for i in range(len(maxOutsideVec)):
            if maxOutsideVec[i] == 0.0:
                maxOutsideVec[i] = math.copysign(config.minDenormalFloat, minOutsideVec[i])                
        rangeList.append([minOutsideVec, maxOutsideVec, initialGuess])    
        if totalMaxInc < maxInc:
            totalMaxInc = maxInc

    # filter out small inconsistency range when large ones are present
    if totalMaxInc > 50.0:
        newRangeList = []
        for inputRange in rangeList:
            if abs(inputRange[2][6]) > 3.0:
                newRangeList.append(inputRange)
        rangeList = newRangeList
    return rangeList

def TuneRanges(rangeList):
    result = rangeList.copy()
    print("before tune ranges:", len(result))
    if len(result) > 200:
        return rangeList
    while True:
        marki = -1
        markj = -1
        for i in range(len(result) - 1):
            for j in range(i+1, len(result)):
                range1 = result[i]
                range2 = result[j]

                ixmin = []
                ixmax = []
                area1 = 1.0
                area2 = 1.0
                intersect = True
                areaintersect = 1.0
                for k in range(len(range1[0])):
                    axismin = max(range1[0][k], range2[0][k])
                    axismax = min(range1[1][k], range2[1][k])
                    if axismin > axismax:
                        intersect = False
                        break
                    ixmin.append(axismin)
                    ixmax.append(axismax)

                    area1 = area1 * abs(math.log2(abs(range1[1][k])) - math.log2(abs(range1[0][k])))
                    area2 = area2 * abs(math.log2(abs(range2[1][k])) - math.log2(abs(range2[0][k])))
                    areaintersect = areaintersect * abs(math.log2(abs(axismin)) - math.log2(abs(axismax)))

                intersectpercent = areaintersect / (area1 + area2 - areaintersect)
                intersectin1 = areaintersect / area1
                intersectin2 = areaintersect / area2
                if intersect and \
                    (intersectpercent > math.pow(0.5, len(range1[0])) or \
                    intersectin1 > math.pow(0.5, len(range1[0])) or \
                    intersectin2 > math.pow(0.5, len(range1[0]))):
                    marki = i
                    markj = j
                    break
                if ExitControl():
                    break
            if ExitControl():
                break
            if marki != -1:
                break

        if ExitControl():
            break

        if marki == -1 and markj == -1:
            break
        else:
            newRangeList = []
            for i in range(len(result)):
                if i != marki and i != markj:
                    newRangeList.append(result[i])
                elif i == marki:
                    # merge marki and markj
                    range1 = result[marki]
                    range2 = result[markj]
                    unionmin = []
                    unionmax = []
                    for k in range(len(range1[0])):
                        axismin = min(range1[0][k], range2[0][k])
                        axismax = max(range1[1][k], range2[1][k])
                        unionmin.append(axismin)
                        unionmax.append(axismax)
                    if abs(range1[2][6]) > abs(range2[2][6]):
                        initialGuess = range1[2]
                    else:
                        initialGuess = range2[2]
                    newRangeList.append([unionmin, unionmax, initialGuess])
            result = newRangeList[:]
    return result

# first, enlarge the input range a bit; then put more samples in these ranges
# to check out the range's statistics
def InspectRanges(rangeList):
    print("total ranges: ", len(rangeList ))
    if config.TrimRangeLimit > 0 and len(rangeList) >= config.TrimRangeLimit:
        return rangeList
    if ExitControl():
        return rangeList    
    for idx, r in enumerate(rangeList):
        if ExitControl():
            return rangeList
        zero_inc = []
        all_inc = []

        for i in range(numVariables):
            if r[0][i] > 0.0:
                lowerBound = r[0][i] / 10.0
            else:
                lowerBound = r[0][i] * 10.0
            if lowerBound == 0.0:
                lowerBound = math.copysign(config.minDenormalFloat, r[0][i])
            if te.IsSpecialValue(str(lowerBound)):
                lowerBound = math.copysign(config.maxFloat, r[0][i])
            r[0][i] = lowerBound
        for i in range(numVariables):
            if r[1][i] > 0.0:
                upperBound = r[1][i] * 10.0
            else:
                upperBound = r[1][i] / 10.0
            if upperBound == 0.0:
                upperBound = math.copysign(config.minDenormalFloat, r[1][i])
            if te.IsSpecialValue(str(upperBound)):
                upperBound = math.copysign(config.maxFloat, r[1][i])
            r[1][i] = upperBound
        print("range: ")
        print(r[0])
        print(r[1])
        if len(r) > 2:
            print(r[2][3], r[2][6])   

        # sampling matrix!
        DenseSampleAmount = 256 * (2**numVariables)
        r0_parts = []
        r1_parts = []
        for i in range(numVariables):
            r0_parts.append(te.GetParts(r[0][i], config.FpId))
            r1_parts.append(te.GetParts(r[1][i], config.FpId))
        for sample in range(DenseSampleAmount):
            inputValues = []
            inputValueStr = []
            for i in range(numVariables):
                if r[2][0][i] == 0:
                    f, _, _ = GetRandomFloat(config.FpId, r0_parts[i][0], r0_parts[i][1], r1_parts[i][0], r1_parts[i][1], r[2][0][i])
                else:
                    f, _, _ = GetRandomFloat(config.FpId, r1_parts[i][0], r1_parts[i][1], r0_parts[i][0], r0_parts[i][1], r[2][0][i])
                inputValues.append(f)
                inputValueStr.append(str(f))
            zero_inc, all_inc = AddInconsistencyParallelTask(inputValues, inputValueStr, r[2][0], zero_inc, all_inc)
            if ExitControl():
                break
        zero_inc, all_inc = WaitInconsistencyParallelTask(zero_inc, all_inc)
        all_inc.sort(key=SortInc)
        if numVariables <= 5:
            print(*all_inc, sep="\n")
        else:
            print("total inc:", len(all_inc))
        all_inc_num = [r[2][6]]
        for inc in all_inc:
            all_inc_num.append(abs(inc[6]))
        avg_inc = sum(all_inc_num) / len(all_inc_num)
        if len(all_inc) > 0 and abs(all_inc[-1][6]) > abs(r[2][6]):
            rangeList[idx][2] = all_inc[-1]
        rangeList[idx].append([(len(all_inc) + 1) / (DenseSampleAmount + 1), avg_inc])
        print("range after: ")
        print(r[0])
        print(r[1])
        if len(r) > 2:
            print(r[2][3], r[2][6])   

    return rangeList

def OptimizationMethod(argList, fpId, exponentList, rangeList, numInExponent):
    print("use", config.ResultSuffix, " to optimize")
    funcName = Opt_DifferentialEvolution
    parallelNum = mp.cpu_count()
    processList = []
    for inputRange in rangeList:
        if ExitControl():
            break            
        while len(processList) >= parallelNum:
            time.sleep(0.01)
            newProcessList = []
            for p in processList:
                if p.is_alive():
                    newProcessList.append(p)
                else:
                    p.join()
            processList = newProcessList
        p = mp.Process(target=funcName, args=(argList, fpId, exponentList, [inputRange], numInExponent,))
        processList.append(p)
        p.start()

    for p in processList:
        p.join()

def MutateRoutine(driverPath, argList, varSet, limitSet):
    varindex = 0
    mantissaList = []
    exponentList = []
    varList = []
    config.TotalLimit = limitSet
    for idx, i in enumerate(varSet):    
        if "f" in argList[i].lower():
            fpId = "fp32"
            betweenId = "between32"
            _, exponent, mantissa = GetFloatStrComponents(argList[i])
            maxexponent = 256
            numInExponent = 2**23        
        elif "." in argList[i].lower() or "e" in argList[i].lower():
            fpId = "fp64"
            betweenId = "between64"     
            _, exponent, mantissa = GetFloatStrComponents(argList[i])
            maxexponent = 2048
            numInExponent = 2**52
        else: # integer
            fpId = "int"
            mantissa = int(argList[i].lower())
        
        mantissaList.append(mantissa)
        exponentList.append(exponent)

        config.FpId = fpId
        varList.append(i)
        print("test input " + str(i) + ", value: " + str(argList[i]))
        varindex += 1

    te.BuildExe(driverPath, fpId)
    config.StartTime = time.time()
    config.VarList = varList
    CreateInconsistencyParallelWorkers(len(argList))
    ret = []
    all_inc, zero_inc = RandomSampling(maxexponent, numInExponent, varList, exponentList, mantissaList)
    all_inc.sort(key=SortInc)
    if numVariables <= 5:
        print(*all_inc, sep="\n")
    else:
        print("total inc:", len(all_inc))
    all_inc_sign = []
    zero_inc_sign = []
    ret = [len(all_inc) + len(zero_inc), len(all_inc), config.GlobalInconsistency[0].value]
    if ExitControl():
        return ret    
    for i in range(2**numVariables):
        all_inc_sign.append([])
        zero_inc_sign.append([])
    for inc in all_inc:
        signIndex = 0
        for i in range(numVariables):
            signIndex += inc[0][i]*(2**i)
        all_inc_sign[signIndex].append(inc)
    for inc in zero_inc:
        signIndex = 0
        for i in range(numVariables):
            signIndex += inc[0][i]*(2**i)
        zero_inc_sign[signIndex].append(inc)      
    rangeList = []      
    for i in range(2**numVariables):
        print("check cluster:", i)
        rangeList.extend(FindRangeWithClustering(all_inc_sign[i], zero_inc_sign[i], numVariables, numInExponent))
    rangeList = TuneRanges(rangeList)

    if config.PlotRandomPtMode and numVariables <= 2:
        PlotPoints(numVariables, all_inc, zero_inc, rangeList)

    if ExitControl():
        return ret
    rangeList = InspectRanges(rangeList)
    if ExitControl():
        return ret
    OptimizationMethod(argList, fpId, exponentList, rangeList, numInExponent)
    return ret

if __name__ == '__main__':
    driverPath = os.path.dirname(os.path.realpath(__file__))
    currentdir = os.getcwd()
    config.ExpName = os.path.basename(currentdir)
    
    os.system("make clean")

    for compiler in config.Compiler:
        if compiler[0] == "nvcc":
            x = subprocess.check_output(['nvidia-smi']).decode("utf-8") 
            if "Failed" in x or "Unknown Error" in x:
                exit(0)
    
    config.RandomSeed = int(datetime.now().timestamp()) % 300000
    config.ResultSuffix = "_de"
    gTestOnly = False
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("-seed="):
                config.RandomSeed = int(arg.replace("-seed=", ""))
            if arg.startswith("-jobs="):
                NumJobs = int(arg.replace("-jobs=", ""))
            if arg.startswith("-testonly"):
                gTestOnly = True
            if arg.startswith("-nolog"):
                config.NoLogMode = True
            if arg.startswith("-plot"):
                config.PlotRandomPtMode = True

    os.system("mkdir -p ~/cigen/results/" + config.ResultSuffix)
    os.system("mkdir -p ~/cigen/results/" + config.ResultSuffix + "/" + config.ExpName + "/")
    resultLogName = osHomePath + "/cigen/results/" + config.ResultSuffix + "/" + config.ExpName + "/timelapse.txt"
    resultFigName = osHomePath + "/cigen/results/" + config.ResultSuffix + "/" + config.ExpName + "/incgraph.pdf"
    resultRangeLogName = osHomePath + "/cigen/results/" + config.ResultSuffix + "/" + config.ExpName + "/ranges.csv"
    with open(resultLogName, 'w') as f:
        print("time,inc,max,min,input", file=f)

    if gTestOnly:
        te.Test()
    else:
        numpy.random.seed(config.RandomSeed)
        random.seed(config.RandomSeed) 

        te.ReadSetup()

        argList, varSet, limitSet = te.ReadInput()
        config.DriverPath = driverPath
        config.ArgList = argList 
        numVariables = len(argList)
        config.InitializeConfig(len(argList))
        #if numVariables >= 4 or config.FpId == "fp32":
        #    exit(0)
        with open(resultRangeLogName, "w") as f:
            header = []
            for i in range(len(argList)):
                header.append("min_" + str(i))
                header.append("max_" + str(i))
            for i in range(len(argList)):
                header.append("best_" + str(i))
            header.append("max_inc")     
            header.append("per")
            header.append("avg_inc")           
            writer_object = csv.writer(f)
            writer_object.writerow(header)

        ret = MutateRoutine(driverPath, argList, varSet, limitSet)

        endTime = time.time()
        print("=== results ===: seed", config.RandomSeed, "time:", endTime - config.StartTime, "callcount:", config.ProgramCallCount.value)
        print("callcount_random, num_random, inc_random =", ret[0], ret[1], ret[2])
        if config.GlobalInconsistency[0].value > 0.0:
            print("best all input: ", [float(x) for x in config.GlobalInputStr[0]], "inconsistency:", config.GlobalInconsistency[0].value, " (", config.GlobalMaximum[0].value, ",", config.GlobalMinimum[0].value, ")")

        csvSuffix = [""]
        for i in range(len(csvSuffix)):
            resultFilename = osHomePath + "/cigen/results/results" + config.ResultSuffix + csvSuffix[i] + ".csv"
            with open(resultFilename, "a") as f:
                f.write(os.path.basename(os.path.normpath(os.getcwd())) + "," \
                        + str(endTime - config.StartTime) + "," \
                        + str(ret[0]) + "," \
                        + str(ret[1]) + "," \
                        + str(ret[2]) + "," \
                        + str(config.ProgramCallCount.value) + "," \
                        + str(config.GlobalInconsistency[i].value) + "," \
                        + str(config.GlobalMaximum[i].value) + "," \
                        + str(config.GlobalMinimum[i].value) + "," \
                        + str(config.RandomSeed))
                if config.GlobalInconsistency[i].value > 0.0:
                    for inputStr in config.GlobalInputStr[i]:
                        f.write("," + inputStr)
                f.write("\n")

        plotx = [0]
        ploty = [0.0]
        with open(resultLogName, "r") as f:
            lines = csv.DictReader(f, delimiter=',')
            for line in lines:
                plotx.append(float(line["time"]))
                ploty.append(float(line["inc"]))
        plotx.append(endTime - config.StartTime)
        ploty.append(config.GlobalInconsistency[0].value)

        plt.plot(plotx, ploty, color='b')

        plt.xlabel('Time Elapsed') 
        plt.ylabel('Inconsistency Bits') 
        plt.title('Inconsistency Time-lapse', fontsize = 20) 
        plt.grid() 
        plt.show() 
        SaveFig(resultFigName)

        for i in range(3):
            config.GlobalInputStr[i].shm.close()
            config.GlobalInputStr[i].shm.unlink()

        TerminateInconsistencyParallelWorker()

    os.system("make clean")