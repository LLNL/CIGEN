#!/usr/bin/env python3
from cmath import nan
import os
import math
import subprocess
import time
from posixpath import split
import multiprocessing as mp
from multiprocessing import shared_memory
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
import csv
import matplotlib.pyplot as plt
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

# 0 - free, 1 - occupied, 2 - done
WorkerProcess = []
IsWorkerOccupied = []
WorkerX = []
WorkerXStr = []
WorkerGen = []
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

def TerminateInconsistencyParallelWorker():
    TerminateWorkers.value = 1
    for i, process in enumerate(WorkerProcess):
        process.join()
        WorkerX[i].shm.close()
        WorkerX[i].shm.unlink()
        WorkerResults[i].shm.close()
        WorkerResults[i].shm.unlink()

def CreateInconsistencyParallelWorkers(size):
    global NumJobs, WorkerProcess, IsWorkerOccupied, WorkerX, WorkerResults, TerminateWorkers, WorkerXStr
    if NumJobs == -1:
        NumJobs = mp.cpu_count()
    
    TerminateWorkers = mp.Value('i', 0)

    for i in range(NumJobs):
        IsWorkerOccupied.append(mp.Value('i', 0))
        WorkerX.append(shared_memory.ShareableList(["x" * 8] * size))
        WorkerXStr.append([""] * size)
        WorkerGen.append([])
        WorkerResults.append(shared_memory.ShareableList(["x" * 64] * 3))
        p = mp.Process(target=InconsistencyParallelWorker, args=(i,))
        WorkerProcess.append(p)
        p.start()

def GetInconsistency(x):
    inc, maximum, minimum = GetInconsistencyParallelJob(x, None, None)
    return inc

def GetInconsistencyParallelJob(x, queue, index):
    with config.ProgramCallCount.get_lock():
        currentCallCount = config.ProgramCallCount.value 
        config.ProgramCallCount.value += 1
    if config.OptimizerMode:
        temp = []
        for i1, i2 in zip(x, config.Base):
            temp.append(MultiplyPow2(i1, i2))
        x = temp
        for i in range(len(config.Base)):
            if x[i] > config.RangeLimit[1][i]:
                x[i] = config.RangeLimit[1][i]
            if x[i] < config.RangeLimit[0][i]:
                x[i] = config.RangeLimit[0][i]
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
        timeout, resolved, config.Minimum, config.Maximum, errorOccurred, diffInTrace = te.TestInput(config.DriverPath, config.FpId, dirName, argChanged)
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
    elif logAbsResolved > config.GlobalInconsistency.value:
        with config.GlobalInconsistency.get_lock():
            for index, item in enumerate(argChanged):
                config.GlobalInputStr[index] = item
            config.GlobalMaximum.value = float(config.Maximum)
            config.GlobalMinimum.value = float(config.Minimum)
            config.GlobalInconsistency.value = logAbsResolved
        elapsedTime = time.time() - config.StartTime
        with open(resultLogName, 'a') as f:
            print(elapsedTime, ",", config.GlobalInconsistency.value, \
                ",", config.GlobalMaximum.value, ",", config.GlobalMinimum.value, \
                ",", config.GlobalInputStr, file=f, sep='')
        print(str(currentCallCount) + " (" + str(diffInTrace) + "): set input " + str(config.VarList) + " to " + str(argChanged) + " : inconsistency " + str(logAbsResolved) + " (" + str(config.Minimum) + ", " + str(config.Maximum) + ")")    
    elif currentCallCount % 50 == 0:
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
                print(elapsedTime, ",", config.GlobalInconsistency.value, \
                    ",", config.GlobalMaximum.value, ",", config.GlobalMinimum.value, \
                    ",", config.GlobalInputStr, file=f, sep='')
    if queue:
        queue.put([inc, float(config.Maximum), float(config.Minimum), diffInTrace])
    return inc, float(config.Maximum), float(config.Minimum)

def GetRandomFloat(fpId, minexponent, minmantissa, maxexponent, maxmantissa, sign):
    if fpId == "fp32":
        numInExponent = 2**23
    else:
        numInExponent = 2**52
    totalNum = (maxexponent - minexponent) * numInExponent + (maxmantissa - minmantissa)
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

def mcmc_callback(x, f, accept):
    currentTime = time.time()
    if currentTime - config.StartTime > config.TimeLimit * (2**len(x)):
        return True
    return False

def MonteCarloMC(inputRange):
    config.Base = inputRange[0]
    config.RangeLimit = inputRange
    print("config.Base = ", config.Base)
    print("limit = ", inputRange[1])
    initialGuess = []
    for i in range(len(config.Base)):
        initialGuess.append(float(config.ConvertFloat(inputRange[2][3][i])))
    #initialGuess = [(math.log2(abs(i1)) - math.log2(abs(i2))) for i1, i2 in zip(initialGuess, config.Base)]
    print("initial guess:", initialGuess)
    #config.OptimizerMode = True
    result = scipy.optimize.basinhopping(GetInconsistency, initialGuess, niter=300, niter_success=20, minimizer_kwargs={'method':'Powell'}, callback=mcmc_callback)
    #config.OptimizerMode = False        
    maxPoint = [MultiplyPow2(i1, i2) for i1, i2 in zip(result.x, config.Base)]
    print("max point: ", maxPoint, " inconsistency bits: ", result.fun)
    
gEdgeValues32 = [(0, 254, (2**23)-1), (1, 254, (2**23)-1), (0, 0, (2**23)-1), (1, 0, (2**23)-1)]
gEdgeValues64 = [(0, 2046, (2**52)-1), (1, 2046, (2**52)-1), (0, 0, (2**52)-1), (1, 0, (2**52)-1)]

gMultiRange32 = [0, 2, 94, 125, 128, 131, 161, 252, 254, 256]
gMultiRange64 = [0, 692, 991, 1024, 1057, 1356, 2043, 2048]

gMultiRange_Random32 = [0, 256]
gMultiRange_Random64 = [0, 2048]

def BinOneRange(currentSign, currentMinExponent, currentMaxExponent, useUp):
    if useUp != 0 and useUp != 1:
        return currentSign, currentMinExponent, currentMaxExponent
    if config.FpId == "fp32":
        minPossibleExp = -149.0
    else:
        minPossibleExp = -1074.0
    if currentSign == 2:
        retSign = 0 if useUp else 1
        retMinExp = minPossibleExp
        retMaxExp = currentMaxExponent if useUp else currentMinExponent
    else:
        averageExp = (currentMinExponent + currentMaxExponent) / 2
        retSign = currentSign
        retMinExp = averageExp if useUp else currentMinExponent
        retMaxExp = currentMaxExponent if useUp else averageExp
    return retSign, retMinExp, retMaxExp

def BinRanges(currentSign, currentMinExponent, currentMaxExponent, useUp):
    retSign = []
    retMinExp = []
    retMaxExp = []
    for i in range(len(currentSign)):
        rSign, rMinExp, rMaxExp = BinOneRange(currentSign[i], currentMinExponent[i], currentMaxExponent[i], useUp[i])
        retSign.append(rSign)
        retMinExp.append(rMinExp)
        retMaxExp.append(rMaxExp)
    return retSign, retMinExp, retMaxExp

def BinRanges_NoLog(currentRange, useUp):
    newRange = []
    for i in range(len(currentRange)):
        if False: #(currentRange[i][0] <= 0.0 and currentRange[i][1] >= 0.0):
            if useUp[i]:
                newRange.append([currentRange[i][0], -config.minDenormalFloat])
            else:
                newRange.append([config.minDenormalFloat, currentRange[i][1]])
        else:
            mid = currentRange[i][0] * 0.5 + currentRange[i][1] * 0.5
            if useUp[i]:
                newRange.append([mid, currentRange[i][1]])
            else:
                newRange.append([currentRange[i][0], mid])
    return newRange

def BinaryGuidedRandomTesting(maxexponent, numInExponent, varList, exponentList, mantissaList):
    numVariables = len(exponentList) 
    limitMinSign = []
    limitMaxSign = []
    limitMinExponent = []
    limitMaxExponent = []
    currentSign = []
    currentMinExponent = []
    currentMaxExponent = []
    all_inc = []
    zero_inc = []
    if config.NoLogMode:
        currentGen = config.TotalLimit
    else:
        for idx in range(0, numVariables):
            limitMinSign.append(0 if config.TotalLimit[idx][0] > 0.0 else 1)
            limitMaxSign.append(0 if config.TotalLimit[idx][1] > 0.0 else 1)
            limitMinExponent.append(math.log2(abs(config.TotalLimit[idx][0])))
            limitMaxExponent.append(math.log2(abs(config.TotalLimit[idx][0])))
            if limitMinSign[idx] != limitMaxSign[idx]:
                currentSign.append(2)
            else:
                currentSign.append(limitMinSign[idx])
        currentMinExponent = limitMinExponent
        currentMaxExponent = limitMaxExponent
    npart = 1
    k = 10
    parallelBatch = 6
    terminate = False

    while terminate == False: # exit when timeout
        nextGen = []
        # first generate perfect split of current range
        if config.NoLogMode:
            useUp = [0] * numVariables        
            nextGen.append(BinRanges_NoLog(currentGen, useUp))
            useUp = [1] * numVariables        
            nextGen.append(BinRanges_NoLog(currentGen, useUp))
            for i in range(0, npart):
                useUp = []
                for j in range(numVariables):
                    useUp.append(random.randint(0, 1))
                nextGen.append(BinRanges_NoLog(currentGen, useUp))
                for j in range(numVariables):
                    useUp[j] = 1 - useUp[j]
                nextGen.append(BinRanges_NoLog(currentGen, useUp))
        else:
            useUp = [0] * numVariables
            nextGen.append(BinRanges(currentSign, currentMinExponent, currentMaxExponent, useUp))
            useUp = [1] * numVariables
            nextGen.append(BinRanges(currentSign, currentMinExponent, currentMaxExponent, useUp))
            for i in range(0, npart):
                useUp = []
                for j in range(numVariables):
                    useUp.append(random.randint(0, 1))
                nextGen.append(BinRanges(currentSign, currentMinExponent, currentMaxExponent, useUp))
                for j in range(numVariables):
                    useUp[j] = 1 - useUp[j]
                nextGen.append(BinRanges(currentSign, currentMinExponent, currentMaxExponent, useUp))
        localInc = 0.0
        localGen = []
        SignSets = [ [] for _ in range(NumJobs)]
        Gens = [ [] for _ in range(NumJobs)]
        for genIndex, gen in enumerate(nextGen):
            for i in range(k):
                inputValues = []
                inputValueStr = []
                for j in range(numVariables):
                    if config.NoLogMode:
                        #exp0, mant0 = te.GetParts(gen[j][0], config.FpId)
                        #exp1, mant1 = te.GetParts(gen[j][1], config.FpId)
                        #if gen[j][0] < 0.0:
                        #    outputNum, _, _ = GetRandomFloat(config.FpId, exp1, mant1, exp0, mant0, 1)
                        #else:
                        #    outputNum, _, _ = GetRandomFloat(config.FpId, exp0, mant0, exp1, mant1, 0)
                        r = random.uniform(0, 1)
                        outputNum = (1 - r) * gen[j][0] + r * gen[j][1]
                        outputValue = outputNum
                        output = str(outputNum)
                        if config.FpId == "fp32":
                            output = str(outputNum) + "f"
                    else:
                        randInput = random.uniform(gen[1][j], gen[2][j])
                        outputNum = (1 - 2 * gen[0][j]) * math.pow(2.0, randInput)
                        output = str(outputNum)
                        if config.FpId == "fp32":
                            output = str(outputNum) + "f"
                        outputValue = float(config.ConvertFloat(output.strip()))
                    inputValues.append(outputValue)
                    inputValueStr.append(output)

                # clean up for a free slot in job system
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
                            if config.NoLogMode:
                                signSet = [0] * numVariables
                            else:
                                signSet = SignSets[i]
                            inc = (signSet, exponentSet, mantissaSet, WorkerXStr[i], float(minimum), float(maximum), resolved)
                            if abs(resolved) > 1.0:
                                all_inc.append(inc)
                            else:
                                zero_inc.append(inc)
                            if localInc < abs(resolved) or localInc == 0.0:
                                localInc = resolved
                                localGen = Gens[i]
                            IsWorkerOccupied[i].value = 0
                    if HasFreeSlots == False:
                        time.sleep(0.001)

                # insert into the job system
                for i in range(len(IsWorkerOccupied)):
                    if IsWorkerOccupied[i].value == 0:
                        for j, xd in enumerate(inputValues):
                            WorkerX[i][j] = xd
                        if not config.NoLogMode:
                            SignSets[i] = currentSign
                        WorkerXStr[i] = inputValueStr
                        Gens[i] = gen
                        IsWorkerOccupied[i].value = 1
                        break

        # clean up all free slots
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
                    if config.NoLogMode:
                        signSet = [0] * numVariables
                    else:
                        signSet = SignSets[i]
                    inc = (signSet, exponentSet, mantissaSet, WorkerXStr[i], float(minimum), float(maximum), resolved)
                    if abs(resolved) > 1.0:
                        all_inc.append(inc)
                    else:
                        zero_inc.append(inc)
                    if localInc < abs(resolved) or localInc == 0.0:
                        localInc = resolved
                        localGen = Gens[i]
                    IsWorkerOccupied[i].value = 0
                if IsWorkerOccupied[i].value != 0:
                    HasOccupiedSlots = True 
            if HasOccupiedSlots:
                time.sleep(0.001)

        if config.NoLogMode:
            if random.uniform(0, 1) > 0.8:
                print("====== reset")   
                currentGen = config.TotalLimit     
            else:
                currentGen = localGen
        else:
            if random.uniform(0, 1) > 0.8:
                print("====== reset")
                currentSign = []
                for idx in range(0, numVariables):
                    if limitMinSign[idx] != limitMaxSign[idx]:
                        currentSign.append(2)
                    else:
                        currentSign.append(limitMinSign[idx])
                currentMinExponent = limitMinExponent
                currentMaxExponent = limitMaxExponent            
            else:
                currentSign = localGen[0]
                currentMinExponent = localGen[1]
                currentMaxExponent = localGen[2]
        if time.time() - config.StartTime > config.TimeLimit * (2**numVariables):
            terminate = True
    return all_inc, zero_inc

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
    bucket_inc = None
    bucket_all_inc = []
    numVariables = len(exponentList) 
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
        if config.RandMode:
            multiRange = gMultiRange_Random32.copy()
        else:
            multiRange = gMultiRange32.copy()
        baseExp = 128
    else:
        if config.RandMode:
            multiRange = gMultiRange_Random64.copy()
        else:
            multiRange = gMultiRange64.copy()
        baseExp = 1024

    sample = 0
    if config.RandMode:
        bucketSize = sys.maxsize
    else:
        bucketSize = 81
    terminate = False
    cleanUp = False
    prevInputAreaList = [0] * numVariables
    SignSets = [ [] for _ in range(NumJobs)]
    while terminate == False:
        currentSample = sample // bucketSize
        inputValues = []
        inputValueStr = []
        signSet = []
        exponentSet = []
        mantissaSet = []
        inputAreaList = [0] * numVariables

        for idx in range(numVariables):
            totalLimit0 = config.TotalLimit[idx][0]
            totalLimit1 = config.TotalLimit[idx][1]
            if config.TotalLimit[idx][0] == 0:
                totalLimit0 = config.minDenormalFloat
            if config.TotalLimit[idx][1] == 0:
                totalLimit1 = -config.minDenormalFloat                    
            if limitMinSign[idx] != limitMaxSign[idx]:
                inputArea = currentSample % 2
                currentSample = (currentSample - inputArea) // 2
                if config.RandMode:
                    sign = random.randint(0, 1)
                else:
                    sign = inputArea
                startRange = 0
                endRange = 0
                if sign == 1:
                    limitMax = math.log2(abs(totalLimit0)) + baseExp
                else:
                    limitMax = math.log2(abs(totalLimit1)) + baseExp                                         
                for i in range(len(multiRange)):
                    if limitMax < multiRange[i]:
                        endRange = i
                        break
                    if i == len(multiRange) - 1:
                        endRange = i - 1
                        break
            else:
                sign = limitMinSign[idx]
                startRange = 0
                endRange = 0
                limitMin = math.log2(abs(totalLimit0)) + baseExp
                limitMax = math.log2(abs(totalLimit1)) + baseExp   
                for i in range(len(multiRange)):
                    if limitMin > multiRange[i] or (i == 0 and limitMin <= 0):
                        startRange = i
                    if limitMax < multiRange[i]:
                        endRange = i
                        break
                    if i == len(multiRange) - 1:
                        endRange = i - 1
                        break

            inputArea = currentSample % (endRange - startRange + 1)
            inputAreaList[idx] = inputArea
            currentSample = (currentSample - inputArea) // (endRange - startRange + 1)
            if idx == numVariables - 1 and currentSample > 0:
                terminate = True
                break
            if multiRange[inputArea] < limitMax:
                abandoned = False
                while True:
                    f, _, _ = GetRandomFloat(config.FpId, multiRange[inputArea], 0, multiRange[inputArea + 1], 0, sign)
                    randInput = math.log2(abs(f)) + baseExp
                    #randInput = random.uniform(multiRange[inputArea], multiRange[inputArea + 1])
                    if randInput < limitMax:
                        break           
                outputNum = f     
                #outputNum = (1 - 2 * sign) * math.pow(2.0, randInput - baseExp)
                if config.RandMode and config.NoLogMode:
                    r = random.uniform(0, 1)
                    outputNum = (1 - r) * config.TotalLimit[idx][0] + r * config.TotalLimit[idx][1]
                    outputValue = outputNum
                    output = str(outputNum)
                    if config.FpId == "fp32":
                        output = str(outputNum) + "f"
                else:
                    output = str(outputNum)
                    if config.FpId == "fp32":
                        output = str(outputNum) + "f"
                    outputValue = float(config.ConvertFloat(output.strip()))
                inputValues.append(outputValue)
                inputValueStr.append(output.strip())
                signSet.append(sign)
            else:
                abandoned = True
        
        if abandoned:
            prevInputAreaList = inputAreaList
            sample += 1
            continue  

        if time.time() - config.StartTime > config.TimeLimit * (2**numVariables):
            terminate = True
        if prevInputAreaList != inputAreaList or terminate:
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
                            if config.BucketMode:
                                if bucket_inc == None:
                                    bucket_inc = inc
                                elif bucket_inc[6] < resolved:
                                    bucket_inc = inc
                        else:
                            zero_inc.append(inc)
                        IsWorkerOccupied[i].value = 0
                    if IsWorkerOccupied[i].value != 0:
                        HasOccupiedSlots = True
                if HasOccupiedSlots:
                    time.sleep(0.001)
            if config.BucketMode and bucket_inc:
                bucket_all_inc.append(bucket_inc)
                bucket_inc = None

            cleanUp = False

        if terminate:
            break

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
                        if config.BucketMode:
                            if bucket_inc == None:
                                bucket_inc = inc
                            elif bucket_inc[6] < resolved:
                                bucket_inc = inc
                    else:
                        zero_inc.append(inc)
                    IsWorkerOccupied[i].value = 0
            if HasFreeSlots == False:
                time.sleep(0.001)

        for i in range(len(IsWorkerOccupied)):
            if IsWorkerOccupied[i].value == 0:
                for j, xd in enumerate(inputValues):
                    WorkerX[i][j] = xd
                SignSets[i] = signSet
                WorkerXStr[i] = inputValueStr
                IsWorkerOccupied[i].value = 1

        prevInputAreaList = inputAreaList
        sample += 1
    if config.BucketMode:
        return bucket_all_inc, zero_inc
    else:
        return all_inc, zero_inc

def MutateRoutine(driverPath, argList, varSet, limitSet):
    varindex = 0 
    mantissaList = []
    exponentList = []
    varList = []
    config.TotalLimit = limitSet
    for idx, i in enumerate(varSet):    
        if "f" in argList[i].lower():
            fpId = "fp32"
            config.FpId = fpId
            betweenId = "between32"
            _, exponent, mantissa = GetFloatStrComponents(argList[i])
            maxexponent = 256
            numInExponent = 2**23        
        elif "." in argList[i].lower() or "e" in argList[i].lower():
            fpId = "fp64"
            config.FpId = fpId
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

    config.SetMinDenormalFloat(config.FpId)
    numVariables = len(exponentList)
    te.BuildExe(driverPath, fpId)
    config.VarList = varList
    CreateInconsistencyParallelWorkers(len(argList))
    if config.BinaryMode:
        all_inc, zero_inc = BinaryGuidedRandomTesting(maxexponent, numInExponent, varList, exponentList, mantissaList)
        ret = [len(all_inc) + len(zero_inc), len(all_inc), config.GlobalInconsistency.value]
    else:
        all_inc, zero_inc = RandomSampling(maxexponent, numInExponent, varList, exponentList, mantissaList)
        ret = [len(all_inc) + len(zero_inc), len(all_inc), config.GlobalInconsistency.value]
        if not config.RandMode:
            print(*all_inc, sep='\n')
            processList = []
            for inc in all_inc:
                inputRange = []
                inputRange.append([])
                inputRange.append([])
                for i in range(numVariables):
                    if inc[0][i] == 0:
                        inputRange[0].append(config.minDenormalFloat)
                        inputRange[1].append(limitSet[i][1])
                    else:
                        inputRange[0].append(limitSet[i][0])
                        inputRange[1].append(-config.minDenormalFloat)
                inputRange.append(inc)
                print("run MCMC for:", inc)
                p = mp.Process(target=MonteCarloMC, args=(inputRange,))
                while (len(processList)) >= 12:
                    newProcessList = []
                    for proc in processList:
                        if proc.is_alive():
                            newProcessList.append(proc)
                    processList = newProcessList
                processList.append(p)
                p.start()
                inputRange.pop(2)
            for p in processList:
                p.join()
    return ret

if __name__ == '__main__':
    driverPath = os.path.dirname(os.path.realpath(__file__))
    currentdir = os.getcwd()
    
    os.system("make clean")

    for compiler in config.Compiler:
        if compiler[0] == "nvcc":
            x = subprocess.check_output(['nvidia-smi']).decode("utf-8") 
            if "Failed" in x or "Unknown Error" in x:
                exit(0)

    config.StartTime = time.time()
    
    config.RandomSeed = int(datetime.now().timestamp()) % 300000
    gTestOnly = False
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("-seed="):
                config.RandomSeed = int(arg.replace("-seed=", ""))
            if arg.startswith("-skip="):
                config.SkipRandom = arg.replace("-skip=", "") == "1"
            if arg.startswith("-rand"):
                config.RandMode = True
                config.ResultSuffix = "_rand"
            if arg.startswith("-bgrt"):
                config.BinaryMode = True
                config.ResultSuffix = "_bgrt"
            if arg.startswith("-time="):
                config.TimeLimit = float(arg.replace("-time=", ""))
            if arg.startswith("-testonly"):
                gTestOnly = True
            if arg.startswith("-jobs="):
                NumJobs = int(arg.replace("-jobs=", ""))
    
    config.NoLogMode = True

    os.system("mkdir -p /root/cigen/results/" + config.ResultSuffix)
    resultLogName = "/root/cigen/results/" + config.ResultSuffix + "/" + os.path.basename(currentdir) + ".csv"
    resultFigName = "/root/cigen/results/" + config.ResultSuffix + "/" + os.path.basename(currentdir) + ".png"

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
        config.InitializeConfig(len(argList))
        ret = MutateRoutine(driverPath, argList, varSet, limitSet)

        endTime = time.time()
        print("=== results ===: seed", config.RandomSeed, "time:", endTime - config.StartTime, "callcount:", config.ProgramCallCount.value)
        print("callcount_random, num_random, inc_random =", ret[0], ret[1], ret[2])
        if config.GlobalInconsistency.value > 0.0:
            print("input: ", [float(x) for x in config.GlobalInputStr], "inconsistency:", config.GlobalInconsistency.value, " (", config.GlobalMaximum.value, ",", config.GlobalMinimum.value, ")")
        if config.BinaryMode:
            suffix = "_bgrt"
        elif config.RandMode:
            suffix = "_rand"
        else:
            suffix = "_civ"
        resultFilename = "/root/cigen/results/results" + suffix + ".csv"
        with open(resultFilename, "a") as f:
            f.write(os.path.basename(os.path.normpath(os.getcwd())) + "," \
                    + str(endTime - config.StartTime) + "," \
                    + str(ret[0]) + "," \
                    + str(ret[1]) + "," \
                    + str(ret[2]) + "," \
                    + str(config.ProgramCallCount.value) + "," \
                    + str(config.GlobalInconsistency.value) + "," \
                    + str(config.GlobalMaximum.value) + "," \
                    + str(config.GlobalMinimum.value) + "," \
                    + str(config.RandomSeed))
            if config.GlobalInconsistency.value > 0.0:
                for str in config.GlobalInputStr:
                    f.write("," + str)
            f.write("\n")

        plotx = [0]
        ploty = [0.0]
        with open(resultLogName, "r") as f:
            lines = csv.DictReader(f, delimiter=',')
            for line in lines:
                plotx.append(float(line["time"]))
                ploty.append(float(line["inc"]))
        plotx.append(endTime - config.StartTime)
        ploty.append(config.GlobalInconsistency.value)

        plt.plot(plotx, ploty, color='b')

        plt.xlabel('Time Elapsed') 
        plt.ylabel('Inconsistency Bits') 
        plt.title('Inconsistency Time-lapse', fontsize = 20) 
        plt.grid() 
        plt.legend() 
        plt.show() 
        plt.savefig(resultFigName)

        config.GlobalInputStr.shm.close()
        config.GlobalInputStr.shm.unlink()

        TerminateInconsistencyParallelWorker()