import os
import time
import sys
import subprocess
import csv
import random

IconText = ["\\", "|", "/", "-"]

if __name__ == "__main__":
    cleanMode = False
    resultSuffix = ""
    skipArg = ""
    seedArg = ""
    plotArg = ""
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == "clean":
                cleanMode = True
            elif arg.startswith("-name="):
                resultSuffix = arg
            elif arg.startswith("-skip="):
                skipArg = arg
            elif arg.startswith("-seed="):
                seedArg = arg.split("=")[1]
            elif arg.startswith("-plot"):
                plotArg = arg

    # read seed file
    seedDict = {}
    if seedArg != "":
        if not seedArg.isnumeric():
            with open(seedArg, "r") as f:
                lines = f.readlines()
                for line in lines:
                    items = line.strip().split()
                    if len(items) == 2 and items[0][0].isdigit():
                        index = int(items[0].split("_")[0])
                        seed = int(items[1])
                        seedDict[index] = seed

    nextpair = next(os.walk('./'))
    root = nextpair[0]
    allDir = nextpair[1]
    total = 176

    numNames = [""] * total
    for index in range(1, total):
        for name in allDir:
            if name.startswith(str(index) + "_"):
                numNames[index] = name

    if os.path.exists("./times.txt"):
        with open("./times.txt", "r") as f:
            reader_object = csv.reader(f)
            for row in reader_object:
                times = row
    else:
        times = [0] * total
    times[0] = 30

    for i in range(30):
        os.system("mkdir -p ../results_" + str(i).zfill(2))

    while True:
        index = random.randint(1, total - 1)
        if times[index] < 30:
            name = numNames[index]
            print("running in " + name)      

            argList = ["python3", "../../driver/test_engine_cigen.py"]
            if resultSuffix != "":
                argList.append(resultSuffix)
            if skipArg != "":
                argList.append(skipArg)
            if plotArg != "":
                argList.append(plotArg)
            if seedArg.isnumeric():
                argList.append("-seed=" + seedArg)
            if index in seedDict:
                argList.append("-seed=" + str(seedDict[index]))
            os.system("mkdir -p ~/cigen/results/_" + resultSuffix.replace("-name=", ""))
            os.system("mkdir -p ~/cigen/results/_" + resultSuffix.replace("-name=", "") + "/" + name + "/")
            stderrName = os.path.expanduser("~") + "/cigen/results/_" + resultSuffix.replace("-name=", "") + "/" + name + "/stderr.txt"

            with open(stderrName, "w") as ferr: #open(stdoutName, "w") as fout, open(stderrName, "w") as ferr:
                p = subprocess.Popen(argList, cwd=os.path.join(root, name), stderr=ferr)
                timeCount = 0
                while p.poll() is None:
                    print(IconText[timeCount] + "================ heartbeat, running " + name + " (" + str(index) + "/" + str(total) + ") =================", end="\r")
                    time.sleep(0.2)
                    timeCount = timeCount + 1
                    if timeCount >= 4:
                        timeCount = 0
                ferr.flush()

            print("size:", os.path.getsize(stderrName))
            with open(stderrName, "r") as ferr:
                print(ferr.read())
            if os.path.getsize(stderrName) == 0:
                os.system("rm " + stderrName)       

            with open("../results/results_de.csv", "r") as f:
                topLine = f.readline().strip()
                resLine = f.readline().strip()
            with open("../results/results_de.csv", "w") as f:
                f.write(topLine + "\n")
            with open("../results_" + str(times[index]).zfill(2) + "/results_de.csv", "a") as f:
                f.write(resLine + "\n")
            os.system("cp -r ../results/_de/" + name + " ../results_" + str(times[index]).zfill(2))
            os.system("rm -r ../results/_de/" + name)

            times[index] += 1
            with open("./times.txt", "w") as f:
                writer_object = csv.writer(f)
                writer_object.writerow(times)

        if all(i >= 30 for i in times):
            break