import os
import time
import sys
import subprocess

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

    for index in range(1, total):
        for name in allDir:
            if name.startswith(str(index) + "_"):
                print("running in " + name)

                if cleanMode:
                    origPath = os.getcwd()
                    os.chdir(os.path.join(root, name))
                    os.system("make clean")
                    os.chdir(origPath)         
                else:          
                    if resultSuffix == "-name=de":
                        argList = ["python3", "../../driver/test_engine_cigen.py"]
                        if plotArg != "":
                            argList.append(plotArg)                        
                    elif resultSuffix == "-name=civ":
                        argList = ["python3", "../../driver_civ/test_engine_civ.py"]
                    else:
                        argList = ["python3", "../../driver_civ/test_engine_civ.py", "-bgrt"]
                    if skipArg != "":
                        argList.append(skipArg)
                    if seedArg.isnumeric():
                        argList.append("-seed=" + seedArg)
                    if index in seedDict:
                        argList.append("-seed=" + str(seedDict[index]))
                    os.system("mkdir -p ~/cigen/results/_" + resultSuffix.replace("-name=", ""))
                    os.system("mkdir -p ~/cigen/results/_" + resultSuffix.replace("-name=", "") + "/" + name + "/")
                    #stdoutName = "~/cigen/results/_" + resultSuffix.replace("-name=", "") + "/" + name + "/stdout.txt"
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