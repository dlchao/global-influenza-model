# global_csv.py
# routines for producing text file output for global_rough.py

import csv
import numpy as np

# outputs numbers of infecteds, susceptibles, and recovereds to "filenamestem"i.csv, "filenamestem"s.csv, and "filenamestem"r.csv  for globalList "gl". Also outputs exchangelists.
# "filenamestem"r4.csv contains recovereds, but the first two numbers indicate 0/1 for subpopulation and 0/1 for risk group
def write_global(gl, filenamestem, splitpop=False):
    # write args file
    fargs = open(filenamestem + "args.csv", 'wt') # arguments
    argWriter = csv.writer(fargs, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    if hasattr(gl, '__class__'):
        argWriter.writerow(["class " + gl.__class__.__name__])
    argWriter.writerow(["name " + gl.name])
    argWriter.writerow(["R0 %1.2f" % gl.R0])
    argWriter.writerow(["lowR0 %1.2f" % gl.lowR0])
    argWriter.writerow(["sympTratio %1.2f" % gl.sympTratio])
    argWriter.writerow(["psymp %1.2f" % gl.psymp])
    argWriter.writerow(["m %1.2f" % gl.m])
    argWriter.writerow(["level %1.2f" % gl.level])
    argWriter.writerow(["theta1 %1.2f" % gl.theta1])
    argWriter.writerow(["theta2 %1.2f" % gl.theta2])
    argWriter.writerow(["VEmaxS %1.2f" % gl.VEmaxS])
    argWriter.writerow(["VEmaxI %1.2f" % gl.VEmaxI])
    argWriter.writerow(["VEmaxP %1.2f" % gl.VEmaxP])
    argWriter.writerow(["starttime %d" % gl.tlist[0]])
    if gl.randomSeed==None:
        argWriter.writerow(["randomseed None"])
    else:
        argWriter.writerow(["randomseed %d" % gl.randomSeed])
    argWriter.writerow(["popfile " + gl.popFile])
    argWriter.writerow(["travelfile " + gl.travelFile])
    fargs.close()

    # write SIR files
    fi = open(filenamestem + "i.csv", 'wt') # infected
    fs = open(filenamestem + "s.csv", 'wt') # susceptible
    fr = open(filenamestem + "r.csv", 'wt') # recovered
    fr4 = open(filenamestem + "r4.csv", 'wt') # recovered
    iWriter = csv.writer(fi, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    sWriter = csv.writer(fs, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    rWriter = csv.writer(fr, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    r4Writer = csv.writer(fr4, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for citynum, cityname in enumerate(gl.cityList):
        if splitpop:
            print gl.Cities[cityname].Ilist[0]
            print gl.Cities[cityname].Slist[0]
            print gl.Cities[cityname].Rlist[0]
            for i in xrange(2): # subpop
                temp = [pair[i] for pair in gl.Cities[cityname].Ilist]
                iWriter.writerow([cityname] + [i] + temp)
                for j in xrange(2): # risk group
                    temp = [pair[i,j] for pair in gl.Cities[cityname].Slist]
                    sWriter.writerow([cityname] + [i,j] + temp)
                    temp = [pair[i,j] for pair in gl.Cities[cityname].Rlist]
                    rWriter.writerow([cityname] + [i,j] + temp)
        else:
            temp = [np.sum(pair) for pair in gl.Cities[cityname].Ilist]
            iWriter.writerow([cityname] + temp)
            temp = [np.sum(pair) for pair in gl.Cities[cityname].Slist]
            sWriter.writerow([cityname] + temp)
            temp = [np.sum(pair) for pair in gl.Cities[cityname].Rlist]
            rWriter.writerow([cityname] + temp)
            for i in xrange(2): # subpop
                for j in xrange(2): # risk group
                    temp = [sum(pair[i,j]) for pair in gl.Cities[cityname].Rlist]
                    r4Writer.writerow([cityname] + [i,j] + temp)

    fi.close()
    fs.close()
    fr.close()
    fr4.close()

    if gl.exchangeArrayList:
        # write exchange file
        fe = open(filenamestem + "exchange.csv", 'wt')
        eWriter = csv.writer(fe, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for citynum, cityname in enumerate(gl.cityList):
            for day in xrange(0, len(gl.exchangeList)):
                temp = []
                temp.append(day)
                for e in gl.exchangeList[day][citynum]:
                    temp.append(e)
                eWriter.writerow(temp)
        fe.close()
