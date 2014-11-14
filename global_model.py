# Global epidemic model
# by Eben Kenah and Dennis Chao
# University of Washington and Fred Hutchinson Cancer Research Center

# Imports
# Standard library
import math
import random as rand
from collections import deque
from itertools import izip
# Third-party
import numpy as np

# define binomial function that handles n = 0 without producing an error
def zbinomial(n, p):
    if n > 0:
        return np.random.binomial(n, p)
    else:
        return 0
# allow zbinomial to be applied element-by-element to an array
zbinomial = np.vectorize(zbinomial)

class epiList:
    """
    The epiList for each city allows importation of infection, vaccination, 
    calculation of vaccine efficacy and transmission probabilities, and 
    sending infected persons to other cities.  It records the progress of an 
    epidemic and keeps records of vaccinations and importations.  In S, incI,
    and R, vaccination status 

    Data attributes:
	City characteristics:
        name -- city name
        destList -- list of cities to which persons can travel; cities may be
                    designated by name or number
        destProbs -- list of asymptomatic probability of traveling to each 
		     destination in a single time step; last entry is 
		     probability of remaining
	sympTratio -- the ratio of travel probabilities for symptomatics to 
		      those of asymptomatics; between 0 and 1
	propArray -- NumPy array with propArray[i, j] = proportion of population
		     in subpopulation i and risk group j
	popArray -- NumPy array with popArray[i, j] = number of persons in
		    subpopulation i and risk group j

	Transmission parameters:
	R0 -- peak seasonal R0
	NGmatrix -- peak seasonal next-generation matrix
        betaMatrix -- betaMatrix[i, j] = beta for transmission from an infected
		      in subpopulation i to a susceptible in subpopulation j
	psymp -- probability of becoming symptomatic
 	m -- asymptomatic/symptomatic infectiousness
	level -- maximum vaccine efficacy ratio achieved after first dose
	theta1, theta2 -- govern the shape of the VE ratio increase after first
			  and second dose, respectively; increase is concave 
			  for theta <= 0 and convex for theta >= 0
	VEmaxS, VEmaxI, VEmaxP -- maximum vaccine efficacy for susceptibility, 
				  infectiousness, and progression, respectively       
	
	Epidemic variables:
	R0ratio -- proportion of peak seasonal R0 attained at present
	S -- NumPy array S[i, j, k] = number of susceptible residents in
	     subpopulation i, risk group j, vaccination status k
        Sdict -- dictionary where Sdict[vtime][i, j] is the number of 
		 susceptibles in subpopulation i and risk group j 
		 vaccinated at t = vtime; vtime = None for the unvaccinated
		 and vtime = 'Past' for those vaccinated >= 28 days ago
	incI -- NumPy array incI[i, j, k] = number of new infections in
		subpopulation i, risk group j, vaccination status k
        I -- NumPy vector I[i] = number of infected persons (visitors and 
	     residents) in subpopulation i
        Idict -- dictionary where Idict[vtime][i,j,v,d] = number of infected 
		 persons vaccinated at t = vtime in subpopulation i with
		 symptom status j, viral load trajectory v, infection day d+1;
		 vtime = None for the unvaccinated and vtime = 'Past' for
		 those vaccinated >= 28 days ago
        R -- NumPy array R[i, j, k] = number of recovered residents in 
	     subpopulation i, risk group j, and vaccination status k
        Rq[d] -- FIFO queue where Rq[d][i, j, k] = number of infected residents
		 in d+1^st day of illness from subpopulation i, risk group j, 
		 and vaccination status k

	Epidemic data storage:
	vacList -- list (t, vacfracArray) for each t on which persons were
		   vaccinated
	imprtList -- list (t, impnum) for each t on which imported infections
		     occurred
	tlist -- list of times for which S, incI, I, and R were recorded
	Slist -- list of S for each time in tlist
	incIlist -- list of incI for each time in tlist
	Ilist -- list of I for each time in tlist
	Rlist -- list of R for each time in tlist
     
    Methods:
        VEratio -- calculate vaccine efficacy ratio (for vaccinate)
        vaccinate -- vaccination
        imprt -- import infections
        infP -- calculate infection probabilities (for increment)
        increment -- advance within-city model by one time step
        send -- send infected individuals to other cities
    Each method is explained further in its own docstring.
    
    """
    # viral load trajectories
    VLtraj = np.array([ 
        [2.0, 5.5, 4.0, 5.5, 3.0, 0.5],
        [1.0, 6.0, 3.0, 1.5, 3.5, 1.3],
        [2.5, 5.0, 5.0, 3.0, 5.5, 3.5],
        [3.5, 5.5, 6.5, 5.5, 3.5, 4.0],
        [2.5, 3.0, 6.5, 6.5, 2.0, 0.8],
        [4.0, 5.0, 5.5, 7.5, 5.5, 1.3]
        ])
    
    def __init__(self, 
		 population, propList=[1], riskProps=[[1]],
		 name='City', latitude=None, longitude=None,
		 destList=None, destProbs=None, sympTratio=1,
		 R0=1.4, NGmatrix=None, betaMatrix = None, 
		 psymp = 2. / 3,    # probability of becoming symptomatic
		 m = .5,            # asymptomatic/symptomatic infectiousness
		 level = .5,	    # vaccine efficacy parameters
		 theta1 = 0, theta2 = 0,
		 VEmaxS = .4, VEmaxI = .4, VEmaxP = .667):
        """
        Initializes epiList.  All arguments are saved as epiList attributes.  
	For arguments or sublists of arguments that must sum to one, the last 
	element need not be specified.

        Arguments:
	    City characteristics: 
            name -- the name of the city
	    index -- position of city in list of cities in globalModel
            population -- population size
            propList -- list of proportion of population in each subpopulation;
                        all subpopulations must have size > 0
	    riskProps -- nested list, where each list has proportion of
			 the corresponding subpopulation in each risk group; 
			 all subpopulations must have the same number of risk 
			 groups but risk groups can have size zero
	    latitude, longitude -- latitude and longitude of city
            destList -- list of possible destinations
            destProbs -- list of the asymptomatic probability of traveling to 
			 each destination in destList in a single time step; 
			 the last entry is the probability of remaining
	    sympTratio -- the ratio of travel probabilities for symptomatics to 
			  those of asymptomatics; between 0 and 1

	    Transmission parameters:
            R0 -- desired within-city R0; if R0 == None, then R0 equals the
                  spectral radius of the next-generation matrix
            NGmatrix -- NumPy array for next-generation matrix; if R0 != None,
                        it will be multiplied by a scalar to obtain desired R0.
	    betaMatrix -- NumPy array for peak seasonal betas; overridden by NGmatrix
			  if both are specified.
	    psymp -- the probability of being symptomatic
	    m -- asymptomatic/symptomatic infectiousness
	    level -- maximum vaccine efficacy ratio achieved after first dose
	    theta1, theta2 -- govern the shape of the VE ratio increase after
			      first and second dose, respectively; increase is
			      concave for theta <= 0 and convex for theta >= 0
	    VEmaxS, VEmaxI, VEmaxP -- maximum vaccine efficacy for
				      susceptibility, infectiousness, and
				      progression, respectively

        """
        ## Dynamic data attributes
        # city characteristics
        self.name = name
        self.latitude = latitude
        self.longitude = longitude

	# population
	self.population = population
	self.propList = propList
	self.riskProps = riskProps
	if sum(propList) < 1:
	    propList.append(1 - sum(propList))
	propList = np.array(propList)
	for riskProp in riskProps:
	    if sum(riskProp) < 1:
		riskProp.append(1 - sum(riskProp))
	riskProps = np.array(riskProps)
	popProps = propList[:, np.newaxis] * riskProps
	popArray = [int(round(population * popProp)) for popProp in popProps.ravel()]
	if sum(popArray) < population:
	    popArray[0] += population - sum(popArray)
	self.propArray = popProps
	self.popArray = np.reshape(popArray, popProps.shape)

        # travel between cities
        self.destList = destList
	if destProbs and sum(destProbs) < 1:
	    destProbs.append(1 - sum(destProbs))
        self.destProbs = destProbs
	self.sympTratio = sympTratio

	# asymptomatic and symptomatic infections
	self.psymp = psymp
	self.m = m

	# vaccine efficacy parameters
	self.level = level
	self.theta1 = theta1
	self.theta2 = theta2
	self.VEmaxS = VEmaxS
	self.VEmaxI = VEmaxI
	self.VEmaxP = VEmaxP

        # infectious contact parameters
        VLavg = (m * (1 - psymp) + psymp) * self.VLtraj.sum()/6.
	subpopVec = self.popArray.sum(1)	# sum over risk groups
	k = len(subpopVec)
	subpopdiag = np.diag(subpopVec)
        invsubpop = np.diag(1./subpopVec)                    
	R0 = float(R0)
	if NGmatrix != None:
	    NGmatrix = np.array(NGmatrix)
	if betaMatrix == None:
	    if NGmatrix == None:
		self.R0 = R0
		self.NGmatrix = R0/population * np.outer(np.ones(k), subpopVec)
	    elif R0 == None:
		self.R0 = max(np.abs(np.linalg.eigvals(NGmatrix)))
		self.NGmatrix = NGmatrix
	    else:
		specrad = max(np.abs(np.linalg.eigvals(NGmatrix)))
		self.R0 = R0
		self.NGmatrix = NGmatrix/float(specrad) * R0
	    self.betaMatrix = np.dot(self.NGmatrix, invsubpop)/VLavg
	else:
	    betaMatrix = np.array(betaMatrix)
	    if R0 == None:
		NGmatrix = VLavg * np.dot(self.betaMatrix, subpopdiag)
		self.R0 = max(np.abs(np.linalg.eigvals(NGmatrix)))
		self.NGmatrix = NGmatrix
		self.betaMatrix = betaMatrix
	    else:	# NGmatrix overrides betaMatrix
		specrad = max(np.abs(np.linalg.eigvals(NGmatrix)))
		self.R0 = R0
		self.NGmatrix = NGmatrix/float(specrad) * R0
		self.betaMatrix = np.dot(self.NGmatrix, invsubpop)/VLavg
	
	# Parray and R0ratio for calculating infection probabilities
	VLarray = np.reshape(self.VLtraj, (1, 1, 1, 6, 6))
	mArray = np.reshape([self.m, 1], (1, 1, 2, 1, 1))
	# rows in betaArray correspond to types of susceptibles, not infecteds
	betaArray = self.betaMatrix.transpose().reshape(k, k, 1, 1, 1)
	self.Parray = np.kron(betaArray, np.kron(mArray, VLarray))
	self.R0ratio = 1
    
        # initialize epidemic variables
	vacS = np.zeros(self.popArray.shape, int)
	self.S = np.concatenate((self.popArray[:, :, np.newaxis], 
				 vacS[:, :, np.newaxis]), 2)
	Sshape = self.S.shape
        self.Sdict = {None:self.popArray.copy(),
		      'Past':np.zeros(self.popArray.shape, int)}
	self.incI = np.zeros(Sshape, int)
        self.I = np.zeros(k, int)
        self.Idict = {None:np.zeros((k, 2, 6, 6), int),
		      'Past':np.zeros((k, 2, 6, 6), int)}
        self.R = np.zeros(Sshape, int)
	self.Rq = deque([np.zeros(Sshape, int) for i in range(6)])
	
	# epidemic data storage
	self.imprtList = []
	self.vacList = []
	self.tlist = []
	self.Slist = []
	self.incIlist = []
	self.Ilist = []
	self.Rlist = []

    def VEratio(self, t, vtime):
        """
        Calculates the proportion of maximum vaccine efficacy (a.k.a. vaccine
        efficacy ratio) achieved as a function of time after vaccination.

        Arguments:
            t = time for which vaccine efficacy ratio is calculated
            vtime = vaccination time
        
        """
	if vtime == 'Past':
	    return 1
        elif vtime == None or t <= vtime:
            return 0
        elif t-vtime < 14:
            return self.level * pow((t - vtime)/14., math.exp(self.theta1))
        elif t-vtime <= 21:
            return self.level
        elif t-vtime < 28:
            return self.level + (1 - self.level) * pow((t - vtime - 21)/7.,
						       math.exp(self.theta2))
	else:
	    return 1

    def imprt(self, impnum, t):
        """
        Sends min('impnum', sum(S)) randomly chosen susceptibles to the
        appropriate infected compartments at time 't'.  This number is 
	converted to an infection probability for available susceptibles.  
	Must occur before vaccinate and increment in each time step.

        Arguments:
            impnum = expected number of susceptibles to infect
            t = absolute time

        """
        impnum = min(impnum, self.S.sum())
        if impnum > 0:
            multinomial = np.random.multinomial
	    impProb = impnum / float(self.S.sum())
	    k = len(self.popArray)
	    unif6 = np.ones(6)/6.

	    # update Sdict
	    Sarray = np.array(self.Sdict.values())
	    impIarray = zbinomial(Sarray, impProb)
	    Sarray -= impIarray
	    self.Sdict.update(zip(self.Sdict.keys(), Sarray))
            
	    # update Idict
	    impIdict = dict(zip(self.Sdict.keys(), impIarray))
	    impI = np.zeros(self.S.shape, int)
            for vtime in self.Sdict.iterkeys():
		impIvtime = impIdict[vtime]
		if vtime == None:
		    impI[:, :, 0] += impIvtime
		else:
		    impI[:, :, 1] += impIvtime
		impIvtime = impIvtime.sum(1)	# sum over risk groups
		
		# divide by symptom status and viral load
                veP = self.VEmaxP * self.VEratio(t - 1, vtime)
                psymp = (1 - veP) * self.psymp
                impIvtime = [multinomial(subI, (1 - psymp, psymp))
                             for subI in impIvtime]
                impIvtime = np.array([[multinomial(sympI, unif6) 
				       for sympI in subI]
				      for subI in impIvtime])

		# update Idict
		if self.Idict.has_key(vtime):
		    self.Idict[vtime][:, :, :, 0] += impIvtime
		else:
		    oldI = np.zeros((k, 2, 6, 5), int)
		    impIvtime = impIvtime.reshape(k, 2, 6, 1)
		    self.Idict[vtime] = np.concatenate((impIvtime, oldI), 3)

            # update S, I, and Rq
            self.S -= impI
	    self.incI += impI
            self.Rq[0] += impI
            self.I += [subI.sum() for subI in impI]
	    self.imprtList.append((t, impnum))
	return impI
    
    def vaccinate(self, vacfracArray, t):
        """
	Vaccinate each susceptible in subpopulation i and risk group j with 
	probability vacfracArray[i, j].  Let k denote the number of
	subpopulations and m denote the number of risk groups.  If vacfracArray
	has shape k x 1, then each vacfrac will be applied to all risk groups
	in the corresponding subpopulation.  If vacfracArray has shape 1 x m,
	then each vacfrac will be applied to the corresponding risk group.

        Arguments:
            t -- absolute time
            vacfracArray -- array vacfracArray[i, j] = vaccination fraction in
			    subpopulation i, risk group j; vacfracArray may
			    have shape k x 1 or 1 x m (see above).
            
        """
        vacS = zbinomial(self.Sdict[None], vacfracArray)
        if vacS.any():
            self.Sdict[t] = vacS
            self.Sdict[None] -= vacS
	    self.S[:, :, 0] -= vacS
	    self.S[:, :, 1] += vacS
	self.vacList.append((t, vacfracArray))
	return vacS

    def infP(self, t):
        """
        Calculates infection escape probabilities at time 't' for susceptibles
        in each possible vaccination time.

        Arguments:
            t = absolute time

        Return values:
            dictionary Pdict: Pdict[vtimeS][i] = infection probability for 
	    susceptibles in subpopulation i vaccinated at time vtimeS. Each 
	    probability is enclosed in its own list, so Pdict[vtimeS] has 
	    shape (k, 1), where k is the number of subpopulations; this 
	    ensures that the same infection probability is used for all risk
	    groups in each subpopulation (see NumPy broadcasting rules).
            
        """
	Parray = self.R0ratio * self.Parray
	
	# make local variables
	prod = np.prod
	power = np.power
        k = len(self.popArray)
	Sdict = self.Sdict
	Idict = self.Idict
	
	# Initialize escape probabilities for susceptibles
        cumQlist = np.array([np.ones(k) for vtimeS in Sdict.keys()])
        veSlist = self.VEmaxS * np.array([self.VEratio(t, vtimeS) 
					  for vtimeS in Sdict.keys()])

	# Calculate escape probabilities
	for vtimeI in Idict.iterkeys():
	    veI = self.VEmaxI * self.VEratio(t, vtimeI)
	    Iarray = Idict[vtimeI]
	    QarrayList = [1 - (1 - veI) * (1 - veS) * Parray
			  for veS in veSlist] 
	    Qlist = [[prod(power(QarrayStype, Iarray))
		      for QarrayStype in QarraySvtime] 
		     for QarraySvtime in QarrayList]
	    cumQlist *= Qlist
	
	# calculate infection probabilities
	Plist = 1 - cumQlist
	Pdict = dict(zip(Sdict.iterkeys(), Plist[:, :, np.newaxis]))
	return Pdict
        
    def increment(self, t):
        """
        Increments epidemic from time t to time t + 1. Occurs last in each
        time step.  

        Arguments:
            t = absolute time

        """
	# record epidemic information
	self.tlist.append(t)
	self.Slist.append(self.S.copy())
	self.incIlist.append(self.incI.copy())
	self.Ilist.append(self.I.copy())
	self.Rlist.append(self.R.copy())

	# make local variables
	zeros = np.zeros
	concatenate = np.concatenate
	multinomial = np.random.multinomial
	unif6 = np.ones(6)/6.
	Sdict = self.Sdict
	Idict = self.Idict

	newI = np.zeros(self.S.shape, int)
	if self.I.any():
	    # clean up Sdict, Idict
	    for vtimeS in Sdict.keys():
		if vtimeS != 'Past' and vtimeS != None and vtimeS <= t - 28:
		    Sdict['Past'] += Sdict[vtimeS]
		    del Sdict[vtimeS]
	    for vtimeI in Idict.keys():
		if vtimeI != 'Past' and vtimeI != None and vtimeI <= t - 28:
		    Idict['Past'] += Idict[vtimeI]
		    del Idict[vtimeI]	    

	    # calculate infection probabilities
            Pdict = self.infP(t)

	    # prepare Idict to receive new infections
	    k = len(self.I)
            newRpres = zeros(k, int)
	    for vtimeI in Idict.iterkeys():
		Iarray = Idict[vtimeI]
		if Iarray.any():
		    # calculate number of recovered persons in each subpopulation
		    recovered = Iarray[:, :, :, -1]
		    newRpres += [subR.sum() for subR in recovered]

		    # move infected people forward one day
		    oldI = Iarray[:, :, :, :-1]
		    newIvtimeI = zeros((k, 2, 6, 1), int)
		    Idict[vtimeI] = concatenate((newIvtimeI, oldI), 3)
	    
	    # infect susceptibles
            for vtimeS in Sdict.iterkeys():
		Sarray = Sdict[vtimeS]
		if Sarray.any():
		    # select new infections
		    newIvtimeS = zbinomial(Sarray, Pdict[vtimeS])
		    Sarray -= newIvtimeS
		    #Sdict[vtimeS] -= newIvtimeS
		    if vtimeS == None:
			newI[:, :, 0] += newIvtimeS
		    else:
			newI[:, :, 1] += newIvtimeS
		    newIvtimeS = newIvtimeS.sum(1)	# sum over risk groups
                
		# divide by symptom status and viral load trajectory
		    veP = self.VEmaxP * self.VEratio(t, vtimeS)
		    psymp = (1 - veP) * self.psymp
		    newIvtimeS = [multinomial(subI, (1 - psymp, psymp)) 
				  for subI in newIvtimeS]
		    newIvtimeS = np.array([[multinomial(sympI, unif6) 
					    for sympI in subI] 
					   for subI in newIvtimeS])

                # add new infections to Idict
		    if Idict.has_key(vtimeS):
			Idict[vtimeS][:, :, :, 0] = newIvtimeS
		    else:
			oldI = zeros((k, 2, 6, 5), int)
			newIvtimeS = newIvtimeS.reshape(k, 2, 6, 1)
			Idict[vtimeS] = concatenate((newIvtimeS, oldI), 3)
	    
	    # update S, I
	    self.S -= newI
	    self.I += [subI.sum() for subI in newI] - newRpres
	
	# record incident infections and update R, Rq
	self.incI = newI
        self.R += self.Rq.pop()
        self.Rq.appendleft(newI)

    def city_epidemic(self, starttime=0, impnum=72):
        """
        Runs within-city epidemic.

        Arguments:
            impnum = number of imported infections at time 0

        """
	# record state of epiList prior to start of epidemic
	self.tlist.append(starttime - 1)
	self.Slist.append(np.copy(self.S))
	self.incIlist.append(np.copy(self.incI))
	self.Ilist.append(np.copy(self.I))
	self.Rlist.append(np.copy(self.R))
	
	# begin epidemic
        t = starttime
        self.imprt(impnum, t)
	while self.I.any() or np.any(self.Rq):
            self.increment(t)
            t += 1
	self.increment(t) # records final state of population

    def send(self, t):
        """
        Send infected persons to other cities. Must occur prior 
	to import and vaccinate in each time step.

        Arguments:
            t = absolute time

        Return values:
	    depDict.items() -- depDict[destination] = [(vtime, depIarray) for 
			       all vtime such that at least one infected person 
			       vaccinated at vtime is traveling to destination]
            
        """
	depList = []
	if self.I.any():
	    # make local variables
	    multinomial = np.random.multinomial
	    array = np.array
	    concatenate = np.concatenate
	    rollaxis = np.rollaxis
	    Idict = self.Idict
	    destProbs = self.destProbs

	    k = len(self.I)
	    sympProbs = self.sympTratio * destProbs
	    self.I = np.zeros(k, int)
	    for vtimeI in Idict.iterkeys():
		Iarray = Idict[vtimeI]
		if Iarray.any():
		    flatIasymp = Iarray[:, 0].ravel()
		    asympDepartures = array([multinomial(I, destProbs) 
					     for I in flatIasymp])
		    asympDepartures = asympDepartures.reshape(k, 1, 6, 6, 
							      len(destProbs))
		    flatIsymp = Iarray[:, 1].ravel()
		    sympDepartures = array([multinomial(I, sympProbs)
					    for I in flatIsymp])
		    sympDepartures = sympDepartures.reshape(k, 1, 6, 6,
							    len(sympProbs))
		    departures = concatenate((asympDepartures,
					      sympDepartures), 1)
		    departures = rollaxis(departures, 4)
		    
		    # update Idict to keep remaining infecteds
		    Istay = departures[-1]
		    self.I += [subI.sum() for subI in Istay]
		    Idict[vtimeI] = Istay
		    
		    # update depDict
		    depList.append((vtimeI, departures[:-1]))
	return depList	

    def receive(self, arrList):
	"""
	Receive infected visitors.  Must occur prior to import and vaccinate
	in each time step.

	Arguments:
	    arrList -- list of (vtime, arrivals), where arrivals is a NumPy
		       array of infected persons with shape (k, 2, 6, 6) and 
		       k is the number of subpopulations.

	"""
	Idict = self.Idict
	visitors = np.zeros(len(self.I))
	for (vtimeI, arrivals) in arrList:
	    visitors += [subI.sum() for subI in arrivals]
	    if Idict.has_key(vtimeI):
		Idict[vtimeI] += arrivals
	    else:
		Idict[vtimeI] = arrivals
	self.I += visitors
	return visitors


# next-generation matrices from McBryde et al.
VnextGen2 = np.array([[1.8, 0.5], [0.5, 0.2]])
VnextGen3 = np.array([[2.7, 0.7], [0.5, .4]])

class globalModel:
    """
    The globalModel is the base class for our global influenza models.  It 
    initializes and controls the epiLists for all cities in the model.  The
    base class has no seasonality; seasonality is implemented in subclasses.

    Subpopulation 0 is children, subpopulation 1 is adults.  Risk group 0 is
    low-risk, risk group 1 is high-risk.

    Data attributes:
	Model characteristics:
	name -- name for model; used in plots
	randomSeed -- seed for random number generator; defaults to None
	propArray -- NumPy array with propArray[i, j] = proportion of population
		     in subpopulation i and risk group j
	popArray -- NumPy array with popArray[i, j] = number of persons in
		    subpopulation i and risk group j
	population -- total population
	cityData -- data on each city in the model
	cityList -- list of cities in the model
	cityDict -- dictionary cityDict[index] = name; the index is the
		    position of the city in cityData
	airArray -- air travel matrix; airArray[i, j] = average daily
		    number of travelers from city i to city j
	Cities -- dictionary Cities['name'] is the epiList for city 'name'
	sympTratio -- the ratio of travel probabilities for symptomatics to 
		      those of asymptomatics; between 0 and 1
	exchangeArrayList -- Boolean; if exchangeArrayList == True, the 
			     exchangeArray for each time step will be saved
	
	
	Transmission parameters:
	R0 -- peak seasonal R0
	lowR0 -- off-season R0
	NGmatrix -- next-generation matrix
	betaMatrix -- matrix of peak seasonal betas
	psymp -- the probability of being symptomatic
	m -- asymptomatic/symptomatic infectiousness
	level -- maximum vaccine efficacy ratio achieved after first dose
	theta1, theta2 -- govern the shape of the VE ratio increase after
			  first and second dose, respectively; increase is
			  concave for theta <= 0 and convex for theta >= 0
	VEmaxS, VEmaxI, VEmaxP -- maximum vaccine efficacy for susceptibility, 
				  infectiousness, and progression, respectively

	Epidemic variables:
	S -- NumPy array S[i, j] = number of susceptible persons in
	     subpopulation i and risk group j
	incI -- NumPy array incI[i, j] = number of new infections in
		subpopulation i and risk group j
        I -- NumPy vector I[i] = number of infected persons in subpopulation i
        R -- NumPy array R[i, j] = number of recovered person in subpopulation i 
	     and risk group j
	exchangeArray -- array exchangeArray[i, j] is the number of infected
			 visitors traveling from the city with city.index = i 
			 to the city with city.index = j

	Sequencing variables: (used to maintain proper ordering of events)
	lastex -- time of last exchange
	lastimp -- time of last importation
	lastvac -- time of last vaccination
	lastinc -- time of last incrementation
	
	Epidemic data storage:
	vacList -- list [(t, [(city, vacfracArray) for city vaccinated at time t]) 
		   for each t on which persons were vaccinated]
	imprtList -- list [(t, [(city, impnum) for city with imported infections at
		     time t]) for each t on which imported infections occurred]
	tlist -- list of times for which S, incI, I, and R were recorded
	Slist -- list of S for each time in tlist
	incIlist -- list of incI for each time in tlist
	Ilist -- list of I for each time in tlist
	Rlist -- list of R for each time in tlist
	exchangeList -- list of exchangeArray for each time in tlist; saved 
			only if exchangeArrayList == True

    Methods:
        seasonal_beta -- does nothing in this class; used to implement
                         seasonality in subclasses
        exchange -- exchange visitors between cities
        imprt -- import infection to cities
        vaccinate -- vaccinate cities
        increment -- increment model from time t to time t+1
        global_epidemic -- run a global epidemic 
    
    """
    def __init__(self, name='Erth', 
		 popFile = 'DataFiles/population_321_age.txt',
		 travelFile = 'DataFiles/travel_321.txt', sympTratio=1,
		 R0=1.4, lowR0=.6, NGmatrix=VnextGen2,
		 psymp=2. / 3,      # probability of becoming symptomatic
		 m=.5,              # asymptomatic/symptomatic infectiousness
		 level=.5,          # vaccine efficacy parameters
		 theta1=0, theta2=0,
		 VEmaxS = .4, VEmaxI = .4, VEmaxP = .667,
		 exchangeArrayList=False, randomSeed=None):
	"""
	Initializes globalModel.  All arguments are saved as globalModel 
	attributes.  

	Arguments:
	    Global model characteristics:
	    name -- name for plots, etc.
	    popFile -- file from which cityData is created
	    travelFile -- file used to generate airArray
	    sympTratio -- ratio of symptomatic travel probabilities to
			  asymptomatic travel probabilities; between 0 and 1
	    exchangeArrayList -- defaults to False; set to True to save 
				 exchangeArray at each time step.
	    randomSeed -- seed for random number generator

	    Transmission parameters:
            R0 -- desired R0; if R0 == None, then R0 equals the spectral radius 
		  of the next-generation matrix
            NGmatrix -- NumPy array for next-generation matrix; if R0 != None,
                        it will be multiplied by a scalar to obtain desired R0.
	    psymp -- the probability of being symptomatic
	    m -- asymptomatic/symptomatic infectiousness
	    level -- maximum vaccine efficacy ratio achieved after first dose
	    theta1, theta2 -- govern the shape of the VE ratio increase after
			      first and second dose, respectively; increase is
			      concave for theta <= 0 and convex for theta >= 0
	    VEmaxS, VEmaxI, VEmaxP -- maximum vaccine efficacy for
				      susceptibility, infectiousness, and
				      progression, respectively    
	    
	"""
        self.name = name
	self.popFile = popFile
	self.travelFile = travelFile
	self.randomSeed = randomSeed
	np.random.seed(randomSeed)

	self.sympTratio = sympTratio
        self.lowR0 = float(lowR0)
	NGmatrix = np.array(NGmatrix)
	specrad = max(np.abs(np.linalg.eigvals(NGmatrix)))
	if R0 == None:
	    self.R0 = float(specrad)
	    self.NGmatrix = NGmatrix
	else:
	    self.R0 = float(R0)
	    self.NGmatrix = NGmatrix/float(specrad) * R0

	# initialize cityData
	# cityData contains information for all cities in the model
	cityData = np.loadtxt(popFile,
			      dtype={'names':('City', 'Number', 'Zone', 
					      'Population', 'Longitude', 
					      'Latitude', 'Nation',
					      'Percent0_14', 'Percent15_65'),
				     'formats':('S24', 'i4', 'S1', 'i4', 'f4',
						'f4', 'S16', 'f4', 'f4')},
			      skiprows=1)
	cityData['Number'] -= 1
	self.cityData = cityData
	self.cityList = self.cityData['City']
	self.cityDict = dict(zip(cityData['Number'], cityData['City']))

	# create airArray[i,j] = number of visitors to city i from city j
	travel_file = open(travelFile)
	travel_list = [line.split()[2:] for line in travel_file.readlines()]
	travel_file.close()
	del travel_list[0]
	self.airArray = np.array(travel_list, float)
        
	# create epiList for each city
        self.Cities = {}
        popArray = S = 0
        for i in xrange(len(self.cityData)):
            city = self.cityData[i]
	    cityName = city['City']
	    self.Cities[cityName] = epiList(population = city['Population'], 
					    propList = [city['Percent0_14']
							/100.], 
					    riskProps = [[.9, .1], 
							 [.83, .17]],
					    name = city['City'],
					    latitude = city['Latitude'],
					    longitude = city['Longitude'],
					    sympTratio = sympTratio,
					    R0 = R0, NGmatrix = NGmatrix, 
					    psymp = psymp, m = m, 
					    level = level,
					    theta1 = theta1, theta2 = theta2,
					    VEmaxS = VEmaxS, VEmaxI = VEmaxI,
					    VEmaxP = VEmaxP)

	    # create destList, destProbs
	    destIndices = np.nonzero(self.airArray[i])
            destList = self.cityData['City'][destIndices]
            destProbs = self.airArray[i][destIndices]
            pop = self.Cities[cityName].population
            destProbs = np.append(destProbs, pop - destProbs.sum())
            destProbs /= float(pop)
            self.Cities[cityName].destList = destList
            self.Cities[cityName].destProbs = destProbs
	    self.Cities[cityName].index = i

            popArray += self.Cities[cityName].popArray
	    S += self.Cities[cityName].S

        # population
	self.population = popArray.sum()
	self.propArray = popArray / float(self.population)
	self.popArray = popArray
	
	# asymptomatic and symptomatic infections
	self.psymp = psymp
	self.m = m

	# vaccine efficacy parameters
	self.level = level
	self.theta1 = theta1
	self.theta2 = theta2
	self.VEmaxS = VEmaxS
	self.VEmaxI = VEmaxI
	self.VEmaxP = VEmaxP

        # initialize epidemic variables
	k = len(S)
	Sshape = np.shape(S)
        self.S = S
	self.incI = np.zeros(Sshape, int)
        self.I = np.zeros(k, int)
        self.R = np.zeros(Sshape, int)

	# sequencing variables
	self.lastex = None
	self.lastimp = None
	self.lastvac = None
	self.lastinc = None
	
	# epidemic data storage
	self.imprtLists = []
	self.vacLists = []
	self.tlist = []
	self.Slist = []
	self.incIlist = []
	self.Ilist = []
	self.Rlist = []
	self.exchangeArrayList = exchangeArrayList
	if self.exchangeArrayList:
	    self.exchangeList = []
    
    def exchange(self, t):
        """
        Exchange visitors between cities.  Must occur before importation,
        vaccination, and incrementation in each time step.

        """
	Cities = self.Cities
	exchangeArrayList = self.exchangeArrayList
        if t > max(self.lastex, self.lastimp, self.lastvac, self.lastinc):
	    if exchangeArrayList:
		exchangeArray = np.zeros((len(Cities),) * 2, int)
            for orig in Cities.itervalues():
		destList = orig.destList
                depList = orig.send(t)
		destinations = (Cities[destname] for destname in destList)
		arrLists = (((vtime, depArray[i]) 
			     for (vtime, depArray) in depList 
			     if depArray[i].any())
			    for i in xrange(len(destList)))
		departureList = [dest.receive(arrList) 
				 for (dest, arrList) 
				 in izip(destinations, arrLists)]
		if exchangeArrayList:
		    origIndex = orig.index
		    destIndices = [Cities[destname].index for destname in destList]
		    departureSums = [destvisitors.sum() 
				     for destvisitors in departureList]
		    exchangeArray[origIndex, destIndices] = departureSums
	    if exchangeArrayList:
		self.exchangeList.append(exchangeArray)
	    self.lastex = t
        else:
            print "Sequence error: exchange at t = %s." % t
	    
    def imprt(self, imprtList, t):
        """
        Import infection to cities in the global model.  Must occur after
        exchange and before vaccination and incrementation in each time step.

        Arguments:
            imprtList -- list of (city, impnum) for each city to which
                         infection will be imported; each city may be
                         designated by name or number
            t = absolute time
        
        """
        if (t >= self.lastex and
            t > max(self.lastimp, self.lastvac, self.lastinc)):
            for (city, impnum) in imprtList:
		impI = self.Cities[city].imprt(impnum, t)
		self.S -= impI
		self.incI += impI
		self.I += [subI.sum() for subI in impI]
            self.lastimp = t
	    self.imprtLists.append((t, imprtList))
        else:
            print "Sequence error: imprt at t = %s." % t

    def vac_list(self, t):
	"""
	Generates vacList for self.vaccinate.  In the base class, this is just
	a placeholder.
	
	Arguments:
	    t = time

	Returns:
	    vacList = [(city, vacfrac) for city with vacfrac > 0]

	"""
	return []

    def vaccinate(self, vacList, t):
        """
        Vaccinate cities in the global model.  Must occur after exchange and
        import but before incrementation in each time step.

        Arguments:
            vacList -- list of (city, vacfracArray) for each city to be 
		       vaccinated at time t
            t = absolute time
        
        """
        if (t >= max(self.lastex, self.lastimp) and
            t > max(self.lastvac, self.lastinc)):
            for (city, vacfracArray) in vacList:
		vacS = self.Cities[city].vaccinate(vacfracArray, t)
		self.S[:, :, 0] -= vacS
		self.S[:, :, 1] += vacS
            self.lastvac = t
	    self.vacLists.append((t, vacList))
        else:
            print "Sequence error: vaccination at t = %s." % t

    def seasonal_beta(self, t):
        """
        Does nothing in the base class.  Used to implement seasonality in
	subclasses by setting city.R0ratio in each city.

        Arguments:
            t = absolute time; t = 0 is January 1
        """
        pass

    def increment(self, t):
        """
        Increment cities in the global model from time t to time t+1.  Records
        S, incI, I, and R at time t.  Must occur last in each time step.

        Arguments:
            t = absolute time
        
        """
        if (t >= max(self.lastex, self.lastimp, self.lastvac) and
            t > self.lastinc):
	    # record epidemic data
            self.tlist.append(t)
	    self.Slist.append(self.S.copy())
	    self.incIlist.append(self.incI.copy())
	    self.Ilist.append(self.I.copy())
	    self.Rlist.append(self.R.copy())
	    
	    # increment epidemics and record S, incI, I, R
	    S = incI = I = R = 0
            self.seasonal_beta(t)
            for city in self.Cities.itervalues():
		city.increment(t)
		S += city.S
		incI += city.incI
		I += city.I
		R += city.R
	    self.lastincr = t
	    self.S = S
	    self.incI = incI
	    self.I = I
	    self.R = R  
	    return t + 1
        else:
            print "Sequence error: incrementation at t = %s." % t

    def global_epidemic(self, starttime=105, 
			imprtList=[('Mexico_City', 10000)]):
        """
        Runs a global epidemic

        Arguments:
            starttime = day of year on which to start
            imprtList = [(city, impnum) for each city receiving imported
                         infection at starttime]
                         
        """
	# record state of population prior to epidemic
        self.tlist.append(starttime - 1)
	self.Slist.append(self.S.copy())
	self.incIlist.append(self.incI.copy())
	self.Ilist.append(self.I.copy())
	self.Rlist.append(self.R.copy())

	# begin epidemic
        t = starttime
        self.imprt(imprtList, t)
        t = self.increment(t)
        while self.I.any():
            self.exchange(t)
	    vacList = self.vac_list(t)
	    if vacList:
		self.vaccinate(vacList, t)
            t = self.increment(t)
        self.exchange(t)
        self.increment(t)

class step_globalModel(globalModel):
    """
    Global model with step-function seasonality

    """
    def __init__(self, Nstart=243, Sstart=59, *args, **kwargs):
	globalModel.__init__(self, *args, **kwargs)
	self.Nstart = int(Nstart)
	self.Sstart = int(Sstart)
    
    def seasonal_beta(self, t):
        """
        Calculates betamat for each city based on season and latitude.  In this
        subclass, it is a step function.  The northern flu season starts on
        October 1 and ends on March 31.  The southern flu season starts on
        April 1 and ends on September 30.  During flu season, the city R0 is
        R0.  Otherwise, the city R0 is lowR0.

        Arguments:
            t = absolute time; t = 0 is January 1
        """
        # yrday = 0 for January 1 and yrday = 364 for December 31
        # Jan = [0-30],    Feb = [31-58],   Mar = [59-89],
        # Apr = [90-119],  May = [120-150], Jun = [151-180],
        # Jul = [181-211], Aug = [212-242], Sep = [243-272],
        # Oct = [273-303], Nov = [304-333], Dec = [334-364]
        yrday = t % 365
        lowR0ratio = self.lowR0/self.R0
        # northern spring/summer, southern fall/winter
        if yrday >= self.Sstart and yrday < self.Nstart:
            for city in self.Cities.itervalues():
                if city.latitude > 23.5:
                    city.R0ratio = lowR0ratio
                elif city.latitude < -23.5:
                    city.R0ratio = 1
        else:
            for city in self.Cities.itervalues():
                if city.latitude > 23.5:
                    city.R0ratio = 1
                elif city.latitude < -23.5:
                    city.R0ratio = lowR0ratio


class linear_globalModel(globalModel):
    """
    Global model with local R0 increasing linearly from lowR0 to R0
    from yrday = Xstart - 30 days to yrday = Xstart + 30 days.

    """
    def __init__(self, Nstart=243, Sstart=59, *args, **kwargs):
	globalModel.__init__(self, *args, **kwargs)
	self.Nstart = int(Nstart)
	self.Sstart = int(Sstart)

    def seasonal_beta(self, t):
        """
        Calculates beta for each city based on season and latitude.  In this
        subclass, it is a smooth function.  During flu season, the city R0 is
        R0.  Otherwise, the city R0 is lowR0.

        Arguments:
            t = absolute time; t = 0 is January 1
        """
        # yrday = 0 for January 1 and yrday = 364 for December 31
        # Jan = [0-30],    Feb = [31-58],   Mar = [59-89],
        # Apr = [90-119],  May = [120-150], Jun = [151-180],
        # Jul = [181-211], Aug = [212-242], Sep = [243-272],
        # Oct = [273-303], Nov = [304-333], Dec = [334-364]
        yrday = t % 365
        lowR0ratio = self.lowR0/float(self.R0)
        R0diff = (self.R0 - self.lowR0)
        # Northern autumn, Southern spring
        if yrday > self.Nstart - 30 and yrday < self.Nstart + 30:
            Nseasonality = ((yrday - self.Nstart)/60. + .5) * (1 - lowR0ratio)
            for city in self.Cities.itervalues():
                if city.latitude > 23.5:
                    city.R0ratio = lowR0ratio + Nseasonality
                elif city.latitude < -23.5:
                    city.R0ratio = lowR0ratio + (1 - Nseasonality)
        # Northern winter, Southern summer
        elif yrday >= self.Nstart + 30 and yrday <= self.Sstart - 30:
            for city in self.Cities.itervalues():
                if city.latitude > 23.5:
                    city.R0ratio = 1
                elif city.latitude < -23.5:
                    city.R0ratio = lowR0ratio
        # Northern spring, Southern autumn
        elif yrday > self.Sstart - 30 and yrday < self.Sstart + 30:
            Sseasonality = ((yrday - self.Sstart)/60. + .5) * (1 - lowR0ratio)
            for city in self.Cities.itervalues():
                if city.latitude > 23.5:
                    city.R0ratio = lowR0ratio + (1 - Sseasonality)
                elif city.latitude < -23.5:
                    city.R0ratio = lowR0ratio + Sseasonality
        # Northern summer, Southern winter
        else:
            for city in self.Cities.itervalues():
                if city.latitude > 23.5:
                    city.R0ratio = lowR0ratio
                elif city.latitude < -23.5:
                    city.R0ratio = 1


class season_globalModel(globalModel):
    def __init__(self, seasonFile='DataFiles/seasonality_321.csv', 
		 *args, **kwargs):
	globalModel.__init__(self, *args, **kwargs)
	self.seasonFile = seasonFile
	season_file = open(self.seasonFile)
	season_lines = season_file.readlines()
	season_cities = [line.split(',')[0] for line in season_lines]
	season_file.close()

	for city in self.Cities.itervalues():
	    index = season_cities.index(city.name)
	    city.seasonList = np.array(season_lines[index].split(',')[1:], 
				       float)

    def seasonal_beta(self, t):
        """
        Calculates R0ratio for each city based on season and latitude.
        In this subclass, multipliers for each city for each day are
        read from 'seasonality.csv'. The multiplier for R0 can be from 0 to 1,
        where 0 indicates R0 should be lowR0, 1 is R0, and intermediate
        values are interpolated.

        Arguments:
            t = absolute time; t = 0 is January 1
        """
        yrday = (t % 365)
        lowR0ratio = self.lowR0/self.R0
        for city in self.Cities.itervalues():
            city.R0ratio = lowR0ratio + (1 - lowR0ratio) * city.seasonList[yrday]

class prevac_globalModel(season_globalModel):
    def vac_list(self, t):
	if t == self.Nvactime:
	    Europe = ['Amsterdam', 'Athens', 'Barcelona', 'Berlin', 'Brussels', 
		    'Budapest', 'Copenhagen', 'Dublin', 'Dusseldorf', 
		    'Frankfurt', 'Geneva', 'Hamburg', 'Helsinki', 'London',
		    'Madrid', 'Manchester', 'Milan', 'Munich', 'Nice', 'Oslo',
		    'Palma_de_Mallorca', 'Paris', 'Rome', 'Sofia', 'Vienna',
		    'Warsaw', 'Zurich']
	    NorthAmerica = ['Atlanta', 'Boston', 'Charlotte', 'Chicago', 
			    'Cincinnati', 'Cleveland', 'Dallas', 'Denver', 
			    'Detroit', 'Honolulu', 'Houston', 'Indianapolis',
			    'Kansas_City', 'Las_Vegas', 'Los_Angeles', 
			    'Memphis', 'Miami', 'Minneapolis', 'Montreal', 
			    'Nashville', 'New_Orleans', 'New_York', 'Orlando', 
			    'Philadelphia', 'Phoenix', 'Pittsburgh', 
			    'Portland', 'Raleigh', 'Salt_Lake_City', 
			    'San_Diego', 'San_Francisco', 'San_Juan', 
			    'Seattle', 'St_Louis', 'Tampa', 'Toronto',
			    'Vancouver', 'Washington']
	    EastAsia = ['Fukuoka', 'Kaohsiung', 'Nagoya', 'Okinawa', 'Osaka', 
			'Pusan', 'Sapporo', 'Seoul', 'Singapore', 'Taegu', 
			'Taipei', 'Tokyo']
	    vacList = [(city, self.prevacfrac) for city 
		       in Europe + NorthAmerica + EastAsia]
	elif t == self.Svactime:
	    Anzac = ['Auckland', 'Brisbane', 'Melbourne', 'Perth', 'Sydney', 
		     'Wellington']
	    vacList = [(city, self.prevacfrac) for city in Anzac]
	else:
	    vacList = []
	return vacList
    
    def global_epidemic(self, prevacfrac=.10, Nvactime=75, Svactime=285,
			*args, **kwargs):
	# assumes Nvactime < starttime and Svactime >= starttime
	self.prevacfrac = prevacfrac
	self.Nvactime = Nvactime
	self.Svactime = Svactime
	self.vaccinate(self.vac_list(Nvactime), Nvactime)
	season_globalModel.global_epidemic(self, *args, **kwargs)

##############
class vacfile_globalModel(season_globalModel):
    """
    Seasonal global model with vaccination schedule read from the .csv file
    vacFilename, a required argument.

    """
    def __init__(self, vacFile='DataFiles/vaclist-sept.csv', 
		 highriskfirst=False, *args, **kwargs):
	season_globalModel.__init__(self, *args, **kwargs)
	vac_file = open(vacFile)
	self.vaccinationList = [line.split(',') for line in vac_file.readlines()]
        self.highriskfirst = highriskfirst
	vac_file.close()

    def vac_cityallocate(self, cityname, vfrac):
        """
        Allocates vaccine to risk groups and subpopulations in the city.
        Returns either vfrac or an array of vaccination fractions for
        each subpopulation/risk group combination.
        """
        if self.highriskfirst:
            vfracarray=np.array([[0.0,0.0],[0.0,0.0]])
            vacnum=self.Cities[cityname].population*vfrac # vaccines available
            # vaccinate high risk children, HR adults, normal children, then normal adults
            for risk in xrange(1,-1,-1):
                for pop in xrange(2):
                    frac=float(vacnum)/self.Cities[cityname].popArray[pop,risk]
                    if frac>0.5:
                        frac=0.5
                    elif frac<0.0:
                        frac=0.0
                    vacnum-=frac*self.Cities[cityname].popArray[pop,risk]
                    vfracarray[pop,risk]=frac
            return vfracarray
        else:
            return vfrac

    def vac_list(self, t):
        vacList = []
        while len(self.vaccinationList)>0 and t==int(self.vaccinationList[0][0]):
            vfrac=float(self.vaccinationList[0][2])
            cityname=self.vaccinationList[0][1]
            vacList.append((cityname,self.vac_cityallocate(cityname, vfrac)))
            self.vaccinationList = self.vaccinationList[1:]
	return vacList
    
    def global_epidemic(self, starttime=163, 
			imprtList=[('Hong_Kong', 100)]):
        vacList = []
        lasttime=starttime+1
        while True:
            if len(self.vaccinationList)==0 or int(self.vaccinationList[0][0])!=lasttime:
                if len(vacList)>0:
                    self.vaccinate(vacList, lasttime)
                    vacList = []
                if len(self.vaccinationList)==0:
                    break
                lasttime=int(self.vaccinationList[0][0])
            if int(self.vaccinationList[0][0])>=starttime:
                break
            vfrac=float(self.vaccinationList[0][2])
            cityname=self.vaccinationList[0][1]
            vacList.append((cityname,self.vac_cityallocate(cityname, vfrac)))
            self.vaccinationList = self.vaccinationList[1:]
	season_globalModel.global_epidemic(self, starttime, imprtList)

