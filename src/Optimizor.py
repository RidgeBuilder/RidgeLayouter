import pandas as pd
import numpy as np
import scipy as sp
import math
from vrpSolver.ipTSP import _ipTSPGurobiLazyCuts
import constants

class Optimizor(object):
    def __init__(self) -> None:
        pass
    
    # Process input data into a more usable format for optimization
    def process_data(self, odata):
        # Convert to NumPy array
        keys = list(odata.columns[:])
        data = odata[keys][1:].T.values.tolist()
        
        # Convert to NumPy array and normalize the values
        data = np.array(data, dtype=float)
        N, M = data.shape
        for i in range(N):
            data[i] /= float(max(data[i]))

        # Find the best position for data rotation based on the minimum sum of values
        recSum = constants.MAX_INT
        recJ = 0
        for j in range(M):
            tmpSum = 0
            for i in range(N):
                tmpSum += data[i][j]
            if tmpSum < recSum:
                recJ = j
                recSum = tmpSum
        
        # Rotate the data based on the calculated best index (recJ)
        rdata = np.zeros((N,M))
        for i in range(N):
            for j in range(M):
                rdata[i][j] = data[i][(j+recJ)%M]

        return rdata, keys, recJ
    
    # Calculate the distance between ridges using different cost functions
    def compute_dis(self, data, offset, theta, cost_func = constants.COST_FUNCTION.PEAK_SHIFTING_AND_HIDDEN): 
        N,M = data.shape
        
        # Find the peaks and max peaks in the data
        peaks = find_peaks(data)
        maxPeak = find_maxPeaks(data)
                    
        peak_dis = np.zeros((N+1,N+1))
        hidden_dis = np.zeros((N+1,N+1))
        dis = np.zeros((N+1,N+1))
        
        # Iterate over all ridges and compute distance between them based on the cost function
        for i in range(N):
            peak_dis[i+1][i+1] = constants.MAX_INT
            dis[i+1][i+1] = constants.MAX_INT
            for j in range(i+1, N):
                if cost_func == constants.COST_FUNCTION.NEAREST_AVG_PEAK:
                # Nearest average diff on the left side of peak
                    cnt1 = 0
                    cnt2 = 0
                    for p in range(len(peaks[i])):
                        rec = -1
                        for q in range(len(peaks[j])):
                            if peaks[i][p] >= peaks[j][q] and peaks[j][q] > peaks[j][rec]:
                                rec = q
                        if rec != -1:
                            cnt1 += 1
                            peak_dis[i+1][j+1] += peaks[i][p] - peaks[j][rec]
                    
                    for q in range(len(peaks[j])):
                        rec = -1
                        for p in range(len(peaks[i])):
                            if peaks[j][q] >= peaks[i][p] and peaks[i][p] > peaks[i][rec]:
                                rec = p
                        if rec != -1:
                            cnt2 += 1
                            peak_dis[j+1][i+1] += peaks[j][q] - peaks[i][rec]
                                
                    if cnt1 == 0:
                        peak_dis[i+1][j+1] = constants.MAX_INT
                    else:
                        peak_dis[i+1][j+1] /= cnt1
                    
                    if cnt2 == 0:
                        peak_dis[j+1][i+1] = constants.MAX_INT
                    else:
                        peak_dis[j+1][i+1] /= cnt2
                elif cost_func == constants.COST_FUNCTION.GLOBAL_AVG_PEAK:
                # Global average diff on the left side of peak
                    cnt1 = 0
                    cnt2 = 0
                    for p in range(len(peaks[i])):
                        for q in range(len(peaks[j])):
                            if peaks[i][p] < peaks[j][q]:
                                peak_dis[j+1][i+1] += peaks[j][q] - peaks[i][p]
                                cnt2 += 1
                            else:
                                peak_dis[i+1][j+1] += peaks[i][p] - peaks[j][q]
                                cnt1 += 1
                    if cnt1 == 0:
                        peak_dis[i+1][j+1] = constants.MAX_INT
                    else:
                        peak_dis[i+1][j+1] /= cnt1
                    
                    if cnt2 == 0:
                        peak_dis[j+1][i+1] = constants.MAX_INT
                    else:
                        peak_dis[j+1][i+1] /= cnt2
                    # compute the cost of hidden
                    for k in range(M):
                        if data[j][k] > data[i][k] + offset:
                            hidden_dis[i+1][j+1] += 1
                        if data[i][k] > data[j][k] + offset:
                            hidden_dis[j+1][i+1] += 1
                elif cost_func == constants.COST_FUNCTION.GLOBAL_AREA_DIFF:
                # Weighted area diff
                    for k in range(M):
                        peak_dis[i+1][j+1] += (k * data[i][k] - k * data[j][k]) / M
                        peak_dis[j+1][i+1] += (k * data[j][k] - k * data[i][k]) / M
                    if peak_dis[i+1][j+1] < 0:
                        peak_dis[i+1][j+1] = constants.MAX_INT
                    if peak_dis[j+1][i+1] < 0:
                        peak_dis[j+1][i+1] = constants.MAX_INT
                elif cost_func == constants.COST_FUNCTION.GLOBAL_PEAK_DIS_2D:
                # Weighted euclidean distance diff(monotonicity not satisfied)
                    cnt1 = 0
                    cnt2 = 0
                    tmp1 = 0
                    tmp2 = 0
                    for p in range(len(peaks[i])):
                        for q in range(len(peaks[j])):
                            if peaks[i][p] < peaks[j][q]:
                                cnt1 += 1
                                tmp1 += (data[i][peaks[i][p]] + offset - data[j][peaks[j][q]]) * (data[i][peaks[i][p]] + offset - data[j][peaks[j][q]]) + (peaks[i][p] - peaks[j][q]) * (peaks[i][p] - peaks[j][q])
                            else:
                                cnt2 += 1
                                tmp2 += (data[j][peaks[j][q]] + offset - data[i][peaks[i][p]]) * (data[j][peaks[j][q]] + offset - data[i][peaks[i][p]]) + (peaks[i][p] - peaks[j][q]) * (peaks[i][p] - peaks[j][q])
                    if cnt1 > 0:
                        peak_dis[i+1][j+1] = tmp1 / cnt1
                    else:
                        peak_dis[i+1][j+1] = constants.MAX_INT

                    if cnt2 > 0:
                        peak_dis[j+1][i+1] = tmp2 / cnt2
                    else:
                        peak_dis[j+1][i+1] = constants.MAX_INT
                elif cost_func == constants.COST_FUNCTION.MAX_PEAK_DIS_2D:
                # data maxpeak is 1
                    peak_dis[i+1][j+1] = (data[i][maxPeak[i]] + offset - data[j][maxPeak[j]]) * (data[i][maxPeak[i]] + offset - data[j][maxPeak[j]]) + (maxPeak[i] - maxPeak[j]) * (maxPeak[i] - maxPeak[j])
                    peak_dis[j+1][i+1] = (data[j][maxPeak[j]] + offset - data[i][maxPeak[i]]) * (data[j][maxPeak[j]] + offset - data[i][maxPeak[i]]) + (maxPeak[i] - maxPeak[j]) * (maxPeak[i] - maxPeak[j])
                elif cost_func == constants.COST_FUNCTION.NEAREST_WEIGHT_PEAK:
                    cnt1 = 0
                    cnt2 = 0
                    for p in range(len(peaks[i])):
                        rec = -1
                        t = 0
                        for q in range(len(peaks[j])):
                            if peaks[i][p] >= peaks[j][q] and peaks[j][q] > peaks[j][rec]:
                                rec = q
                                t += 1
                        if rec != -1:
                            cnt1 += 1
                            peak_dis[i+1][j+1] = (peaks[i][p] - peaks[j][rec]) / t
                    
                    for q in range(len(peaks[j])):
                        rec = -1
                        t = 0
                        for p in range(len(peaks[i])):
                            if peaks[j][q] >= peaks[i][p] and peaks[i][p] > peaks[i][rec]:
                                rec = p
                                t += 1
                        if rec != -1:
                            cnt2 += 1
                            peak_dis[j+1][i+1] = (peaks[j][q] - peaks[i][rec]) / t
                                
                    if cnt1 == 0:
                        peak_dis[i+1][j+1] = constants.MAX_INT
                    # else:
                    #     peak_dis[i+1][j+1] /= cnt1
                    
                    if cnt2 == 0:
                        peak_dis[j+1][i+1] = constants.MAX_INT
                    # else:
                    #     peak_dis[j+1][i+1] /= cnt2
                elif cost_func == constants.COST_FUNCTION.BEST_PERFORM:
                    cnt1 = 0
                    cnt2 = 0
                    for p in range(len(peaks[i])):
                        for q in range(len(peaks[j])):
                            if peaks[i][p] < peaks[j][q]:
                                peak_dis[i+1][j+1] = peaks[j][q] - peaks[i][p]
                                cnt2 += 1
                            else:
                                peak_dis[j+1][i+1] = peaks[i][p] - peaks[j][q]
                                cnt1 += 1
                            if cnt1 == 0:
                                peak_dis[j+1][i+1] = constants.MAX_INT
                            else:
                                peak_dis[j+1][i+1] /= cnt1
                            
                            if cnt2 == 0:
                                peak_dis[i+1][j+1] = constants.MAX_INT
                            else:
                                peak_dis[i+1][j+1] /= cnt2
                elif cost_func == constants.COST_FUNCTION.PEAK_SHIFTING_AND_HIDDEN:
                    # compute the cost of peaks
                    cnt1 = 0
                    cnt2 = 0
                    for p in range(len(peaks[i])):
                        for q in range(len(peaks[j])):
                            if peaks[i][p] < peaks[j][q]:
                                peak_dis[j+1][i+1] += peaks[j][q] - peaks[i][p]
                                cnt2 += 1
                            else:
                                peak_dis[i+1][j+1] += peaks[i][p] - peaks[j][q]
                                cnt1 += 1
                    if cnt1 == 0:
                        peak_dis[i+1][j+1] = constants.MAX_INT
                    else:
                        peak_dis[i+1][j+1] /= cnt1
                    
                    if cnt2 == 0:
                        peak_dis[j+1][i+1] = constants.MAX_INT
                    else:
                        peak_dis[j+1][i+1] /= cnt2
                    # compute the cost of hidden
                    for k in range(M):
                        if data[j][k] > data[i][k] + offset:
                            hidden_dis[i+1][j+1] += 1
                        if data[i][k] > data[j][k] + offset:
                            hidden_dis[j+1][i+1] += 1

        # Normalize the peak and hidden distances
        dis = normalize(peak_dis, hidden_dis, theta)   

        return dis

    # Solve the optimization problem
    def solve(self, nodeIDs, tau, outputFlag, timeLimit, gapTolerance):
        return _ipTSPGurobiLazyCuts(nodeIDs, tau, outputFlag, timeLimit, gapTolerance)

    # Process the result from the solver to generate a readable output
    def deal_ans(self, ans, ridgeNames):
        seq = ans['seq']
        ret = {}
        visited = [False for x in range(len(seq)-1)]
        
        visited[0] = True
        zeroId = -1 
        for i in range(len(seq)):
            if seq[i] == 0:
                zeroId = i
                break
        
        true_seq = []
        for i in range(zeroId+1,len(seq)):
            if visited[seq[i]] == False:
                true_seq.append(seq[i]-1)
                visited[seq[i]] = True
        for i in range(zeroId+1):
            if visited[seq[i]] == False:
                true_seq.append(seq[i]-1)
                visited[seq[i]] = True
        for i in range(len(true_seq)):
            ret[ridgeNames[true_seq[i]]] = len(true_seq) - i - 1
        return ret, true_seq

def normalize(peak_dis, hidden_dis, theta):
    N,M = peak_dis.shape
    dis = np.zeros((N,M))

    peak_maxx = 0
    hidden_maxx = 0
    
    for i in range(N):
        for j in range(M):
            if peak_dis[i][j] != constants.MAX_INT:
                peak_maxx = max(peak_maxx, peak_dis[i][j])
            if hidden_dis[i][j] != constants.MAX_INT:
                hidden_maxx = max(hidden_maxx, hidden_dis[i][j])

    for i in range(N):
        for j in range(M):
            if peak_dis[i][j] != constants.MAX_INT and peak_maxx != 0:
                peak_dis[i][j] /= peak_maxx
            if hidden_dis[i][j] != constants.MAX_INT and hidden_maxx != 0:
                hidden_dis[i][j] /= hidden_maxx
            dis[i][j] = theta * peak_dis[i][j] + (1-theta) * hidden_dis[i][j]

    return dis 

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def find_peaks(data, peakW = 5, H = 0.8):
    N,M = data.shape
    
    # Hyperparameter，the distance of adjacent peaks should be more than 1/10
    D = max(math.floor(M / peakW), 1) 
    
    peaks = []
    for i in range(N):
        peak,_ = sp.signal.find_peaks(data[i], height=H, distance=D)
        peaks.append(peak.tolist())
    
    return peaks

def find_maxPeaks(data):
    N,M = data.shape
    
    
    # maxPeak starts from the 1st point, 0 means nothing
    maxPeak = [0 for x in range(N+1)]
    for i in range(N):
        for j in range(M):
            if data[i][j] > data[i][maxPeak[i+1]]: 
                maxPeak[i+1] = j
    # print(maxPeak)
                
    return maxPeak