# -*- coding: utf-8 -*-
"""
@name: QIHLOA
"""

# In[]
import numpy as np
import n_model as md
import os
import random

def fit_fun(param, X):  # fit function, here model training
    # get model parameters
    train_data = param['data']
    train_label = param['label']
    model = md.DenseNet201(X)
    # Pass the learning rate parameter to be optimized
    res_model = model.model_create(X[0])
    history = res_model.fit(train_data, train_label, epochs=5, batch_size=16, validation_split=0.1)
    # Get the minimum loss, become learning_rate when loss is minimum
    val_loss = min(history.history['val_loss'])
    val_loss = np.float64(val_loss)
    return val_loss


class QIHLOA():
    def __init__(self, model_param, hloa_param):
        self.SearchAgents_no = hloa_param['SearchAgents_no']
        self.lb = hloa_param['lb']
        self.ub = hloa_param['ub']
        self.dim = hloa_param['dim']
        self.Max_iter = hloa_param['Max_iter']
        self.model_param = model_param

    def alpha_melanophore(self, fit, vMin, vMax):
        o = np.zeros(len(fit))
        for i in range(len(fit)):
            o[i] = (vMax - fit[i]) / (vMax - vMin)
        return o

    def getColor(self, colorPalette):
        band = True
        c1, c2 = 0, 0
        while band:
            c1 = colorPalette[np.random.randint(0, 30)]
            c2 = colorPalette[np.random.randint(0, 30)]
            if c1 != c2:
                band = False
        return c1, c2

    def getBinary(self):
        if np.random.random() < 0.5:
            val = 0
        else:
            val = 1
        return val

    def checkO(self, o):
        o = np.array(o)
        o[o < 0] = np.abs(o[o < 0])
        for i in range(len(o)):
            if o[i] < self.lb[i] or o[i] > self.ub[i]:
                o[i] = random.uniform(self.lb[i], self.ub[i])
        return o

    def R(self, NP):
        band = True
        r1, r2, r3, r4 = 0, 0, 0, 0
        while band:
            r1 = np.round(1 + (NP - 1) * np.random.rand())
            r2 = np.round(1 + (NP - 1) * np.random.rand())
            r3 = np.round(1 + (NP - 1) * np.random.rand())
            r4 = np.round(1 + (NP - 1) * np.random.rand())
            if r1 == NP: r1 -= 1
            if r2 == NP: r2 -= 1
            if r3 == NP: r3 -= 1
            if r4 == NP: r4 -= 1
            if (r1 != r2) and (r2 != r3) and (r1 != r3) and (r4 != r3) and (r4 != r2) and (r1 != r4):
                band = False
        return int(r1), int(r2), int(r3), int(r4)

    def mimicry(self, Xbest, X, Max_iter, SearchAgents_no, t):
        colorPalette = np.array(
            [0, 0.00015992, 0.001571596, 0.001945436, 0.002349794, 0.00353364, 0.0038906191, 0.003906191, 0.199218762,
             0.19999693, 0.247058824, 0.39999392, 0.401556397, 0.401559436, 0.498039216, 0.498046845, 0.499992341,
             0.49999997, 0.601556397, 0.8, 0.900000447, 0.996093809, 0.996109009, 0.996872008, 0.998039245, 0.998046875,
             0.998431444, 0.999984801, 0.999992371, 1])
        Delta = 2
        r1, r2, r3, r4 = self.R(SearchAgents_no)
        c1, c2 = self.getColor(colorPalette)
        o = Xbest + (Delta - Delta * t / Max_iter) * (
                c1 * ((np.sin(X[r1, :]) - np.cos(X[r2, :])) - ((-1) ** self.getBinary()) * c2 * np.cos(
            X[r3, :])) - np.sin(
            X[r4, :]))
        o = self.checkO(o)
        return o

    def shootBloodstream(self, Xbest, X, Max_iter, t):
        g = 0.009807  # 9.807 m/s2   a kilometros    =>  0.009807 km/s2
        epsilon = 1E-6
        Vo = 1  # 1E-2
        Alpha = np.pi / 2

        o = (Vo * np.cos(Alpha * t / Max_iter) + epsilon) * Xbest + (
                Vo * np.sin(Alpha - Alpha * t / Max_iter) - g + epsilon) * X
        o = self.checkO(o)
        return o

    def CauchyRand(self, m, c):
        cauchy = c * np.tan(np.pi * (np.random.rand() - 0.5)) + m
        return cauchy

    def randomWalk(self, Xbest, X):
        e = self.CauchyRand(0, 1)
        walk = -1 + 2 * np.random.rand()  # -1 < d < 1
        o = Xbest + walk * (0.5 - e) * X

        o = self.checkO(o)
        return o

    def Skin_darkening_or_lightening(self, Xbest, X, SearchAgents_no):
        darkening = [0.0, 0.4046661]
        lightening = [0.5440510, 1.0]

        dark1 = darkening[0] + (darkening[1] - darkening[0]) * np.random.rand()
        dark2 = darkening[0] + (darkening[1] - darkening[0]) * np.random.rand()
        light1 = lightening[0] + (lightening[1] - lightening[0]) * np.random.rand()
        light2 = lightening[0] + (lightening[1] - lightening[0]) * np.random.rand()

        r1, r2, r3, r4 = self.R(SearchAgents_no)

        if self.getBinary():
            o = Xbest + light1 * np.sin((X[r1, :] - X[r2, :]) / 2) - ((-1) ** self.getBinary()) * light2 * np.sin(
                (X[r3, :] - X[r4, :]) / 2)
        else:
            o = Xbest + dark1 * np.sin((X[r1, :] - X[r2, :]) / 2) - ((-1) ** self.getBinary()) * dark2 * np.sin(
                (X[r3, :] - X[r4, :]) / 2)
        o = self.checkO(o)
        return o

    def remplaceSearchAgent(self, Xbest, X, SearchAgents_no):
        band = True
        r1, r2 = 0, 0
        while band:
            r1 = np.round(1 + (SearchAgents_no - 1) * np.random.rand())
            r2 = np.round(1 + (SearchAgents_no - 1) * np.random.rand())
            if r1 == SearchAgents_no: r1 -= 1
            if r2 == SearchAgents_no: r2 -= 1
            if r1 != r2:
                band = False
        r1, r2 = int(r1), int(r2)

        o = Xbest + (X[r1, :] - ((-1) ** self.getBinary() * X[r2, :]) / 2)
        o = self.checkO(o)
        return o

    def checkBoundaries(self, position):
        position = np.maximum(position, self.lb)
        position = np.minimum(position, self.ub)
        return position

    def elite_differential_mutation(self,x_t, population, x_best, F=0.5):
        # 随机选择两个不同的个体作为差分操作的基础
        idxs = np.random.choice(len(population), 2, replace=False)
        x1, x2 = population[idxs[0]], population[idxs[1]]

        # 生成x_t+1的精英差分变异
        x_new = x_t + F * (x_best - x_t) + F * (x1 - x2)

        return x_new

    def run(self):
        Positions = np.random.uniform(self.lb, self.ub, (self.SearchAgents_no, self.dim))
        Positions = np.array(Positions)
        Fitness = np.zeros(self.SearchAgents_no)

        for i in range(Positions.shape[0]):
            Fitness[i] = fit_fun(self.model_param, Positions[i, :])

        minIdx = np.argmin(Fitness)
        vMin = Fitness[minIdx]
        theBestVct = Positions[minIdx, :]
        maxIdx = np.argmax(Fitness)
        vMax = Fitness[maxIdx]

        Convergence_curve = np.zeros(self.Max_iter)
        Convergence_curve[0] = vMin

        alphaMelanophore = self.alpha_melanophore(Fitness, vMin, vMax)
        self.v = np.zeros((self.SearchAgents_no, self.dim))
        self.v = np.array(self.v)

        for t in range(1, self.Max_iter + 1):
            print(f"{t} iteration")
            for r in range(self.SearchAgents_no):

                if 0.5 < np.random.rand():
                    self.v[r, :] = self.mimicry(theBestVct, Positions, self.Max_iter, self.SearchAgents_no, t)
                else:
                    if t % 2 == 1:
                        self.v[r, :] = self.shootBloodstream(theBestVct, Positions[r, :], self.Max_iter, t)
                    else:
                        self.v[r, :] = self.randomWalk(theBestVct, Positions[r, :])
                Positions[maxIdx, :] = self.Skin_darkening_or_lightening(theBestVct, Positions, self.SearchAgents_no)

                self.v[r, :] = self.checkBoundaries(self.v[r, :])

                Fnew = fit_fun(self.model_param, self.v[r, :])

                if alphaMelanophore[r] <= 0.3:
                    x_new2 = self.remplaceSearchAgent(theBestVct, Positions, self.SearchAgents_no)
                    x_new2 = self.checkO(x_new2)
                    Fnew2 = fit_fun(self.model_param, x_new2)
                    if Fnew2<Fnew:
                        Fnew=Fnew2
                        self.v[r, :]=x_new2

                self.now_iter_x_best = self.v[r, :]
                self.now_iter_y_best = Fnew
                # 精英差分变异
                X_new=self.elite_differential_mutation(self.now_iter_x_best, Positions, vMin)
                X_new=self.checkO(X_new)
                Y_new=fit_fun(self.model_param,X_new)

                if Y_new<self.now_iter_y_best:
                    self.now_iter_x_best,self.now_iter_y_best=X_new,Y_new

                if self.now_iter_y_best <= vMin:
                    theBestVct = self.now_iter_x_best
                    vMin = self.now_iter_y_best

            maxIdx = np.argmax(Fitness)
            vMax = Fitness[maxIdx]
            alphaMelanophore = self.alpha_melanophore(Fitness, vMin, vMax)
            Convergence_curve[t-1] = vMin

        return vMin, theBestVct