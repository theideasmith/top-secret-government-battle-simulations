import random
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

class Dice:
    def __init__(self, dicemax, ndice):
        self.dicemax = dicemax
        self.ndice = ndice
        self.initial_ndice = ndice
        self.inital_dicemax = dicemax

    def roll(self):
        dice = []
        for i in range(self.ndice):
            dice.append(random.randint(1,self.dicemax))
        return sorted(dice, reverse=True)

    def reset(self):
        self.ndice = self.initial_ndice
        self.dicemax = self.inital_dicemax

class Player:
    # Then players can subclass to implement different strategies
    def __init__(self, dice, number_of_people):
        self.dice = dice
        self.npeople = number_of_people
        self.initial_people = number_of_people

    def roll(self):
        return self.dice.roll()

    def ready(self, opponent):
        return True

    def reset(self):
        self.npeople = self.initial_people
        self.dice.reset()

class Attacker(Player):
    def roll(self):
        if self.npeople in [3,2]:
            self.dice.ndice=2
        if self.npeople ==1:
            self.dice.ndice=1
        return Player.roll(self)

    def ready(self, opponent):
        return self.npeople > 1

class Defender(Player):
    def ready(self, opponent):
        return self.npeople > 0

class Offensive:
    def __init__(self, attacker, defender, toprint=False):
        self.a = attacker
        self.d = defender
        self.toprint = toprint
        self.niter = 0
        self.menA = []
        self.menD = []

    def extension(self,rollA, rollD):
        return np.array([0,0])

    def compareDice(self, diceA, diceD):
        if diceA > diceD:
            return (0,-1)
        elif diceA <= diceD:
            return (-1,0)

    def standardBattleContract(self, rolla, rolld):
        minind = min(len(rolla), len(rolld))
        aligneda = rolla[:minind]
        alignedd = rolld[:minind]
        contract = map(
            lambda x: self.compareDice(*x),
            zip(aligneda, alignedd))
        lossA = 0
        lossD = 0
        for c in contract:
            lossA += c[0]
            lossD += c[1]
        return np.array([lossA, lossD])

    def show(self, loss, rolla, rolld):
        lossA, lossD = loss
        print "Dice: "
        print "A: {}".format(rolla)
        print "D: {}".format(rolld)
        print "   +++   "
        print "Loss: "
        print "A: {}".format(lossA)
        print "B: {}".format(lossD)
        print "   +++   "
        print "Men:"
        print "A: {}".format(self.a.npeople)
        print "D: {}".format(self.d.npeople)
        print "------------------"

    def didWin(self):
        return self.d.npeople==0

    def canIterate(self):
        can= self.a.ready(self.d) and self.d.ready(self.a)
        return can

    def iterate(self):
        rolld = self.d.roll()
        rolla = self.a.roll()
        loss = self.extension(rolla, rolld) \
                + self.standardBattleContract(rolla, rolld)
        self.a.npeople += loss[0]
        self.d.npeople += loss[1]
        if self.toprint:
            self.show(loss, rolla,rolld)
        self.menA.append(self.a.npeople)
        self.menD.append(self.d.npeople)
        self.niter+=1

    def reset(self):
        self.a.reset()
        self.d.reset()
        self.niter = 0
        self.menA = []
        self.menD = []

    def simulate(self):
        while self.canIterate():
            self.iterate()

        return self.a.npeople, self.d.npeople

class LipshitzianOffensive(Offensive):
    def extension(self, rollA, rollD):
        allgreater = min(rollD) >= max(rollA)
        if allgreater:
            return np.array([-1,0])
        else:
            return np.array([0,0])

def main():
    T = 100
    battle = LipshitzianOffensive(
        Attacker(Dice(6,3), 3),
        Defender(Dice(6,2), 2),
        toprint=False)

    battle.reset()
    resA = []
    resD = []
    histA = []
    histD = []
    for i in range(T):
        ra, rd = battle.simulate()
        histA.append(battle.menA)
        histD.append(battle.menD)
        resA.append(ra)
        resD.append(rd)
        battle.reset()

    histmatA = np.zeros((T, max(map(len, histA)) ))*np.nan
    histmatD = np.zeros((T, max(map(len, histD)) ))*np.nan
    for i in xrange(T):
        histmatA[i,:len(histA[i])] = histA[i]
        histmatD[i, :len(histD[i])] = histD[i]


    histmatA = ma.array(histmatA, mask=np.isnan(histmatA))
    histmatD = ma.array(histmatD, mask=np.isnan(histmatD))

    muA = ma.mean(histmatA, axis=0)
    muD = ma.mean(histmatD, axis=0)
    sigmaA = ma.std(histmatA, axis=0)
    sigmaD = ma.std(histmatD, axis=0)
    t = np.arange(muA.shape[0])

    pointWin = np.argwhere(muD==0)
    fig, ax = plt.subplots(1)
    ax.plot(t, muA, lw=2, label='mean trajectory attacker', color='blue')
    ax.plot(t, muD, lw=2, label='mean trajectory defender', color='red')
    ax.fill_between(t, muA+sigmaA, muA-sigmaA, facecolor='blue', alpha=0.5)
    ax.fill_between(t, muD+sigmaD, muD-sigmaD, facecolor='red', alpha=0.5)

    ax.set_title("Npeople trajectories for Offensive")
    ax.legend(loc='lower left')
    ax.set_xlabel("Time [iterations]")
    ax.set_ylabel("Number of men")
    plt.savefig("./npeople_trajectories_A:{},{},{}_D:{},{},{}.png".format(
        battle.a.dice.dicemax,
        battle.a.dice.ndice,
        battle.a.initial_people,
        battle.d.dice.dicemax,
        battle.d.dice.ndice,
        battle.d.initial_people
    ))
    resA = np.array(resA)
    resD = np.array(resD)

    plt.figure(figsize=(10,10))
    plt.hist2d(resA, resD, alpha=.6, normed=True, bins=[50,50])
    plt.colorbar()
    plt.ylabel("Defender")
    plt.title("Distribution of Attack Outcomes for 1000 Runs")
    plt.xlabel("Attacker")
    plt.savefig("./results.png")
    print "On average"
    print "Attacker"; print np.mean(resA)
    print "Defender"; print np.mean(resD)
    print "Attacker wins by"; print np.mean(resA - resD)
    return resA, resD

if __name__=="__main__":
    main()
