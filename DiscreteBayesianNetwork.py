from BayesianNetwork import BayesianNetwork
from LabelledTensor import LabelledTensor

class DiscreteBayesianNetwork(BayesianNetwork):
    """
    class for making discrete bayesian networks.
    """
    class DiscreteDAGnode(BayesianNetwork.DAGnode):
        """
        class representing nodes in a discrete bayesian network.
        """
        def __init__(self, identifier:str, parents:dict):
            """
            constructor method for creating discrete distribution nodes.
            """
            super().__init__(identifier, parents)
            self.__levels = None
            self._probs = None # LabelledTensor

        def setLevels(self, levels):
            """
            sets the output possibilities of the node.
            """
            self.__levels = levels
            return self

        def getLevels(self):
            """
            getter for the output possibilities of a node.
            """
            return self.__levels

        def setProbs(self, probs):
            """
            sets the probability distribution of a node.
            """
            self._probs = probs
            return self

        def recalculateProbs(self):
            """
            automatically restructures the probability distribution structure
            of a node based on its parents' output possibilities.
            """
            labels = sorted(list(self._parents.values()), 
                            key = lambda x: x.getID())
            labels.append(self)
            self._probs = LabelledTensor([i.getLevels() for i in labels])

        def __str__(self):
            """
            overridden __str__() method for debugging, also displays
            probability distribution.
            """
            probsStr = str(self._probs)
            levelStr = str(self.__levels)
            superStr = super().__str__()
            return f'{superStr:<20} {levelStr:<40} {probsStr}'
        
    
    def __init__(self, nodes):
        """
        constructor for making discrete bayesian networks.
        """
        super().__init__(nodes)

    def setLevels(self, levelPairs):
        
        for i in levelPairs:
            self._nodesDict[i[0]].setLevels(i[1])
        return self
    
    def recalcProbs(self):
        for node in self._nodes:
            node.recalculateProbs()
        return self
    
    def customFit(self, probs):
        self.recalcProbs()
        #TODO

        for node in self._nodes:
            node._probs.normalise()

    def fit(self, data, method = "mle", iss = 10):
        self.recalcProbs()
        if len(data) == 0:
            raise ValueError("please provide at least a header")
        
        namesToColumns = dict(zip(data[0], range(len(data[0]))))
        if (x:=method.lower()) == 'mle':
            self.mleFit(data, namesToColumns)
        elif x == 'bayes':
            self.bayesianFit(data, namesToColumns, iss)
        
        return self

    def fitSum(self, data, IDtoIndexMap):
        for node in self._nodes:
            f = sorted(list(node.getParents().keys()))
            for line in data[1:len(data) - 1]:
                nestedDict = node._probs.getDataFrame()
                for key in f:
                    nestedDict = nestedDict[line[IDtoIndexMap[key]]]
                nestedDict[line[IDtoIndexMap[node.getID()]]] += 1

    def mleFit(self, data, IDtoIndexMap):
        self.fitSum(data, IDtoIndexMap)
        for node in self._nodes:
            node._probs.normalise()

    def bayesianFit(self, data, IDtoIndexMap, iss):
        self.fitSum(data, IDtoIndexMap)
        for node in self._nodes:
            node._probs.bayesNormalise(iss, len(data) - 1)
    
    @staticmethod
    def DiscreteBayesianFromString(s:str):
        return BayesianNetwork.fromModelString(s, 
            DiscreteBayesianNetwork.DiscreteDAGnode,
            DiscreteBayesianNetwork)


if __name__ == '__main__':
    a = DiscreteBayesianNetwork.DiscreteBayesianFromString(\
        "A, S, E|A:S, O|E, R|E, T|O:R")
    a.setLevels([
        ("A",["young", "adult", "old"]),
        ("S",["M", "F"]),
        ("E",["high", "uni"]),
        ("O",["emp", "self"]),
        ("R",["small", "big"]),
        ("T",["car", "train", "other"])
    ])

    # a.customFit([
    #     ("A", np.array([0.3,0.5,0.2])),
    #     ("S", np.array([0.6,0.4])),
    #     ("E", np.array([0.75,0.25,0.72,0.28,0.88,0.12,0.64,0.36,0.7,0.3,0.9,0.1]).reshape(2,3,2)),
    #     ("O", np.array([0.96,0.04,0.92,0.08]).reshape(2,2)),
    #     ("R", np.array([0.25,0.75,0.20,0.80]).reshape(2,2)),
    #     ("T", np.array([0.48,0.42,0.10,0.56,0.36,0.08,0.58,0.24,0.18,0.70,0.21,0.09]).reshape(2,2,3))
    # ])

    survey = list(map(lambda x: x.split(","), open("survey.txt", "r").read().split("\n")))
    a.fit(survey, method = "bayes", iss = 20)
    print(a)