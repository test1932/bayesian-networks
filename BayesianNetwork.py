import random

class BayesianNetwork:
    """
    Class representing Bayesian Network of nodes as a DAG of 
    DirectedAcyclicNode objects.
    """
    class DAGnode:
        def __init__(self, identifier:str, parents:dict):
            self.__identifier = identifier
            self.__pathsTo = set()
            self.__val = None
            self._parents = parents

            self.__markovBlanket = None
            self.__validMarkov = False

        def getVal(self):
            return self.__val
        
        def addParent(self, parent):
            self._parents[parent.getID()] = parent
            self.__validMarkov = False
            return self
        
        def getParents(self):
            return self._parents

        def __iter__(self):
            return self.__pathsTo
        
        def addArcTo(self, nodeTo):
            self.__pathsTo.add(nodeTo)
            self.__validMarkov = False
            return self
        
        def removeArcTo(self, nodeToID):
            self.__pathsTo = set(filter(\
                lambda x: x.getID() != nodeToID, self.__pathsTo))
            self.__validMarkov = False
            return self

        def getID(self):
            return self.__identifier
        
        def getPathsTo(self):
            return self.__pathsTo
        
        def getMarkovBlanket(self) -> set:
            if self.__validMarkov:
                return self.__markovBlanket
            
            spouses = set()
            for i in self.__pathsTo:
                spouses = spouses.union(set(i.getParents().values()))
            self.__markovBlanket =  set(self._parents.values())\
                .union(self.__pathsTo).union(spouses).difference({self})
            self.__validMarkov = True
            return self.__markovBlanket
        
        # DEBUG
        def __str__(self) -> str:
            return f'<{self.__identifier}, {[i.getID() for i in self.__pathsTo]}>'
        
        def __repr__(self) -> str:
            return self.__str__()
    

    def __init__(self, nodes:list[DAGnode]):
        """
        Constructor method. Please use fromModelString(), please do not call
        this directly - for your sake (it doesn't check for cycles).TODO
        @param Nodes - represents the nodes in the DAG. 
        @param Data - represents the data used to form the Bayesian Network.
        """
        self._nodesDict = dict([(node.getID(),node) for node in nodes])
        self._nodes = set(nodes)
        self.__avgMarkovBlanket = None
        self.__validMarkov = False

    def addNode(self, node):
        self._nodes.add(node)
        self._nodesDict[node.getID()] = node
        self.__avgMarkovBlanket *= ((x:=len(self._nodes)) - 1) / x
        return self
    
    def getNode(self, strID):
        return self._nodesDict[strID]

    @property
    def avgMarkovBlanket(self) -> int:
        sum_Blanket = sum([len(n.getMarkovBlanket()) for n in self._nodes])
        self.__avgMarkovBlanket = sum_Blanket / len(self._nodes)\
            if not self.__validMarkov else self.__avgMarkovBlanket
        self.__validMarkov = True
        return self.__avgMarkovBlanket
    
    @property
    def avgBranchingFactor(self) -> int:
        totalBranches = sum([len(i.getParents()) for i in self._nodes])
        return totalBranches / len(self._nodes)
    
    def addArc(self, arc):
        """
        Adds an arc from arc[0] to arc[2].
        @param arc - represents pair of node IDs. i.e. ('A', 'R') or ['A', 'R']
        """
        self._nodesDict[arc[0]].addArcTo(self._nodesDict[arc[1]])
        if BayesianNetwork.__cycleStartsAt(self._nodesDict[arc[0]]):
            self._nodesDict[arc[0]].removeArcTo(arc[1])
            raise AssertionError("resulting graph contains cycles")
        self.__isMarkovBlanketSizeValid = False
        return self
    
    def getNodes(self) -> list[DAGnode]:
        return " ".join([*self._nodes])
    
    def getArcs(self) -> list:
        arcs = []
        for i in self._nodes:
            for j in i.getPathsTo():
                arcs.append(f'"{i.getID()}" -> "{j.getID()}"')
        return "\n".join(arcs)
    
    def __str__(self) -> str:
        """
        overridden __str__ method for string representation of Baysian Network
        """
        header = f'Avg markov blanket: {round(self.avgMarkovBlanket, 2)}\n'
        nodes = "\n".join(map(lambda x: str(x), list(self._nodes)))
        return header + nodes
    
    def __repr__(self) -> str:
        """
        overridden __repr__ method to print Bayesian Network
        """
        return f'Baysian Network with {len(self._nodes)} nodes'
    
    def __eq__(self, __value: object) -> bool:
        pass #TODO

    @staticmethod
    def __cycleStartsAt(node):
        """
        Static method for checking if a particular node has a cycle to itself.
        No idea about structure, so best first search or A* is impossible.
        @param node - DirectedAcyclicNode representing node to check for.
        """
        frontier = [*node.getPathsTo()]
        while frontier:
            curNode = frontier[0]
            del frontier[0]
            if curNode == node:
                return True
            frontier = frontier + list(curNode.getPathsTo())
        return False
    
    @staticmethod
    def __addDependentNode(nodes:dict, nodeStr:str, classType):
        """
        for use by fromModelString().
        """
        components = nodeStr.split("|")
        dependents = components[1].split(":")
        nodes[components[0]] = classType(components[0], \
            dict([(depI, nodes[depI]) for depI in dependents]))
        
        # add arcs to nodes
        for i in dependents:
            nodes[i].addArcTo(nodes[components[0]])
            if BayesianNetwork.__cycleStartsAt(nodes[i]):
                raise AssertionError("resulting graph contains cycles")

    @staticmethod
    def fromModelString(nodeIDs:str, NodeType, NetworkType):
        """
        produces Bayesian Network from string.
        E.g. "A, S, E|A:S, O|E, R|E, T|O:R"
        produces: A, S, S -> E <- S, E -> O, E -> R, O -> T <- R
        Similar to the equivalent method in R. Just use yer noggin'
        """

        nodes = dict()
        nodeInputs = nodeIDs.replace(" ", "").split(",")
        for nodeStr in nodeInputs:
            if not '|' in nodeStr:
                nodes[nodeStr] = NodeType(nodeStr, dict())
                continue
            BayesianNetwork.__addDependentNode(nodes, nodeStr, NodeType)
        return NetworkType(nodes.values())
    
if __name__ == '__main__':
    network = BayesianNetwork.fromModelString("A, S, E|A:S, O|E, R|E, T|O:R", 
                                              BayesianNetwork.DAGnode, 
                                              BayesianNetwork)
    print(network.getArcs())