class LabelledTensor:
    def __init__(self, labels):
        """
        labelled tensor constructor
        """
        self.__dataFrame = LabelledTensor.makeDim(labels)
        self.__levels = labels[-1]

    def normalise(self):
        f = lambda sumLayer, val: val / sumLayer
        self.normaliseRec(self.__dataFrame, f)

    def bayesNormalise(self, iss, n):
        prior = 1 / len(self.__levels)
        f = lambda sumLayer, val: prior * (iss / (iss + n))\
            + (val / sumLayer) * (n / (iss + n))
        self.normaliseRec(self.__dataFrame, f)

    def normaliseRec(self, dataframe, func):
        a = list(dataframe.values())
        if type(a[0]) == int:
            s = sum(a)
            for i in dataframe:
                dataframe[i] = func(s, dataframe[i])
            return True
        for i in dataframe:
            self.normaliseRec(dataframe[i], func)

    def getDataFrame(self):
        return self.__dataFrame

    def __str__(self):
        return str(self.__dataFrame)

    @staticmethod
    def makeDim(labels):
        if not labels:
            return 0
        
        dataframe = dict()
        for i in labels[0]:
            dataframe[i] = LabelledTensor.makeDim(labels[1:])
        return dataframe


if __name__ == '__main__':
    a = LabelledTensor([
            ["M","F"],
            ["young", "adult", "old"],
            ["emp", "high"]
        ])