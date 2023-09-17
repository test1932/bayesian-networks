class LabelledTensor:
    def __init__(self, labels):
        """
        labelled tensor constructor
        """
        self.__dataFrame = LabelledTensor.makeDim(labels)

    def normalise(self):
        self.normaliseRec(self.__dataFrame)

    def normaliseRec(self, dataframe):
        a = list(dataframe.values())
        if type(a[0]) == int:
            s = sum(a)
            for i in dataframe:
                dataframe[i] /= s
            return True
        for i in dataframe:
            self.normaliseRec(dataframe[i])

    def getDataFrame(self):
        return self.__dataFrame

    def __str__(self):
        return str(self.__dataFrame)

    @staticmethod
    def makeDim(labels):
        if not labels:
            return 1
        
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