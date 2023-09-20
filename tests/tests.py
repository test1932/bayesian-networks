import pyproptest.basicGenerators as bg
import pyproptest.testing as pytest

def test(generators):
    def func(f):
        def innerF():
            return (generators, f)
        return innerF
    return func


class pathTests:
    @staticmethod
    @test([bg.intListArb(10,-100,100)])
    def prop_equalLength(i):
        return len(i) == len(sorted(i))

    @staticmethod
    @test([bg.intListArb(10,-100,100)])
    def prop_sortedResult(i):
        res = sorted(i)
        return all([res[i] <= res[i + 1] for i in range(len(res) - 1)])
    
    @staticmethod
    @test([bg.intListArb(10,-100,100)])
    def prop_containsSameElements(i):
        res = sorted(i)
        ifreq = {x:i.count(x) for x in i}
        return ifreq == {x:res.count(x) for x in res}


testerObj = pytest.tester(classes = [pathTests])
testerObj.runTests()