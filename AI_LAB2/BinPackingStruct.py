class BinPackingStruct:
    def __int__(self):
        self.N = 0
        self.arr = []
        self.arr2 = []
        self.fitness = -1
        self.specynum = 0
        self.diversity=0
        self.binsNum = 0
        self.classy=0
        self.frequency = 0



    def updateN(self, number):
        self.N = number
        self.arr = []
        self.arr2 = []
        for i in range(number):
            self.arr.append(0)
            self.arr2.append(0)
        self.binsNum = 0
        self.fitness = -1
        self.specynum=0
