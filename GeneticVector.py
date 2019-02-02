import random
import keras

neurons=[16,32,64,128]
actfunc=['relu','sigmoid','tanh']
optfunc=[keras.optimizers.Adam(),keras.optimizers.SGD(),keras.optimizers.Adagrad(),keras.optimizers.Adamax()]
batchsize=[64,128,256]
epoch=[30,40,50]

class newGenClass():
    def __init__(self):
        self.neuron1=None
        self.neuron2=None
        self.neuron3=None
        self.neuron4=None
        self.act1=None
        self.act2=None
        self.act3=None
        self.act4=None
        self.opt=None
        self.bs=None
        self.ep=None
        self.vec1=None

    def selection(self):
        self.neuron1=random.choice(neurons)
        self.neuron2=random.choice(neurons)
        self.neuron3=random.choice(neurons)
        self.neuron4=random.choice(neurons)
        self.act1=random.choice(actfunc)
        self.act2=random.choice(actfunc)
        self.act3=random.choice(actfunc)
        self.act4=random.choice(actfunc)
        self.opt=random.choice(optfunc)
        self.bs=random.choice(batchsize)
        self.ep=random.choice(epoch)
        return ([self.neuron1,self.neuron2,self.neuron3,self.neuron4,self.act1,self.act2,self.act3,self.act4,self.opt,self.bs,self.ep])

    def crossover(self):
        vec1=self.selection()
        vec2=self.selection()
        n = random.randint(0,8)
        a = vec1[0:n]
        b = vec1[n:]
        c = vec2[0:n]
        d = vec2[n:]
        a.extend(d)
        c.extend(b)
        self.vec1=a
        
    def mutation(self):
        cr=self.vec1
        mut_percent=30
        x=random.randint(0,100)
        if(x>mut_percent):
            return(cr)
        else:
            n=random.randint(0,10)
            if(n<=3):
                newNeu=list(neurons)
                newNeu.remove(cr[n])
                cr[n]=random.choice(newNeu)
            elif(n<=7):
                newfuncs=list(actfunc)
                newfuncs.remove(cr[n])
                cr[n]=random.choice(newfuncs)
            elif(n==8):
                cr[n]=random.choice(optfunc)
            elif(n==9):
                newBatch=list(batchsize)
                newBatch.remove(cr[n])
                cr[n]=random.choice(newBatch)
            elif(n==10):
                newEpochs=list(epoch)
                newEpochs.remove(cr[n])
                cr[n]=random.choice(newEpochs)
            return(cr)