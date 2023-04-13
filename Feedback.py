import numpy as np

#accept human feedback / knowledge
#Update option
FREQUENCY = 10
class OptionFeedback():
    def __init__(self, options=None, num_options=None):
        if num_options is None:
            if options is None:
                self.num_options = 0
            else:
                self.num_options = len(options)
        else:
            self.num_options = num_options
        
        self.qCumulative = [0]*self.num_options

        self.feedback = [0]*self.num_options #feed back on an option are 1=good, 0=nothing, -1=bad
        self.interf = np.zeros((self.num_options,self.num_options))
    
    def updateOptions(self, options=None, num_options=None):
        if options is not None:
            self.num_options = len(options) 
        else:
            self.num_options = num_options
        self.refreshFeedback()

    def refreshFeedback(self):
        self.feedback = [0]*self.num_options #feed back on an option are 1=good, 0=nothing, -1=bad
        self.interf = np.zeros((self.num_options,self.num_options))
   
    def getInterOptionFeedback(self, D):
        print("inter option feedback")
        topk = np.argpartition(np.ravel(D),-3)[-3:]
        DnonZ = np.nonzero(D)
        bottomk = np.argpartition(np.ravel(D[DnonZ]),3)[:3]
        #print(topk)
        self.interf = np.zeros((self.num_options,self.num_options))
        n = -3
        merge=[]
        print("Following options are very close. Do you want to merge any: (Merge <optionIds>)", [np.unravel_index(i, D.shape) for i in bottomk])
        answer = input()
        if answer=='':
            for i in bottomk:
                self.interf[np.unravel_index(i, D.shape)[0]][np.unravel_index(i, D.shape)[1]] = n
                print(np.unravel_index(i, D.shape))
                n += 1
        else:
            if len(answer.split(" ")) > 2:
                o = [int(x) for x in answer.split(" ")[1:]]
                merge = o
        
        return self.interf, merge
        




    def getOptionFeedback(self, uncertainty=None, dist=None, all=False, step=0, D=None):
        
        if uncertainty is not None and dist is not None:
            q=0
            uncertainty = np.array(uncertainty)
            dist = np.array(dist)
            uind = np.argpartition(uncertainty,-3)[-3:]
            distind = np.argpartition(dist, -3)[-3:]
            #print(uind, distind)
            intersect = [value for value in uind if value in distind]
            print(intersect)
            if len(intersect) > 0:
                topintersect = np.argmax(dist[intersect])
                print("top intersect=",dist[intersect],dist[intersect[topintersect]],intersect[topintersect])
                #print("The following prototypes are unstable. Please suggest:", intersect[topintersect])
                q = intersect[topintersect]
            else: 
                #print("The following prototypes are unstable. Please suggest:", uind[np.argmax(uncertainty[uind])])
                q= uind[np.argmax(uncertainty[uind])]
            print("Q:",q)
            #self.qCumulative[q] +=1
        print("STEP: ",step)
        ind = -1
        answer = 0 
        splits=[]
        merge = []
        if all == False: 
            if step % FREQUENCY == 0:
                print("The following prototypes have an unstable trend. Please suggest: (1=good, 0=nothing, -1=bad OR \"split <option#>\")", q)
                ind = q # np.argmax(self.qCumulative)
                
                
        #else:
            #print("The following prototypes are unstable. Please suggest:", q)
            #ind = q
        

        if ind >-1:
            answer = input()
            if D is not None:
                self.interf, merge = self.getInterOptionFeedback(D=D)
            if answer.startswith("split"):
                if len(answer.split(" ")) > 1:
                    ind = int(answer.split(" ")[1])
                    splits.append(ind)
            else:
                try:
                    answer = int(answer)
                    self.feedback[ind] += answer
                except:
                    print("input format is not correct.")
        
        return np.array(self.feedback), splits, np.array(self.interf), merge