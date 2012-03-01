import random, os, string, sys, math, operator
from porterStemmer import PorterStemmer
import emailLib 
import clusteringLib 
import getopt

emailFilters = [emailLib.AlnumFilter(), emailLib.StopwordFilter(), emailLib.HeaderFilter()]
totalFreqDist = {}

class NewsGroups :
    def __init__(self, mainFolder, nbDocumentPerGroup, nbCenters):
        self.__mainFolder = mainFolder
        self.__nbDocumentPerGroup = nbDocumentPerGroup
        self.documents = {}
        self.nbCenters = nbCenters
        
    def loadDocuments(self):
        if (os.access(self.__mainFolder, os.F_OK or os.R_OK)):
            mainCounter = 0
            groupCounter = 0
            check = 0
            for root, dirs, files in os.walk(self.__mainFolder):
                fileGroupCounter = 0
                #for name in files:
                tempFile = files
                while (fileGroupCounter < self.__nbDocumentPerGroup) and len(tempFile)>0 :
                    name = random.choice(tempFile)
                    tempFile.remove(name)
                    filename = os.path.join(root, name)
                                       
                    self.documents[mainCounter] = Document(filename,root)
                    fileGroupCounter += 1
                    mainCounter  += 1
                if fileGroupCounter > 0 : groupCounter +=1
                if groupCounter==self.nbCenters :
                    break;

class KCenter :
    def __init__(self, documents, frequencyDistribution):
        self.documents = documents
#        for fd in frequencyDistribution :
#            frequencyDistribution[fd] = random.randint(0,1000) 
        self.coord = frequencyDistribution
        self.cluster = []
        
    def calcMeanCluster(self):
        newFrequencyDistribution = {}
        i = 0
        for c in self.cluster :
            if i==0:
                newFrequencyDistribution= self.documents[c].frequencyDistribution
            else :
                for f in self.documents[c].frequencyDistribution :
                    if f in newFrequencyDistribution :
                        newFrequencyDistribution[f] += self.documents[c].frequencyDistribution[f]
                    else :
                        newFrequencyDistribution[f] = self.documents[c].frequencyDistribution[f]
            i+=1
        total = float(len(self.cluster))
        self.coord = dict([(f,newFrequencyDistribution[f]/total) for f in newFrequencyDistribution])
                
    def getTermFrequency(self):
        return computeTermFrequency(self.coord)
    
    def getTFIDF(self):
        return computeTFIDF(self.coord)

class Document :
    def __init__(self, fileName, group):
        self.fileName = fileName
        self.actualGroup = group
        self.frequencyDistribution = self.__computeFrequencyDistribution()
        self.centerID = None

### use count bag of words
    def __computeFrequencyDistribution(self):
        counts = {}
        reader = emailLib.EmailReader(self.fileName)
        e = emailLib.EmailFilter(reader.getWordList(), emailFilters)
        p = PorterStemmer()
        for word in e.wordlist :
            #stemming
            word = p.stem(word.lower(),0,len(word)-1)
            # calc TF
            #word = word.lower()
            if word in counts :
                counts[word] += 1
            else :
                counts[word] = 1
            # calc for idf
            if word in totalFreqDist:
                totalFreqDist[word] += 1
            else :
                totalFreqDist[word] = 1
        return dict([(w, counts[w]) for w in counts if counts[w] > 1 ])
        #if len(counts) < 20 :
        #    return counts
        #else :
#            #get top 20 only
        #    sortedCounts = sorted(counts.items(), key=operator.itemgetter(1))
        #    return dict(sortedCounts[-20:])
   
    def getTermFrequency(self):
        return computeTermFrequency(self.frequencyDistribution)
    
    def getTFIDF(self):
        return computeTFIDF(self.frequencyDistribution)

if __name__ == '__main__' :
#user input
#    try:
#        options, args = getopt.getopt(sys.argv[1:],"",['kmeans=','kdocs=','kdistance='])
#        kmeans = 0 
#        kdocs   = 0
#        kdistance = 0 
#        print options
#        for o in options:
#            if o[0]=='--kmeans':
#                kmeans = o[1]
#            if o[0]=='--kdocs':
#                kdocs  = o[1]
#            if o[0]=='--kdistance':
#                kdistance  = o[1]
#        #print indata, inlearn
#    except getopt.GetoptError:
#        print 'Argument error, Usage: kclustering.py --kmeans=d1 --kdocs=d2 --kdistance=d3'
#        print 'd1 : [1,2,3,...,20]'
#        print 'd2 : [1,2,...,1000]'
#        print 'd3 : [1=cosineTF,2=cosineTFIDF]'
#        sys.exit(0)

### main parameter
    kmeans = 5
    kdocs  = 20
    kdistance = 2 ##cosineTFIDF


### initialize variable    
    maxiteration = 50
    k = kmeans
    docsPerGroup = kdocs
    distanceType = 'cosineTF'   #default
    if kdistance == 1: distanceType = 'cosineTF'  #cosineTF/cosineTFIDF
    
#read files
    newsgroups = NewsGroups("20_newsgroups",docsPerGroup, k)
    print "--> Read newsgroups"
    newsgroups.loadDocuments()
    clusteringLib.totalFreqDist = totalFreqDist
    totalDocuments = len(newsgroups.documents)
    
    centers = {}
    i = 0
    print "--> Initialize centers"
# initialize center
    while i < k :
        center = KCenter(newsgroups.documents, clusteringLib.initializeCenter(newsgroups.documents,k))
        centers[i] = center
        i+=1

#clustering iteration
    nbIteration = 1
    while True:
        prevCenters = dict([(d,newsgroups.documents[d].centerID) for d in newsgroups.documents]) 
        for d in newsgroups.documents :
            index = clusteringLib.findCenter(centers, newsgroups.documents[d], distanceType)
            newsgroups.documents[d].centerID = index
            if (prevCenters[d]!=None) : 
                centers[prevCenters[d]].cluster.remove(d)
            centers[index].cluster.append(d)
        # set center
        for c in centers :
            centers[c].calcMeanCluster()           
        if nbIteration >= maxiteration or (not clusteringLib.hasChanges(prevCenters, newsgroups)):
            break; 
        nbIteration +=1 
    
    i = 0
    print 'iteration required:',nbIteration
    print 'distance calc type:',distanceType
    while i < k :
        print '>>> cluster:',i
        docgroup = {}
        for d in newsgroups.documents:
            if newsgroups.documents[d].centerID==i :
                if newsgroups.documents[d].actualGroup not in docgroup :
                    docgroup[newsgroups.documents[d].actualGroup] = 1
                else :
                    docgroup[newsgroups.documents[d].actualGroup] += 1
        for g in docgroup:
            print g,':',docgroup[g]
        i+=1    