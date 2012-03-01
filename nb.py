import sys
import nltk
import string
import email
import re
import os
import math
import nbutils
from random import choice
import getopt 

### classifier  class
class NBClassifier:
    def __init__(self, file, NBTrainer):
        emailReader = EmailReader(file)
        self.__emailObj = emailReader.parse()
        self.__decisionMatrix = {}
        self.__NBTrainer = NBTrainer
        
### update classifier decision matrix by input frequency
    def __updateMatrix(self, freq):
        statistic = self.__NBTrainer.getStatistic()
        for f in freq:
            if f in statistic:
                if (f in self.__decisionMatrix):
                    self.__decisionMatrix[f]=(self.__decisionMatrix[f][0]+freq[f], #wordfreq in email -> store it...we might need
                                          statistic[f][2],  #wordfreq in NBTrainer-dataset
                                          statistic[f][3]   #wordfreq in NBTrainer-dataset
                                          )
                else:
                    self.__decisionMatrix[f]=(freq[f],      #wordfreq in email -> store it...we might need
                                          statistic[f][2],  #wordfreq in NBTrainer-dataset
                                          statistic[f][3]   #wordfreq in NBTrainer-dataset
                                          )
### classify the input email
    def classify(self):
### process the body and subject
        body = ""
        if self.__emailObj["bodyText"] is not None:
            body += self.__emailObj["bodyText"] + " "
        if self.__emailObj["bodyHtml"] is not None:
            body += htmlStripper(self.__emailObj["bodyHtml"], True)
        if body is not None:
            freq = self.__NBTrainer.computeFreqDist(body, 'c')     #body, type = classifier
            self.__updateMatrix(freq)
        if self.__emailObj["subject"] is not  None:
            freq = self.__NBTrainer.computeFreqDist(self.__emailObj["subject"], 'c') #subject, type = classifier
            self.__updateMatrix(freq)
### get statistic        
        statSpam = self.__NBTrainer.getEmailStat()[0] * 1.0
        statHam  = self.__NBTrainer.getEmailStat()[1] * 1.0
### calculate prior probabilities
        fSpam = math.log10(statSpam / (statSpam+statHam))
        fHam  = math.log10(statHam  / (statSpam+statHam))
### calculate LOG-LIKELIHOOD for an input email        
        for d in self.__decisionMatrix:
            fSpam += self.__decisionMatrix[d][1]
            fHam  += self.__decisionMatrix[d][2] 
        return (fSpam, fHam)

### trainer class
### ignore the english stopword (nltk.corpus.stopwords.word)
class NBTrainer:
    def __init__(self):
        self.__dataSet = {}         ### dataSet {word:(spam, ham)}
        self.__nbEmail = (0,0)      ### nbEmail (spam, ham)
        self.__stopWords = stopwords = nltk.corpus.stopwords.words('english')
               
    def __validateWord(self, word):
### ignore string digits
        if word.isdigit():
            return False
### ignore stopwords
        if word in self.__stopWords:
            return False
### ignore string punctuation (Ending Spam-Zdiziaski, Jonathan A)
        if (word in string.punctuation):
            return False
        return True
        
### update statistic of nb of emails (SPAM, HAM)
### tokenize the message
    def computeFreqDist(self, text, type):
        if type=='s':
            self.__nbEmail=(self.__nbEmail[0]+1, self.__nbEmail[1])
        elif type=='h':
            self.__nbEmail=(self.__nbEmail[0], self.__nbEmail[1]+1)
        freq = nltk.FreqDist()
        text = text.lower()
        for word in nltk.word_tokenize(text):
            if (self.__validateWord(word)):
                freq.inc(word)
        return dict(freq)
    
### update trainer matrix 
    def __updateMatrix(self, freq, type):
        for f in freq:
            if (f in self.__dataSet):
                if (type=='s'):
                    self.__dataSet[f]=(self.__dataSet[f][0]+1, self.__dataSet[f][1])
                elif (type=='h'):
                    self.__dataSet[f]=(self.__dataSet[f][0], self.__dataSet[f][1]+1)
            else:
                if (type=='s'):
                    self.__dataSet[f] = (1, 0)
                elif (type=='h'):
                    self.__dataSet[f] = (0, 1)
                    
### train using an input file   
    def train(self, filename, type):
        emailReader = EmailReader(filename)
### process body email 
        body = ""                
        emailObj = emailReader.parse()
        if emailObj["bodyText"] is not None:
            body += emailObj["bodyText"] + " "
        if emailObj["bodyHtml"] is not None:
            body += htmlStripper(emailObj["bodyHtml"], True)
        if body is not None:
            freq = self.computeFreqDist(body,type)
            self.__updateMatrix(freq, type)   #get freq & update email stat
                
### process subject
        if (emailObj["subject"] is not None):
            freq = self.computeFreqDist(emailObj["subject"], type)  #get freq only
            self.__updateMatrix(freq, type)
            
### calculate log likelihoods and rebuild the dataset as 
###        (N-SPAM, N-HAM, Log(L-SPAM), Log(L-HAM))     
    def buildStatistic(self):
        stat={}      
        sumSpam = 0.0
        sumHam  = 0.0   
        for d in self.__dataSet:
            sumSpam += self.__dataSet[d][0]*1.0
            sumHam  += self.__dataSet[d][1]*1.0

        sumWords=(sumSpam,sumHam)
        for d in self.__dataSet:            
            nSpam = self.__dataSet[d][0]
            nHam  = self.__dataSet[d][1]
            ## set priors 
            if (nSpam==0):
                lSpam = math.log10(1.0   /  (sumWords[1]*1.0))
                lHam  = math.log10(nHam  /  (sumWords[1]*1.0))
            elif (nHam==0):
                lSpam = math.log10(nSpam /  (sumWords[0]*1.0))
                lHam  = math.log10(1.0   /  (sumWords[0]*1.0))
            else:
                lSpam = math.log10(nSpam /  (sumWords[0]*1.0))
                lHam  = math.log10(nHam  /  (sumWords[1]*1.0))
            stat[d]= (self.__dataSet[d][0], self.__dataSet[d][1], lSpam, lHam)
        self.__dataSet = stat

### return statistic data set    
    def getStatistic(self):
        return self.__dataSet

### return email statistic
    def getEmailStat(self):
        return self.__nbEmail

### decode the email (Ending Spam - Zdziarki, Jonathan A)
### we only interest for subject and body of an email (Ending Spam - Zdziarki, Jonathan A)
class EmailReader:
    def __init__(self, file):
        fp = open(file)      
        self.__msg = email.message_from_file(fp)
        self.__email = {}
        fp.close()
    
    def getMessage(self):
        return self.__msg
    
    def __toUnicode(self,str,encoding, replaceBeforeEncode=False):
        try:        
            if encoding:
                encoding = encoding.lower()
                ### assume default_charset is utf-8
                if encoding!="utf-8":
                    if (replaceBeforeEncode):
                        str = unicode(str, encoding,"replace").encode("utf-8","replace")
                    else:
                        str = unicode(str, encoding).encode("utf-8","replace")
        except:
            pass
        return str

### parse email        
    def parse(self):
        self.__email["to"]  = self.__msg["To"]
        self.__email["from"] = self.__msg["From"]
        if self.__msg["Subject"] is not None:
            multiSubject= email.Header.decode_header(self.__msg["Subject"])
            arraySubject = []            
            for str, encoding in multiSubject:               
                arraySubject.append( self.__toUnicode(str,encoding) )
            self.__email["subject"] = ''.join(arraySubject)
        else:
            self.__email["subject"] =  None
        
        bodyText = None
        bodyHtml = None 
        for m in self.__msg.walk():            
            contentType = m.get_content_type().lower()
            if contentType =="text/plain":
                if bodyText is None: bodyText = ""
                bodyText += self.__toUnicode(m.get_payload(decode=True), m.get_content_charset(), True) 
            elif contentType =="text/html":
                if bodyHtml is None: bodyHtml = ""
                bodyHtml += self.__toUnicode(m.get_payload(decode=True), m.get_content_charset(), True)
        self.__email["bodyText"] = bodyText
        self.__email["bodyHtml"] = bodyHtml
        return self.__email

### strip html tags from input a html text
### strip an ascii chars from input a html text
def htmlStripper(html, cleanAll=False):
    try:
        text = nltk.clean_html(html) ### use nltk first, if error do manual stripper
    except:
        pattern = re.compile("<(script|style).*?>.*?</(script|style)>", re.DOTALL)
        text = re.sub(pattern, "", html)
        text = re.sub("<(.|\n)*?>", "", html)
    #drop non alpanumeric characters
    if cleanAll: text=re.sub("&[^;]*; ?", "", text)
    #drop ascii chars
    regexp = re.compile('[^\x09\x0A\x0D\x20-\x7F]')
    text = regexp.sub('', text)
    return text

### get the list emails from input directory
def getEmails(dir):
    emails = {}
    if (os.access(dir, os.F_OK or os.R_OK)):
        type = ""
        id = 0
        for root, dirs, files in os.walk(dir):
            for name in files:
                filename = os.path.join(root, name)
                if root.find("spam")>-1:
                    type = "s"
                elif root.find("ham")>-1:
                    type = "h"
                emails[id]=(filename, type)
                id+=1
    else:
        print 'can not access the input directory'
    return emails

### run the trainer using train emails
def runTrainer(emails):
    nbtrainer = NBTrainer()
    for id in emails:
        nbtrainer.train(emails[id][0], emails[id][1])
    nbtrainer.buildStatistic()
    return nbtrainer

### classify the test emails
def runClassifier(emails, nbtrainer):
    result = {}
    testClass = ""
    classNames = []
    for id in emails:
        nbClassifier = NBClassifier(emails[id][0], nbtrainer)
        testValue = nbClassifier.classify()
        if testValue[0] > testValue[1]:
            testClass = 's' # classified as spam
        elif testValue[0] < testValue[1]:
            testClass = 'h' # classified as ham
        else:
            testClass = 'u' # unclassified
        if testClass not in classNames: classNames.append(testClass)
        result[id] = (emails[id][0], emails[id][1], testClass)
    return result, classNames

### calculate the precision, accuracy, f-measure
def calculatePerformance(result, className):
    nOriginal  = 0
    nClassified = 0
    nMisclassified = 0
    for r in result:
        if result[r][1]==className:
            nOriginal +=1
        if result[r][2]==className:
            if result[r][1]==result[r][2]:
                nClassified +=1
            else:
                nMisclassified +=1
    precision = (nClassified*1.0) /((nClassified+nMisclassified)*1.0)
    accuracy  = (nClassified*1.0) / (nOriginal*1.0)
    fmeasure  = (2 * accuracy * precision) / (accuracy+precision)
    return (fmeasure, precision, accuracy, nClassified, nMisclassified) 
        
if __name__ == '__main__' :   
    try:
        options, args = getopt.getopt(sys.argv[1:],"",['traindir=','testdir='])
        traindir = ""
        testdir  = ""
        for o in options:
            if o[0]=='--traindir':
                traindir = o[1]
            if o[0]=='--testdir':
                testdir  = o[1]
        trainEmails = getEmails(traindir)
        print 'Training set:'
        print 'Spam size:', len([e for e in trainEmails if trainEmails[e][1]=='s']), 'emails'
        print 'Ham size:',  len([e for e in trainEmails if trainEmails[e][1]=='h']), 'emails'
        nbtrainer   = runTrainer(trainEmails)
        testEmails  = getEmails(testdir)
        print 'Test set:'
        print 'Spam size:', len([e for e in testEmails if testEmails[e][1]=='s']), 'emails'
        print 'Ham size:',  len([e for e in testEmails if testEmails[e][1]=='h']), 'emails'
        result, classNames = runClassifier(testEmails, nbtrainer)
        for c in classNames:
            performance = calculatePerformance(result, c)
            if c == 's':
                print 'Spam: f-measure', performance[0] #,",Precision:", performance[1], ",Accuracy:",performance[2]
                print  'TrueSpam:',performance[3],', FalseSpam:',performance[4]
            if c == 'h':
                print 'Ham: f-measure:', performance[0] #,",precision:", performance[1], ",accuracy:",performance[2]
                print  'TrueHam:',performance[3],', FalseHam:',performance[4]
    except getopt.GetoptError:
        print 'Argument error, Usage: nb.py --traindir=d1 --testdir=d2'
        sys.exit(0)