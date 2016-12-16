#!/usr/bin/python
# -*- coding: utf-8 -*-

"""ags-ds2.py: A German Spy's Devil Summoner 2 ellipses challenge."""

__author__      = "TetrisFinalBoss"
__version__     = "0.4.1"

import sys
import cv2
import numpy
import pafy
import re
import os.path
import getopt

AGS_DS2_PLAYLIST = 'PL_ftpUY_ldBTtHOUQLt5irghX1XfIzoy-'

def getFile(media):
    for stream in media.streams:
        if stream.dimensions[1] == 360 and stream.extension=='mp4':
            m = re.search('\[Part\s?([\d]+)\]',media.title)
            fname = "%s - %s.%s"%(m.group(1),media.videoid,stream.extension)
            if not os.path.isfile(fname):
                print 'Downloading video %s from playlist'%(m.group(1))
                stream.download(fname)
            else:
                print "File '%s' already exists, skipping download"%(fname)
            return fname
            
def tsum(t1,t2):
    return tuple(map(lambda x,y: x + y,t1,t2))
    
class DialogLocator:
    D_MATCHMINIMUM = 0.5
    D_WINDOW = {'left':396, 'top':61, 'right':418, 'bottom':84}
    
    def __init__(self):
        self.__dialogPattern = cv2.imread(os.path.dirname(sys.argv[0]) + "/dialog_pattern.png",cv2.IMREAD_GRAYSCALE)
    
    def locate(self, img):
        res = cv2.matchTemplate(img[self.D_WINDOW['top']:self.D_WINDOW['bottom'],self.D_WINDOW['left']:self.D_WINDOW['right']],
                                self.__dialogPattern,
                                cv2.TM_SQDIFF_NORMED)
        min_val = cv2.minMaxLoc(res)[0]
        if min_val>self.D_MATCHMINIMUM:
            return False
        
        return True

class SomethingExplainedDetector:
    MATCHMINIMUM = 0.4
    
    QUOT_WINDOW = {'left':20, 'top':15, 'right':50, 'bottom':42}
    EXPL_WINDOW = {'left':20, 'top':15, 'right':400, 'bottom':42}
    
    def __init__(self):
        self.__count = 0
        self.__ncount = 0
        
        self.__quotPattern = cv2.imread(os.path.dirname(sys.argv[0]) + "/quot_pattern.png",cv2.IMREAD_GRAYSCALE)
        self.__explPattern = cv2.imread(os.path.dirname(sys.argv[0]) + "/expl_pattern.png",cv2.IMREAD_GRAYSCALE)
        
        self.__quotDim = self.__quotPattern.shape
        self.__explDim = self.__explPattern.shape
    
    def detect(self, img):
        ret = []
        
        # Search for patterns, first quotation mark
        res = cv2.matchTemplate(img[self.QUOT_WINDOW['top']:self.QUOT_WINDOW['bottom'],
                                    self.QUOT_WINDOW['left']:self.QUOT_WINDOW['right']],
                                self.__quotPattern,
                                cv2.TM_SQDIFF_NORMED)
        minmax = cv2.minMaxLoc(res)
        if minmax[0] < self.MATCHMINIMUM:
            top_left = tsum(minmax[2], (self.QUOT_WINDOW['left'],self.QUOT_WINDOW['top']))
            bottom_right = tsum(top_left, (self.__quotDim[1],self.__quotDim[0]))
            
            ret.append((top_left,bottom_right))
        else:
            # No new objects, but __count stays the same until dialog is over
            self.__ncount = 0
            return ret
        
        # Second 'explained' word
        res = cv2.matchTemplate(img[self.EXPL_WINDOW['top']:self.EXPL_WINDOW['bottom'],
                                    self.EXPL_WINDOW['left']:self.EXPL_WINDOW['right']],
                                self.__explPattern,
                                cv2.TM_SQDIFF_NORMED)
        minmax = cv2.minMaxLoc(res)
        if minmax[0] < self.MATCHMINIMUM:
            top_left = tsum(minmax[2], (self.EXPL_WINDOW['left'],self.EXPL_WINDOW['top']))
            bottom_right = tsum(top_left, (self.__explDim[1],self.__explDim[0]))
            
            ret.append((top_left,bottom_right))
        else:
            # No new objects, but __count stays the same until dialog is over
            self.__ncount = 0
            return ret
        
        # Both are found, mess with counters
        self.__ncount = self.__count==0 and 1 or 0
        self.__count = 1
        
        return ret
    
    def reset(self):
        self.__count = 0
        self.__ncount = 0
    
    def dialogClosed(self):
        # If no dialog arrow is found reset all values
        self.__ncount = 0
        self.__count = 0
    
    def name(self):
        return "'someone explained something'"
    
    def uniqueObjects(self):
        return False
        
    def newObjectsCount(self):
        return self.__ncount
    
class CircumstancesExplainedDetector:
    MATCHMINIMUM = 0.4
    
    SEARCH_WINDOW = {'left':10, 'top':10, 'right':400, 'bottom':84}
    
    def __init__(self):
        self.__count = 0
        self.__ncount = 0
        self.__pobj = []
        
        self.__ec1Pattern = cv2.imread(os.path.dirname(sys.argv[0]) + "/etc_pattern.png",cv2.IMREAD_GRAYSCALE)
        self.__ec1Dim = self.__ec1Pattern.shape
    
    def detect(self, img):
        ret = self.__pobj
        
        # Search for pattern
        res = cv2.matchTemplate(img[self.SEARCH_WINDOW['top']:self.SEARCH_WINDOW['bottom'],
                                    self.SEARCH_WINDOW['left']:self.SEARCH_WINDOW['right']],
                                self.__ec1Pattern,
                                cv2.TM_SQDIFF_NORMED)
        minmax = cv2.minMaxLoc(res)
        if minmax[0] < self.MATCHMINIMUM:
            top_left = tsum(minmax[2], (self.SEARCH_WINDOW['left'],self.SEARCH_WINDOW['top']))
            bottom_right = tsum(top_left, (self.__ec1Dim[1],self.__ec1Dim[0]))
            
            ret = [(top_left,bottom_right)]
            
            self.__pobj = ret
            
            self.__ncount = self.__count==0 and 1 or 0
            self.__count = 1
        else:
            # Nothing is found, but if we've already found something in this dialog box
            # let's assume that this object is still present, because this detector is blocking one
            self.__count = len(self.__pobj)
            self.__ncount = 0
        
        return ret
    
    def reset(self):
        self.__count = 0
        self.__ncount = 0
        self.__pobj = []
    
    def dialogClosed(self):
        # If no dialog arrow is found reset all values
        self.__ncount = 0
        self.__count = 0
        self.__pobj = []
    
    def name(self):
        return "'explained the circumstances'"
    
    def uniqueObjects(self):
        return True
        
    def newObjectsCount(self):
        return self.__ncount
    
    
class MeaningfulSilenceDetector:
    MATCHMINIMUM = 0.4
    
    PATTERN_SIZE = (16,60,1)
    PATTERN_OFFSET = {'x':18,'y':10}

    SEARCH_WINDOW = {'left':10, 'top':10, 'right':110, 'bottom':40}
    
    PATTERN_COLOR = 127
    
    def __init__(self):
        self.__pattern = numpy.zeros(self.PATTERN_SIZE, numpy.uint8)
        for i in xrange(6):
            cv2.rectangle(self.__pattern,
                          (self.PATTERN_OFFSET['x']+7*i,self.PATTERN_OFFSET['y']),
                          (self.PATTERN_OFFSET['x']+1+7*i,self.PATTERN_OFFSET['y']+1),
                          self.PATTERN_COLOR,
                          -1)
        self.__count = 0
        self.__ncount = 0        
        self.__pobj = []
    
    def detect(self, img):
        # Set "default" return value to previously found object in this dialog entry
        # which is reset to [] when dialog is closed
        ret = self.__pobj
        
        # Search for pattern
        res = cv2.matchTemplate(img[self.SEARCH_WINDOW['top']:self.SEARCH_WINDOW['bottom'],
                                    self.SEARCH_WINDOW['left']:self.SEARCH_WINDOW['right']],
                                self.__pattern,
                                cv2.TM_SQDIFF_NORMED)
        
        # There can be only one "......" in dialog, so we totally fine with global minimum
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if min_val < self.MATCHMINIMUM:
            top_left = tsum(min_loc, (self.SEARCH_WINDOW['left'],self.SEARCH_WINDOW['top']))
            bottom_right = tsum(top_left, (self.PATTERN_SIZE[1],self.PATTERN_SIZE[0]))
            
            # Something is found, set return value and store this object for future use
            ret = [(top_left,bottom_right)]
            self.__pobj = ret
            
            self.__ncount = self.__count==0 and 1 or 0
            self.__count = 1
        else:
            # Nothing is found, but if we've already found something in this dialog box
            # let's assume that this object is still present
            self.__count = len(self.__pobj)
            self.__ncount = 0
        
        return ret
        
    def dialogClosed(self):
        self.__count = 0
        self.__ncount = 0
        self.__pobj = []
    
    def reset(self):
        self.__count = 0
        self.__ncount = 0
        self.__pobj = []
    
    def name(self):
        return "'......'"
    
    def uniqueObjects(self):
        return True
        
    def newObjectsCount(self):
        return self.__ncount
    
class MidSentenceEllipsesDetector:
    MATCHMINIMUM = 0.5
    
    PATTERN_SIZE = (8,20,1)
    PATTERN_OFFSET = {'x':1,'y':3}
    
    PATTERN_COLOR = 127
    
    def __init__(self):
        self.__pattern = numpy.zeros(self.PATTERN_SIZE, numpy.uint8)
        for i in xrange(3):
            cv2.rectangle(self.__pattern,
                          (self.PATTERN_OFFSET['x']+7*i,self.PATTERN_OFFSET['y']),
                          (self.PATTERN_OFFSET['x']+1+7*i,self.PATTERN_OFFSET['y']+1),
                          self.PATTERN_COLOR,
                          -1)
        self.__ncount = 0
        self.__count = 0
    
    def detect(self, img):
        ret = []
        
        res = cv2.matchTemplate(img,self.__pattern,cv2.TM_SQDIFF_NORMED)
        
        # For each row in dialog do recursive search for global minimums
        def localMinInRow(row,offset):
            # Current dimensions
            h,w = row.shape
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(row)
            if min_val < self.MATCHMINIMUM:
                x,y = min_loc
                # Recalculate absolute position and append value
                min_loc = tsum(min_loc,offset)
                ret.append((min_loc,tsum(min_loc, (self.PATTERN_SIZE[1],self.PATTERN_SIZE[0]))))
                
                # Add threshold around this point
                mthresh = self.PATTERN_SIZE[1]
                
                # Now search minimums in left region
                if x-mthresh>self.PATTERN_SIZE[1]:
                    localMinInRow(row[0:h,0:x-mthresh],offset)
                
                # And in right region
                if w-x-mthresh > self.PATTERN_SIZE[1]:
                    localMinInRow(row[0:h,x+mthresh:w],tsum(offset,(x+mthresh,0)))
            
        for i in xrange(3):
            yoff = 20+i*20+4
            localMinInRow(res[yoff:yoff+18,20:400],(20,yoff))
        
        # Sometimes objects may be lost and caught again later
        # Let's try to address this issue
        l = len(ret)
        
        # Get new objects count
        self.__ncount = l - self.__count
        if self.__ncount<0:
            self.__ncount = 0
        
        # Store object count, but assuming, that objects can't disappear
        # during same dialog line, so it alway stays at maximum level
        self.__count = max(l,self.__count)
        
        return ret
    
    def reset(self):
        self.__ncount = 0
        self.__count = 0
    
    def dialogClosed(self):
        self.__ncount = 0
        self.__count = 0
        
    def name(self):
        return "'...'"
    
    def uniqueObjects(self):
        return False
        
    def newObjectsCount(self):
        return self.__ncount
            
class EllipsesSearcher:
    # Threshold values
    THRESHOLD_VALUE = 90
    THRESHOLD_COLOR = 127
    
    # Dialog box window
    DIALOG = {'left':104,'top':248,'right':538,'bottom':340}
    
    def __init__(self):
        # Init detectors
        self.__detectors = []
        
        self.__detectors.append(MeaningfulSilenceDetector())
        self.__detectors.append(MidSentenceEllipsesDetector())
        self.__detectors.append(CircumstancesExplainedDetector())
        self.__detectors.append(SomethingExplainedDetector())
        
        # Init dialog locator
        self.__dialogLocator = DialogLocator()
        
        # Reset other values
        self.__total = len(self.__detectors)*[0]
        
        self.snapshots = False
        self.statFile = None
        self.useStatFile = False
        self.ignoreStat = False
        self.preview = False
        self.detectorMask = 0xff
        
    def __initCounter(self, counter):
        counter.r
            
    def __thresh(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, t = cv2.threshold(gray, self.THRESHOLD_VALUE, self.THRESHOLD_COLOR, cv2.THRESH_TOZERO)
        return t

    def count(self,fname):
        if self.useStatFile:
            self.statFile.write("===\n%s\n===\n"%(fname))
            self.statFile.flush()
            
        count = len(self.__detectors)*[0]
        
        # First - try to get statistics from file,
        # so we don't have to recalculate stats once again
        if not self.ignoreStat:
            try:
                statfile = open('statistics/'+fname+'.stat','r')
                
                for ln in statfile.readlines():
                    m = re.search('([\d]+):([\d]+)\s([\d]+)',ln)
                    if m:
                        # Last parameter is object type - i.e. detector number
                        det = int(m.group(3))
                        # Write stats to user specified file, if enabled
                        if self.useStatFile:
                            self.statFile.write("%s is found at %s:%s (from stat file)\n"%(self.__detectors[det].name(),m.group(1),m.group(2)))
                            self.statFile.flush()
                        # And increase counter
                        count[det]+=1
                
                statfile.close()
                
                # Increase total value
                self.__total = map(lambda x,y: x+y, count, self.__total)
                
                # Display progress
                print "Reading statistics from file: Done  -  %d objects detected"%(sum(count))
                
                # And also write to user specified file
                if self.useStatFile:
                    self.statFile.write("===\n")
                    for e in zip(map(lambda x: x.name(), self.__detectors), count):
                        self.statFile.write("%s is said %d times\n"%e)
                    self.statFile.write("\n")
                    self.statFile.flush()
                
                # And that's it, this file is done
                return
                
            except (OSError, IOError):
                # Reset counter value
                count = len(self.__detectors)*[0]

        # Reset detectors before apply them to new file                
        for d in self.__detectors:
            d.reset()
        
        statfile = open('statistics/'+fname+'.stat','w')
        
        v = cv2.VideoCapture(fname)
        
        frame_count = int(v.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        frame_no = 0
        
        previewRate = 1
        
        while v.isOpened():
            ret, frame = v.read()
            frame_no+=1

            if not ret:
                break
            
            # Use simple threshold for dialog box
            box = frame[self.DIALOG['top']:self.DIALOG['bottom'],self.DIALOG['left']:self.DIALOG['right']]
            t = self.__thresh(box)
            
            objects = []
            shouldSaveSnapshot = False
            secs = int(v.get(cv2.cv.CV_CAP_PROP_POS_MSEC)/1000)
            
            dialogClosed = not self.__dialogLocator.locate(t)
            
            # Now apply all detectors for this frame
            for i in xrange(len(self.__detectors)):
                # Check if detector is enabled
                if (self.detectorMask & (1 << i)) == 0:
                    continue
                
                if dialogClosed:
                    self.__detectors[i].dialogClosed()
                    continue
                
                # Apply detector to thresholded picture and store all found objects for this particular detector
                items = self.__detectors[i].detect(t)
                
                # If some of these objects are new
                ncount = self.__detectors[i].newObjectsCount()
                
                if ncount>0:
                    count[i] += ncount
                    self.__total[i] += ncount
                    
                    shouldSaveSnapshot = self.snapshots
                    
                    for j in xrange(ncount):
                        # Write to user specified file
                        if self.useStatFile:
                            self.statFile.write("%s is found at %d:%d\n"%(self.__detectors[i].name(),secs/60,secs%60))
                            self.statFile.flush()
                        
                        # And store stats for future use
                        statfile.write('%d:%d %d\n'%(secs/60,secs%60,i))

                if len(items):
                    objects += items
                    # If we found unique object (i.e. there can't be any other objects in this picture) - stop applying detectors
                    if self.__detectors[i].uniqueObjects():
                        break
            
            # Prepare images
            if shouldSaveSnapshot or self.preview:
                for item in objects:
                    if shouldSaveSnapshot:
                        cv2.rectangle(box,item[0],item[1],(0xff,0,0))
                    if self.preview:
                        cv2.rectangle(t,item[0],item[1],0xff)
            
            # Save snapshot
            if shouldSaveSnapshot:
                cv2.imwrite("snapshots/%s.%d-%d.png"%(fname,secs/60,secs%60),box)

            # Show preview window if enabled             
            if self.preview:
                cv2.imshow("Picture",t)
                k = cv2.waitKey(previewRate) & 0xff
                if k==ord('q'):
                    sys.exit(0)
                elif k==ord('s'):
                    cv2.imwrite('snapshots/snapshot_orig.png',box)
                    cv2.imwrite('snapshots/snapshot_modified.png',t)
                elif k==ord('n'):
                    previewRate = 0
                elif k==ord('p'):
                    previewRate = 1
            
            # Display some progress
            progress = frame_no*100/frame_count
            sys.stdout.write("Processing video: %d%%  -  %d objects found\r"%(progress, sum(count)))
            sys.stdout.flush()
        
        # Display final state for this file
        print "Processing video: Done - %d objects found"%(sum(count))
        
        # And also write to user specified file
        if self.useStatFile:
            self.statFile.write("===\n")
            for e in zip(map(lambda x: x.name(), self.__detectors), count):
                self.statFile.write("%s is said %d times\n"%e)
            self.statFile.write("\n")
            self.statFile.flush()
        
        v.release()
        statfile.close()
    
    def total(self):
        ret = ""
        for e in zip(map(lambda x: x.name(), self.__detectors), self.__total):
            ret = ret+"%s is said %d times\n"%e
        return ret

if __name__=="__main__":
    el = EllipsesSearcher()
    
    downloadOnly = False
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hirdvf:m:")
    except getopt.GetoptError as err:
        print str(err)
        sys.exit(1)
    
    for opt, arg in opts:
        if opt == '-h':
            print 'Usage: %s [-h] [-i] [-r] [-d] [-v] [-f filename] [-m mask]'%(sys.argv[0])
            print '-h         -- Show this help'
            print '-i         -- Save snapshots each time ellipses is found'
            print '-r         -- Ignore (reset) previously collected statistics'
            print '-d         -- Download only'
            print '-v         -- Display video preview (debug mode)'
            print '-m <mask>  -- Set detector mask to <mask>'
            print '-f <file>  -- Write statistics to <file>'
            sys.exit()
        elif opt == "-i":
            print 'Snapshots is enabled'
            if not os.path.isdir('snapshots'):
                os.mkdir('snapshots')
            el.snapshots = True
        elif opt == "-r":
            el.ignoreStat = True
        elif opt == "-v":
            el.preview = True
        elif opt == "-d":
            downloadOnly = True
        elif opt == "-f":
            el.useStatFile = True
            el.statFile = open(arg,'w')
        elif opt == "-m":
            el.detectorMask = int(arg)
    
    if not os.path.isdir('statistics'):
        os.mkdir('statistics')
            
    # get Devil Summoner 2 playlist
    playList = pafy.get_playlist(AGS_DS2_PLAYLIST)

    for media in playList['items']:
        fname = getFile(media['pafy'])
        if not downloadOnly:
            el.count(fname)

    print "We are done!"
    if downloadOnly:
        print "Playlist downloaded!"
    else:
        print el.total()
        if el.useStatFile:
            el.statFile.write("===\nTotal\n===\n%s"%(el.total()))
            el.statFile.flush()
