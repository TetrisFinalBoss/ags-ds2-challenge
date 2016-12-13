#!/usr/bin/python
# -*- coding: utf-8 -*-

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
            
class EllipsesSearcher:
    # Threshold values
    THRESHOLD_VALUE = 90
    THRESHOLD_COLOR = 127
    
    # Minimum value for square root diff template matching
    ELLIPSES_MATCHMINIMUM = 0.4
    
    # "......" pattern generator values
    PATTERN_SIZE = (16,60,1)
    PATTERN_XOFF = 18
    PATTERN_YOFF = 10
    
    # Search window for "......" (L,T,R,B)
    PATTERN_SEARCH = (10,10,110,40)
    
    # Dialog box window (L,T,R,B)
    DIALOG = (104,248,538,340)
    
    # Offsets to crop "..." part form "......" pattern (L,T,R,B)
    PATTERN3 = (38,0,57,16)
    PATTERN3_SIZE = (16,20)

    # Minimum value for square root diff template matching
    PATTERN3_MATCHMINIMUM = 0.6
    
    def __init__(self):
        # Init '......' pattern
        self.__pattern = numpy.zeros(self.PATTERN_SIZE, numpy.uint8)
        for i in xrange(6):
            cv2.rectangle(self.__pattern,
                          (self.PATTERN_XOFF+7*i,self.PATTERN_YOFF),
                          (self.PATTERN_XOFF+1+7*i,self.PATTERN_YOFF+1),
                          self.THRESHOLD_COLOR,
                          -1)
        # Init '...' pattern
        self.__pattern3 = self.__pattern[self.PATTERN3[1]:self.PATTERN3[3],self.PATTERN3[0]:self.PATTERN3[2]]
        # Reset other values
        self.__total6 = 0
        self.snapshots = False
        self.statFile = None
        self.useStatFile = False
        self.ignoreStat = False
            
    def __thresh(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, t = cv2.threshold(gray, self.THRESHOLD_VALUE, self.THRESHOLD_COLOR, cv2.THRESH_TOZERO)
        return t

    
    def __search6(self,img):
        # Search for pattern
        res = cv2.matchTemplate(img[self.PATTERN_SEARCH[1]:self.PATTERN_SEARCH[3],
                                    self.PATTERN_SEARCH[0]:self.PATTERN_SEARCH[2]],
                                self.__pattern,
                                cv2.TM_SQDIFF_NORMED)
        # There can be only one "......" in dialog, so we totally fine with global minimum
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        ret = []
        if min_val < self.ELLIPSES_MATCHMINIMUM:
            min_loc = tsum(min_loc, (self.PATTERN_SEARCH[0],self.PATTERN_SEARCH[1]))
            ret.append((min_loc,tsum(min_loc, (self.PATTERN_SIZE[1],self.PATTERN_SIZE[0]))))            
        return ret
        
    def __search3_fast(self,img):
        res = cv2.matchTemplate(img,self.__pattern3,cv2.TM_SQDIFF_NORMED)
        ret = []
        # This code search for global minimum, so it can find pattern only once
        # But there can be several "..." in one sentence
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if min_val < self.PATTERN3_MATCHMINIMUM:
            ret.append((min_loc,tsum(min_loc,(self.PATTERN3_SIZE[1],self.PATTERN3_SIZE[0]))))
        return ret
        
    def __search3_slow(self,img):
        res = cv2.matchTemplate(img,self.__pattern3,cv2.TM_SQDIFF_NORMED)
        ret = []
        # This code should be able to find all instances of "..." on screen
        # But it's sooo slooow
        y=0
        while y<res.shape[0]:
            x=0            
            yoff=1
            while x<res.shape[1]:
                if res.item(y,x) < self.PATTERN3_MATCHMINIMUM:
                    ret.append(((x,y),tsum((x,y),(self.PATTERN3_SIZE[1],self.PATTERN3_SIZE[0]))))
                    x+=self.PATTERN3_SIZE[1]
                    yoff = self.PATTERN3_SIZE[0]
                else:
                    x+=1
            y+=yoff
        
        return ret

    def count(self,fname):
        state = 0
        count6 = 0
        
        if self.useStatFile:
            self.statFile.write("===\n%s\n===\n"%(fname))
        
        # First - try to get statistics from file,
        # so we don't have to recalculate stats once again
        if not self.ignoreStat:
            try:
                statfile = open(fname+'.stat','r')
                
                for ln in statfile.readlines():
                    m = re.search('([\d]+):([\d]+)\s([\d]+)',ln)
                    if m:
                        # Last parameter is object type - ignored, because we only search for '......'
                        # Write stats to user specified file, if enabled
                        if self.useStatFile:
                            self.statFile.write("'......' is found at %s:%s (from stat file)\n"%(m.group(1),m.group(2)))
                        # And increase counter
                        count6 += 1
                
                statfile.close()
                
                # Increase total value
                self.__total6+=count6
                
                # Display progress
                print "Reading statistics from file: Done  -  '......' is said %d times"%(count6)
                
                # And that's it, this file is done
                return
                
            except (OSError, IOError) as err:
                # Reset counter value
                count6 = 0
        
        statfile = open(fname+'.stat','w')
        
        v = cv2.VideoCapture(fname)
        
        frame_count = int(v.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        frame_no = 0
        
        while v.isOpened():
            ret, frame = v.read()
            frame_no+=1

            if not ret:
                break
            
            # The only thing we are looking for right now - is when
            # character says nothing, only ellipses, we don't want to gather
            # more complex statistics right now
            # So threshold will work just fine, but we still work with whole dialog box
            box = frame[self.DIALOG[1]:self.DIALOG[3],self.DIALOG[0]:self.DIALOG[2]]
            t = self.__thresh(box)
            
            # Now look for ellipses in particular zone of image
            objects = self.__search6(t)
            if len(objects):
                # Ok, something is found
                if not state:
                    # This is new object, change current state and increase counter
                    state = 1
                    count6 += 1
                    
                    # Get frame time
                    secs = int(v.get(cv2.cv.CV_CAP_PROP_POS_MSEC)/1000)
                    
                    # Save snapshot
                    if self.snapshots:
                        for item in objects:
                            cv2.rectangle(box,item[0],item[1],(0xff,0,0))
                        cv2.imwrite("%s.%d-%d.png"%(fname,secs/60,secs%60),box)
                    
                    # Increase total
                    self.__total6+=1
                    
                    # Write to user specified file
                    if self.useStatFile:
                        self.statFile.write("'......' is found at %d:%d\n"%(secs/60,secs%60))
                    
                    # And store stats for future use
                    statfile.write('%d:%d %d\n'%(secs/60,secs%60,0))
            else:
                # No '......' object in this frame, change current state
                if state == 1:
                    state = 0
                
                # And now we should hunt for "..."
                # But code for it is very inconsistent and gives unreliable result
                # Maybe someone will fix it later
                # Skip this for now
            
            # Display some progress
            progress = frame_no*100/frame_count
            sys.stdout.write("Processing video: %d%%  -  '......' is said %d times\r"%(progress, count6))
            sys.stdout.flush()
        
        # Display final state for this file
        print "Processing video: Done  -  '......' is said %d times"%(count6)
        
        # And also write to user specified file
        if self.useStatFile:
            self.statFile.write("===\n'......' is said %d times\n\n"%(count6))
        
        v.release()
        statfile.close()
    
    def total(self):
        return self.__total
        
    def stats(self):
        return self.__statistics + "===\n'......' is said %d times in this playthrough"%(self.__total)

if __name__=="__main__":
    # get Devil Summoner 2 playlist
    playList = pafy.get_playlist(AGS_DS2_PLAYLIST)
    
    el = EllipsesSearcher()
    
    saveToFile = False
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hirf:")
    except getopt.GetoptError as err:
        print str(err)
        sys.exit(1)
    
    for opt, arg in opts:
        if opt == '-h':
            print 'Usage: %s [-h] [-i] [-r] [-f filename]'%(sys.argv[0])
            print '-h         -- Show this help'
            print '-i         -- Save snapshots each time ellipses is found'
            print '-r         -- Ignore (reset) previously collected statistics'
            print '-f <file>  -- Write statistics to <file>'
            sys.exit()
        elif opt == "-i":
            print 'Snapshots is enabled'
            el.snapshots = True
        elif opt == "-r":
            el.ignoreStat = True
        elif opt == "-f":
            el.useStatFile = True
            el.statFile = open(arg,'w')

    for media in playList['items']:
        fname = getFile(media['pafy'])
        el.count(fname)

    print "We are done!"
    print "'......' is said %d times in this playthrough"%(el.total())
