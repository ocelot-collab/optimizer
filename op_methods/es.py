#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 12:07:24 2017

@author: Alexander Scheinker
"""

import numpy as np
import time

class ES_min: 
    def __init__(self):
        #super(ES_min, self).__init__() 

        k = 100.0
        alpha = 2*0.1
        
        k = 2.0
        alpha = 0.1*1.0
        w0 = 200.0
        alphaES = 10*0.1
        
        self.k = k
        self.alpha = alpha
        self.alphaES = alphaES
        self.w0 = w0
        self.dtES = 2*np.pi/(10*w0)
        self.max_iter = 300
        
        
    def minimize(self, error_func, x):
        """
        Error_func is a function of the vector x, where x is all of the
        parameters being tuned. error_func(x) actually does a lot under
        the hood, it sets actual machine (or simulation) parameters to
        the values giveb by the vector x, it then outputs the machine
        or simulation output, then computes the cost and then returns
        -1 * power for minimization
        """
        self.error_func = error_func
        
        # length of x
        self.nparams = len(x)
        
        # ES frequencies
        self.wES = self.w0*(0.5*(1+np.arange(self.nparams))/(self.nparams+0.0)+1)
        
        # Set upper and lower bounds
        self.pmax = x + np.abs(x)*1.1
        self.pmin = x - np.abs(x)*0.9
        
        self.pmax = 5+0*np.arange(self.nparams)
        self.pmin = -5+0*np.arange(self.nparams)
        
        # Use first 2 steps to get a rough understanding of sensitivity

        [self.dc, self.dp, self.dcdp, self.kES, pnew] = self.ES_sensitivity(x)
    
        cost_val = error_func(x)
        
        
        # Normalize parameters within [-1 1]
        pnorm = self.ES_normalize(pnew)
        
        naves = 10
        dp_track = 0.0+0.0*np.reshape(np.arange(len(x)*naves),[naves,len(x)])
        dc_track = 0.0+0.0*np.arange(naves)
            
        #check_cost = 0
        # Now start the ES process
        count = 0
        kES_count = 0
        k_stop = 1
        
        k_count = 0
        k_N = 100
        
        for step in np.arange(self.max_iter):
            
            print("step number: ", step)

            p_old = pnorm
            cost_old = cost_val
            

#            if step < 50:
#                " Initially, every 50 steps update kES "
#                if count > 20:
#                    print " Updating kES! "
#                    [self.dc, self.dp, self.dcdp, self.kES, pnew] = self.ES_sensitivity(pnew)
#                    count = 0
#                
#            if step > 50:
#                " After 50 steps, every 50 steps update kES "
#                if count > 200:
#                    print " Updating kES! "
#                    [self.dc, self.dp, self.dcdp, self.kES, pnew] = self.ES_sensitivity(pnew)
#                    count = 0
               
            # Normalize parameters within [-1 1]
            pnorm = self.ES_normalize(pnew)
            
              
            pnorm = pnorm + self.dtES*np.cos(self.wES*step*self.dtES+self.k*self.kES*cost_val)*(self.alpha*self.alphaES*self.wES)**0.5
            
            # Check that parameters stay within normalized range [-1, 1]
            for jn in np.arange(self.nparams):
                if abs(pnorm[jn]) > 1:
                    pnorm[jn]=pnorm[jn]/(0.0+abs(pnorm[jn]))
                
            # Calculate unnormalized parameter value for next cost "
            pnew = self.ES_UNnormalize(pnorm)
            
            
            #if step > 10:
            #    pnew = self.p1ES + 10000.0*self.kES
            #    pnew = self.p1ES + 5
                
            cost_val = error_func(pnew)
            
            time.sleep(0.01)
            
            print("Current cost = ", cost_val)

            print ("Old cost = ", cost_old)

            dc = cost_val - cost_old
            
            print("Cost difference =", dc)

            print("Current parameter values = ", pnorm)

            print("Old parameter values = ", p_old)

            dp = pnorm - p_old
            
            print("Parameter value difference = ", dp)

            print("dc/dp should equal = ", dc/dp)

            print("count = ", count)

            print("currently, dp_track = ", dp_track)

            print("currently, dc_track = ", dc_track)

            dp_track[count] = dp
            dc_track[count] = dc           
            count += 1
            kES_count += k_stop*1
            

            print( "print kES_count =", kES_count)

            if count == naves:
                dp_ave = 0
                for jp in dp_track:
                    dp_ave = dp_ave + jp/naves
                dc_ave = 0
                for jc in dc_track:
                    dc_ave = dc_ave + jc/naves
                dcdp_ave = np.abs(dc_ave/dp_ave)
                count = 0
            
                
            if kES_count == 100*naves:
                k_stop = 0
                
                print("Updating kES!")
                if k_count == 0:
                    old_kES = self.kES
                    new_kES = (2*(self.w0/(self.alphaES))**0.5)/(0.0+dcdp_ave)
                    
                print("Old kES =", old_kES)

                
                print("New kES =", new_kES)

                    
                k_count += 1
                
                self.kES = old_kES*(k_N-k_count)/k_N + new_kES*k_count/k_N
                
                print("k_count =", k_count)

                print("Current kES = ", self.kES)

                self.dc = dc_track
                self.dp = dp_track
                self.dcdp = dcdp_ave
                
                if k_count == k_N:
                    k_count = 0
                    kES_count = 0
                    k_stop = 1
                
            
        return cost_val
        
    
        
    def ES_normalize(self,p):
        """
        Normalize parameter values to within [-1 1]

        :param p:
        :return:
        """
        pdiff = (self.pmax - self.pmin)/2
        pmean = (self.pmax + self.pmin)/2
        pnorm = (p - pmean)/pdiff
        return pnorm
    
    def ES_UNnormalize(self,p):
        """
        Un normalize parameters back to physical values
        """
        pdiff = (self.pmax - self.pmin)/2
        pmean = (self.pmax + self.pmin)/2
        pUNnorm = p*pdiff + pmean
        return pUNnorm
    
    def ES_sensitivity(self, p):
        """
        Calculate total change in cost relative to change in parameters

        :param p:
        :return:
        """
        
        # Save initial parameter values "\
        #self.p1ES = self.ES_normalize(x)
        nave = 5.0
        dcdp = 0.0
        
        for jave in np.arange(nave):
            p1 = p
            c1 = self.error_func(p1)
            p1N = self.ES_normalize(p1)
            p2N = p1N + (self.pmax-self.pmin)/10000.0
            p2 = self.ES_UNnormalize(p2N)
            c2 = self.error_func(p2)
            dcdp = dcdp + abs((c2-c1)/(p2N-p1N))/nave
            
        dc = c2 - c1
        dp = p2 - p1
            #if dcdp > 0:
            #    kES = (2*(self.w0/(self.alphaES))**0.5)/(0.0+dcdp)
            #else: 
            #    kES = 1.0
            
        kES = (2*(self.w0/(self.alphaES))**0.5)/(0.0+dcdp)
        
        pnew = p

        return [dc, dp, dcdp, kES, pnew]