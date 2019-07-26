#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 12:07:24 2017

@author: Alexander Scheinker
"""
from __future__ import print_function, absolute_import
import numpy as np
import time
from mint.mint import *

class ESMin(Minimizer):
    def __init__(self):
        super(ESMin, self).__init__()
        self.ES = ES_min()

    def minimize(self, error_func, x):
        self.ES.bounds = self.bounds
        self.ES.max_iter = self.max_iter
        self.ES.norm_coef = self.norm_coef
        self.ES.minimize(error_func, x)
        return

class ES_min: 
    def __init__(self):

        self.norm_coef = 0.05


        w0 = 500.0


        self.kES = 0.5
        self.alphaES = 1
        self.w0 = w0
        self.dtES = 2*np.pi/(10*1.75*w0)
        self.max_iter = 500
        self.bounds = [] # [[min, max], [], []] # n = len(x)
        
        
    def minimize(self, error_func, x):
        """
        Error_func is a function of the vector x, where x is all of the
        parameters being tuned. error_func(x) actually does a lot under
        the hood, it sets actual machine (or simulation) parameters to
        the values giveb by the vector x, it then outputs the machine
        or simulation output, then computes the cost and then returns
        -1 * power for minimization
        """
        x = np.array(x)
        self.error_func = error_func
        
        # length of x
        self.nparams = len(x)
        
        # ES frequencies
        self.wES = self.w0*(0.75*(1+np.arange(self.nparams))/(self.nparams+0.0)+1)
        
        # Set upper and lower bounds
        #self.pmax = x + np.abs(x)*0.1
        #self.pmin = x - np.abs(x)*0.1
        
        # Set upper and lower bounds
        #self.pmax = np.array(x) + np.array([2, 2,2, 100, 10000])
        #self.pmin = np.array(x) - np.array([2, 2,2, 100, 10000])
        print("bounds = ", self.bounds)
        self.pmax = np.array([bound[1] for bound in self.bounds])
        self.pmin = np.array([bound[0] for bound in self.bounds])
        print(self.pmax, self.pmin)
        for i, xi in enumerate(x):
            pmax = self.pmax[i]
            pmin = self.pmin[i]
            if pmax == pmin:
                delta = np.abs(xi)*0.1
                delta = 0.1 if delta == 0 else delta  
                self.pmax[i] = xi + delta
                self.pmin[i] = xi - delta
        print("test = ", self.norm_coef, self.pmax - self.pmin, self.wES,(self.alphaES/self.wES)**0.5 )
        self.alphaES = (self.norm_coef * 2)**2*self.wES/4
        

        cost_val = error_func(x)

        pnew = x

        for step in np.arange(self.max_iter):
            
            print("step number: ", step)

               
            # Normalize parameters within [-1 1]
            pnorm = self.ES_normalize(pnew)

            # Do the ES update
            pnorm = pnorm + self.dtES*np.cos(self.wES*step*self.dtES+self.kES*cost_val)*(self.alphaES*self.wES)**0.5
            
            # Check that parameters stay within normalized range [-1, 1]
            for jn in np.arange(self.nparams):
                if abs(pnorm[jn]) > 1:
                    pnorm[jn]=pnorm[jn]/(0.0+abs(pnorm[jn]))
                
            # Calculate unnormalized parameter value for next cost "
            pnew = self.ES_UNnormalize(pnorm)

                
            cost_val = error_func(pnew)
            
            time.sleep(0.01)
            
            print("Current cost = ", cost_val)

            
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
            #p2N = p1N + (self.pmax-self.pmin)/100.0
            p2N = p1N + 0.01
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
        
        
        
        
    def ES_sensitivity_v2(self, p):
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
            dc_dp_square_sum = 0
            for p_index, p_value in enumerate(p1):
                c1 = self.error_func(p1)
                p1N = self.ES_normalize(p1)
                p2N = p1N
                # Update just one parameter at a time to get that partial derivative
                p2N[p_index] = p2N[p_index] + 0.01
                p2 = self.ES_UNnormalize(p2N)
                c2 = self.error_func(p2)
                dc_dp_square_sum = dc_dp_square_sum + ((c2-c1)/0.01)**2
            # Now update dcdp with the averaged value of all partial derivatives
            dcdp = dcdp + dc_dp_square_sum/nave
            
        dc = c2 - c1
        dp = p2 - p1
            #if dcdp > 0:
            #    kES = (2*(self.w0/(self.alphaES))**0.5)/(0.0+dcdp)
            #else: 
            #    kES = 1.0
            
        kES = (2*(1.25*self.wES/(self.alphaES))**0.5)/(0.0+dcdp**0.5)
        
        pnew = p

        return [dc, dp, dcdp, kES, pnew]
        
        
        
    def ES_sensitivity_v3(self, p):
        """
        Calculate total change in cost relative to change in parameters

        :param p:
        :return:
        """
        
        # Save initial parameter values "\
        #self.p1ES = self.ES_normalize(x)
        nave = 5.0
        dcdp = 0.0
        
        p1 = p
            
        dc_dp_square_sum = 0
        for p_index, p_value in enumerate(p1):
            dc_dp_sum = 0
            for jave_2 in np.arange(nave):
                c1 = self.error_func(p1)
                p1N = self.ES_normalize(p1)
                p2N = p1N
                # Update just one parameter at a time to get that partial derivative
                p2N[p_index] = p2N[p_index] + 0.01
                p2 = self.ES_UNnormalize(p2N)
                c2 = self.error_func(p2)
                dc_dp_sum = dc_dp_sum + ((c2-c1)/0.01)**2
            dc_dp_square_sum = dc_dp_square_sum + dc_dp_sum
        # Now update dcdp with the averaged value of all partial derivatives
        dcdp = dc_dp_square_sum/nave
            
        dc = c2 - c1
        dp = p2 - p1
            #if dcdp > 0:
            #    kES = (2*(self.w0/(self.alphaES))**0.5)/(0.0+dcdp)
            #else: 
            #    kES = 1.0
            
        kES = (2*(1.25*self.wES/(self.alphaES))**0.5)/(0.0+dcdp**0.5)
        
        pnew = p

        return [dc, dp, dcdp, kES, pnew]
        
    def ES_sensitivity_v4(self, p):
        """
        Calculate total change in cost relative to change in parameters

        :param p:
        :return:
        """
        
        # Save initial parameter values "\
        #self.p1ES = self.ES_normalize(x)
        nave = 5.0
        
        # For air coils
        nave = 1.0
        dcdp = 0.0
        
        p1 = p   
        dc_dp_square_sum = 0
        
        c1_sum = 0
        # First, with things fixed, write down cost nave times
        for jave_2 in np.arange(nave):
            c1_sum = c1_sum + self.error_func(p1)
        # Average the cost readings
        c1 = c1_sum/nave 
        
        # Step through individual parameters
        for p_index, p_value in enumerate(p1):   
            p1N = self.ES_normalize(p1)
            p2N = p1N
            # Update just one parameter at a time to get that partial derivative
            p2N[p_index] = p2N[p_index] + 0.01
            p2 = self.ES_UNnormalize(p2N)
            # Now keeping the update fixed, write down the new cost nave times
            c2_sum = 0
            for jave_2 in np.arange(nave):    
                c2_sum = c2_sum + self.error_func(p2)
            # Average the new cost readings
            c2 = c2_sum/nave
            # Calculate the partial derivative squared and add to total sum
            dc_dp_square_sum = dc_dp_square_sum + ((c2-c1)/0.01)**2
        # Now update dcdp with the averaged value of all partial derivatives
        dcdp = dc_dp_square_sum
            
        dc = c2 - c1
        dp = p2 - p1
            #if dcdp > 0:
            #    kES = (2*(self.w0/(self.alphaES))**0.5)/(0.0+dcdp)
            #else: 
            #    kES = 1.0
            
        kES = (2*(1.25*self.wES/(self.alphaES))**0.5)/(0.0+dcdp**0.5)
        
        pnew = p

        return [dc, dp, dcdp, kES, pnew]