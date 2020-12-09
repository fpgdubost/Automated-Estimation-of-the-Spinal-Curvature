from __future__ import division
import numpy as np


def R2(Ymodeled,Ymeasured) :
        Ymean = np.mean(Ymeasured)
        num = 0
        denum = 0
        for i in range(Ymodeled.shape[0]) :
            num += (Ymeasured[i]-Ymodeled[i])**2
            denum += (Ymeasured[i]-Ymean)**2
        return 1-num/denum

def heatSmoothing(Xs,Ys,R2Threshold=0.97,dt=1,t_final=None) :
    """
   takes as inputs :
   - Xs : numpy 1D array of numeric values, every value must be evenly spaced
   - Ys : numpy 1D array of numeric values, must have the same shape as Xs
   [
   - R2Threshold (= 0.9) : the algorithm keeps smoothing the curve as long as the correlation between the result curve and the original data is lower than this point
   - dt (= 1) : time step between 2 smoothing passes, the higher the faster, but if it is too big, the algorithm will go crazy
   ]

   returns :
   - Xs, the one which was passed as an input
   - a smoothed version of Ys

   algorithm :
   The 1D heat PDE is used to smooth out the curve.
   After considering the curve into an array of heat intensities, a finite differences approach is used to slowly transform the intensities of the heat, as it would do normaly.
   The algorithm stops once the correlation between the smoothed curve and the original curve is below the R2Threshold.
   """

    def test() :
        if t_final :
            if t > t_final :
                return False
            return True
        return (R2(T,Ys) > R2Threshold)

    # Converting the problem into a heat problem
    # The experiment happens in a linear bar, which extremities are located at X=0 and X=L.
    # Space is evenly cut into n parts.
    # T_left and T_right are isothermas (no evolution in time)
    n = Xs.shape[0] # number of spatial divisions
    T1s = Ys[0] # left "isotherma" is set to constant initial value
    T2s =  Ys[-1] # right "isotherma" is set to constant last value
    alpha = 0.01 # heat coefficient, taken in the litterature, in physics alpha = K / (rho*Cp)

    T = Ys.copy() # initial temperature intensities vector
    print Xs
    dTdt = np.empty(n) # small change in time

    t = 0

    #    while the correlation between the smoothed curve and the start curve is not too low
    # or while we've not reached t_final
    while test() :
        # > where the schema is well defined (exerywhere except the extremities)
        for i in range(1,n-1) :
            dx = Xs[i+1] - Xs[i]
            dTdt[i] = alpha*(-(T[i]-T[i-1])/dx**2 + (T[i+1]-T[i])/dx**2)
        # > limits equations
        dTdt[0] = alpha*(-(T[0]-T1s)/(Xs[1]-Xs[0])**2 + (T[1]-T[0])/(Xs[1]-Xs[0])**2)
        dTdt[n-1] = alpha*(-(T[n-1]-T[n-2])/(Xs[-1]-Xs[-2])**2 + (T2s-T[n-1])/(Xs[-1]-Xs[-2])**2)

        # updating temperature intensities
        T = T + dTdt*dt

        t += 1

    return Xs, T
