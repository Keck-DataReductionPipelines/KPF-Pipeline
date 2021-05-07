#import pylab as pl
#import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline, interp1d
import pandas as pd
import alphashape
import shapely
#from pathlib import Path
from math import ceil
from scipy import linalg
from astropy.table import Table
from astropy.io import fits

from modules.Utils.config_parser import ConfigHandler
from kpfpipe.models.level0 import KPF0 
from keckdrpframework.models.arguments import Arguments

class ContinuumNorm:
    """
    Continuum normalization module algorithm. Purpose is to measure and remove variability in blaze
    of stellar spectrum.

    Attributes:
        config_param(ConfigHandler): Instance representing pull from config file.
    """

    def __init__(self, config=None, logger=None):
        """Initializes ContinuumNorm class with rawspec, config, and logger.

        Args:
            config (configparser.ConfigParser, optional): Config context. Defaults to None.
            logger (logging.Logger, optional): Instance of logging.Logger. Defaults to None.
        
        Attributes:
            min_order (np.int): Minimum order with coherent light/flux in flux extension. 
                Pulled from config file.
            max_order (np.int): Maximum order with coherent light/flux in flux extension. 
                Pulled from config file.
            flatspec_order (np.int): Degree for flatspec polynomial fitting. Pulled from config file.
            iter (np.int): Number of iterations for Lowess smoothing. Pulled from config file.
            f (np.float): Smoothing span. Pulled from config file.
            a (np.int): Determines value of alpha divided by a. Should be a number between
                3 and 12. Default value is 6. Pulled from config file.
            q (np.float): Upper quantile q within each window will be used to fit a local
                polynomial model. Pulled from config file.
            d (np.float): The smoothing parameter for local polynomial regression, which is the 
                proportion of neighboring points to be used when fitting at one point. 
                Pulled from config file.
            run_cont_norm (str): True or false, determines whether or not to run continuum normalization 
                step in pipeline. Defaults to True.
            cont_norm_poly (str): True or false, determines whether or not to run continuum normalization 
                polynomial fitting. Defaults to True.
            cont_norm_alpha (str): True or false, determines whether or not to run continuum normalization 
                alphashape fitting. Defaults to True.


        """
        configpull=ConfigHandler(config,'PARAM')
        self.min_order=configpull.get_config_value('min_order',0)
        self.max_order=configpull.get_config_value('max_order',117)
        self.flatspec_order=configpull.get_config_value('flatspec_order', 4)
        self.iter=configpull.get_config_value('iter', 3)
        self.f=configpull.get_config_value('f', 0.25)
        self.a=configpull.get_config_value('a', 6)
        self.q=configpull.get_config_value('q', 0.95)
        self.d=configpull.get_config_value('d', 0.25)
        self.run_cont_norm = configpull.get_config_value('run_cont_norm', True)
        self.cont_norm_poly = configpull.get_config_value('cont_norm_poly', True)
        self.cont_norm_alpha = configpull.get_config_value('cont_norm_alpha', True)
        self.config=config
        self.logger=logger

    def run_all_cont_norm(self, data, dataframe):
        """Runs desired continuum normalization algorithms.

        Args:
            data (np.array): Flux spectrum data 
            dataframe (pd.dataframe): Pandas dataframe of flux,wavelength

        Returns:
            poly_norm_spec: Polynomial method normalized spectrum
            poly_yfit: Y-values of fitted polynomial from polynomial method 
            afs_norm_spec: Alphashape method normalized spectrum
            afs_yfit: Y-values of fitted curve from alphashape method 
        """

        if self.run_cont_norm == False:
            result = data
            return data

        if self.run_cont_norm == True:
            orders = order_list()

            if self.cont_norm_poly == True & self.cont_norm_alpha == True:
                    for order in orders:
                        raw_spectrum = data[order]
                        poly_norm_spec, poly_yfit = flatspec(raw_spectrum)
                        afs_norm_spec, afs_yfit = AFS(dataframe)
                        return poly_norm_spec, poly_yfit, afs_norm_spec, afs_yfit

            if self.cont_norm_poly == False & self.cont_norm_alpha == True:
                for order in orders:
                        afs_norm_spec, afs_yfit = AFS(dataframe)
                        return afs_norm_spec, afs_yfit

            if self.cont_norm_poly == True & self.cont_norm_alpha == False:
                for order in orders:
                        raw_spectrum = data[order]
                        poly_norm_spec, poly_yfit = flatspec(raw_spectrum)
                        return poly_norm_spec, poly_yfit

    def order_list(self):
        """Creates list of orders with light for algorithm to iterate through.

        Returns:
            order_list(np.ndarray): List of orders with coherent light/flux.
        """
        order_list=np.arange(self.min_order,self.max_order,1)
        return order_list

    def flatspec(self, rawspec):
        """
        Polynomial fit model of spectrum.

        Args:
            rawspec (np.ndarray): Raw spectrum data.

        Returns:
            normspec: Normalized spectrum
            yfit: Y-values of polynomial fitted to spectrum envelope
        """

        ffrac = .95 #get from config instead?
        x = np.arange(0,len(rawspec),1)

        #plt.plot(x,rawspec)
        #plt.show()
        weight = rawspec/10
        pos = np.where((np.isnan(rawspec)==False)&(np.isnan(weight)==False))[0]
        coef = np.polyfit(x[pos],rawspec[pos],self.order)
        poly = np.poly1d(coef)
        yfit = poly(x)
        for i in range(8):
            normspec = rawspec / yfit
            pos = np.where((normspec >= ffrac))[0]#& (normspec <= 2.)
            coef = np.polyfit(x[pos],rawspec[pos],i+2)
            poly = np.poly1d(coef)
            yfit = poly(x)
        
        #pl.plot(rawspec,'k-')
        #pl.plot(x,yfit,'b-')
        #pl.show()

        normspec = rawspec / yfit
        return normspec, yfit

    def lowess_ag(self,x,y):
        """Lowess smoother. Robust locally weighted regression. 
        The lowess function fits a nonparametric regression curve to a scatterplot.
        The arrays x and y contain an equal number of elements; each pair
        (x[i],y[i]) defines a data point in the scatterplot. The function
        returns the estimated (smooth) values of y. The smoothing span is given by f. 
        A larger value for f will result in a smoother curve. The number of robustifying 
        iterations is given by iter. The function will run faster with a smaller number of 
        iterations. 

        Args:
            x (np.array): Wavelength values
            y (np.array): Intensity values

        Returns:
            yest: Estimated smoothed y values (intensity)
        """
        n = len(x)
        r = int(ceil(self.f * n)) 
        h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
        w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
        w = (1 - w ** 3) ** 3
        yest = np.zeros(n)
        delta = np.ones(n)
        for iteration in range(self.iter):
            for i in range(n):
                weights = delta * w[:, i]
                b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
                A = np.array([[np.sum(weights), np.sum(weights * x)],
                            [np.sum(weights * x), np.sum(weights * x * x)]])
                beta = linalg.solve(A, b)
                yest[i] = beta[0] + beta[1] * x[i]

            residuals = y - yest
            s = np.median(np.abs(residuals))
            delta = np.clip(residuals / (6.0 * s), -1, 1)
            delta = (1 - delta ** 2) ** 2

        return yest

    def AFS(self,order):
        """Algorithm for alpha-shape fitting to spectrum.

        Args:
            order (np.int): Order of spectrum to remove blaze function. It is an n by 2 numpy array,
            where n is number of pixels. Each row is the wavelength and intensity at each pixel

        Returns:
            order["intens"].values/y_final: Normalized spectrum
            y_final: Y-values of curve fitted to spectrum envelope
        """
        # Change the column names and format of the dataset.
        order.columns=["wv","intens"]

        # n records the number of pixels.
        n=order.shape[0]
        # ref is a pandas series recording wavelength
        ref=order["wv"]
        # Variable u is the parameter u in the step 1 of AFS algorithm. It scales the intensity vector.
        u=(ref.max()-ref.min())/10/order["intens"].max()
        order["intens"] = order["intens"]*u

        # Let alpha be 1/6 of the wavelength range of the whole order.
        alpha= (order["wv"].max()-order["wv"].min())/self.a

        # This chunk of code detects loops in the boundary of the alpha shape.
        # Ususally there is only one loop(polygon).
        # Variable loop is a list.
        # The indices of the k-th loop are recorded in the k-th element of variable loop.
        loops=[]
        # Variable points is a list that represents all the sample point (lambda_i,y_i)
        points=[(order["wv"][i],order["intens"][i]) for i in range(order.shape[0])]
        #tl=time()
        alpha_shape = alphashape.alphashape(points, 1/alpha)
        #th=time()
        # print("alphashape function takes ", th-tl)


        # Input Variables:
        # polygon: shapely polygon object
        # return Variable:
        # variable indices is a list recording the indices of the vertices in the polygon
        def find_vertices(polygon):
            coordinates=list(polygon.exterior.coords)
            return [ref[ref==coordinates[i][0]].index[0] for i in range(len(coordinates))]

        # if alpha_shape is just a polygon, there is only one loop
        # if alpha_shape is a multi-polygon, we interate it and find all the loops.
        if (isinstance(alpha_shape,shapely.geometry.polygon.Polygon)):
            temp= find_vertices(alpha_shape)
            loops.append(temp)

        else:
            for polygon in alpha_shape:
                temp= find_vertices(polygon)
                loops.append(temp)

        # Use the loops to get the set W_alpha.
        # Variable Wa is a vector recording the indices of points in W_alpha.
        Wa=[0]
        for loop in loops:
            temp=loop
            temp=loop[:-1]
            temp=[i for i in temp if (i<n-1)]
            max_k=max(temp)
            min_k=min(temp)
            len_k=len(temp)
            as_k=temp
            if((as_k[0] == min_k and as_k[len_k-1] == max_k)==False):
                    index_max= as_k.index(max_k)
                    index_min= as_k.index(min_k)
                    if (index_min < index_max):
                        as_k =as_k[index_min:(index_max+1)]
                    else:
                        as_k= as_k[index_min:]+as_k[0:(index_max+1)]

            Wa=Wa+as_k
        Wa.sort()
        Wa=Wa[1:]

        # AS is an n by 2 matrix recording tilde(AS_alpha). Each row is the wavelength and intensity of one pixel.
        AS=order.copy()
        for i in range(n-1):
            indices=[m for m,v in enumerate(Wa) if v > i]
            if(len(indices)!=0):
                index=indices[0]
                a= Wa[index-1]
                b= Wa[index]
                AS["intens"][i]= AS["intens"][a]+(AS["intens"][b]-AS["intens"][a])*((AS["wv"][i]-AS["wv"][a])/(AS["wv"][b]-AS["wv"][a]))
            else:
            # AS=AS.drop(list(range(i, n)))
                break

        # Run a local polynomial on tilde(AS_alpha), as described in step 3 of the AFS algorithm.
        # Use the function loess_1d() to run a second order local polynomial.
        # Variable y_result is the predicted output from input x
        x=AS["wv"].values
        y=AS["intens"].values
        # covert x and y to R vectors
        #x = robjects.FloatVector(list(x))
        #y = robjects.FloatVector(list(y))
        #df = robjects.DataFrame({"x": x, "y": y})
        # run loess (haven't found a way to specify "control" parameters)
        #loess_fit = r.loess("y ~ x", data=df, degree = 2, span = d, surface="direct")

        B1 = lowess_ag(x, y, f=d, iter=3)
        #B1 =r.predict(loess_fit, x)
        # Add a new column called select to the matrix order.
        # order["select"] records hat(y^(1)).
        select= order["intens"].values/B1

        order["select"]=select
        # Make indices in Wa to the format of small windows.
        # Each row of the variable window is a pair of neighboring indices in Wa.
        window= np.column_stack((Wa[0:len(Wa)-1],Wa[1:]))

        # This chunk of code select the top q quantile of points in each window.
        # The point indices are recorded in variable index, which is S_alpha, q in step 4
        # of the AFS algorithm.
        index=[0]
        for i in range(window.shape[0]):
            loc_window= window[i,]
            temp = order.loc[loc_window[0]:loc_window[1]]
            index_i= temp[temp["select"] >= np.quantile(temp["select"],q)].index
            index=index+list(index_i)
        index=np.unique(index[1:])
        index=np.sort(index)

        # Run Loess for the last time
        x_2=order.iloc[index]["wv"].values
        y_2=order.iloc[index]["intens"].values
        #x_2 = robjects.FloatVector(list(x_2))
        #y_2 = robjects.FloatVector(list(y_2))
        #df2 = robjects.DataFrame({"x_2": x_2, "y_2": y_2})
        #loess_fit2 = r.loess("y_2 ~ x_2", data=df2, degree = 2, span = d,surface="direct")
        #y_final= r.predict(loess_fit2, x)
        y_final = lowess_ag(x_2,y_2,f = d, iter = 3)
        # Return the blaze-removed spectrum.
        #result= order["intens"].values/y_final

        y_final = InterpolatedUnivariateSpline(x_2,y_final, k=2)(x)
        return order["intens"].values/y_final,y_final

