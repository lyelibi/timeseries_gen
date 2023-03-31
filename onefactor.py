''' onefactor_timeries is a function which generates a data-set of synthetic
correlated timeseries
https://arxiv.org/abs/cond-mat/0101237
https://arxiv.org/abs/1908.00951


'''

import numpy as np

def onefactor_timeseries(N, C,L,gs = .8, model='normal', mu=0):
    """Returns a data-set of correlated and clustered timeseries.
    
       :param N: Number of time-series
       :param C: Number of clusters.
       :param L: Time-series length.
       :param gs: coupling parameter.
       :param model: time-series distribution, a string in ['model','student']
       
       :return: the data set of timeseries, and the cluster membership key.
    """
    key = np.sort(np.random.choice(range(C),N))

    if isinstance(gs, np.ndarray):
        gsvector = gs[key]
    elif isinstance(gs, list):
        gs = np.array(gs)
        gsvector = gs[key]

    else:
        gsvector =gs*np.ones(N)
        


    '''one factor model requires:
        eta as the cluster random variable
        epsilon as the object random variable.
        We allow for the selection of gaussian or student-t models.
        i.e. stock market returns have fat tails, and aren't gaussian'''
    if model == 'normal':
        eta = np.random.normal(loc=mu,scale=1,size=(C,L))
        epsilon = np.random.normal(loc=0,scale=1,size = (N,L))
    elif model =='student':
        eta = np.random.standard_t(2.5,size = (C,L))
        epsilon = np.random.standard_t(2.5,size = (N,L))
    
    

    ''' coupling paramater gs: varies from 0 to 1, and determines how much
    an object's features are tied to the cluster. Clusters where gs = 0 are very
    low density, and very noisy, whereas gs->1 means more correlated clusters'''    
    ''' 1-factor model: as seen in Giada-Marsili 2001'''
    
    gsvector = gsvector.reshape(-1,1)

    return gsvector*eta[key]+np.sqrt(1-gsvector**2)*epsilon, key


def onefactorwithvar(N, C,L,gs = 1, var= None):
    """Returns a data-set of correlated and clustered timeseries.
    
       :param N: Number of time-series
       :param C: Number of clusters.
       :param L: Time-series length.
       :param gs: coupling parameter.
       :param var: default value is None, can be passed an array of standard deviation values
       for every clusters.
       
       :return: the data set of timeseries, the cluster membership key, and the standard deviation array.
    """
    rem = N % C
    quo = N // C
    key = np.repeat(range(C),quo).tolist()
    key.extend([C-1]*rem)
    key = np.array(key)


    '''one factor model requires:
        eta as the cluster random variable
        epsilon as the object random variable.
        We allow for the selection of gaussian or student-t models.
        i.e. stock market returns have fat tails, and aren't gaussian'''
    if var == None:
        
        stds = np.random.uniform(0,.25,C)
    else: stds = var
        
    eta = np.array([np.random.normal(loc=0,scale=stds[i],size=L) for i in range(C)])
    epsilon = np.random.normal(loc=0,scale=1,size = (N,L))
    
    
    stds = np.repeat(stds,quo).tolist()
    stds.extend([C-1]*rem)
    stds = np.array(stds)
    ''' 1-factor model: as seen in Giada-Marsili 2001'''

    return gs*eta[key-1]+np.sqrt(1-gs**2)*epsilon, key, stds

# n_objects = 10000
# clusters_number = 5
# timeseries_length = 1000
# data, key = onefactor_timeseries(n_objects, clusters_number,timeseries_length,coupling_parameter = .9, model='normal')
# cor = np.corrcoef(data)
