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

def multifactorcollinear(n_class=3,
                         class_sizes=[4000, 4000, 4000],
                         n_groups=4,
                         group_sizes=[2, 20, 2, 20],
                         signal_strength=[.95, 0.5, 0.95, 0.5],
                         class_separability=[0.7, 0.7, 0.4, 0.40],):

    y = np.concatenate([[i]*class_sizes[i] for i in range(n_class)])
    feat_labels = np.concatenate([[i]*group_sizes[i] for i in range(n_groups)])

    n_obj = sum(class_sizes)
    xmats = []
    gmats = []
    fmats = []

    for g in range(n_groups):

        '''Factor Loading Matrix'''
        G = np.zeros((n_obj, n_class))
        G[np.arange(n_obj), y] = class_separability[g]

        all_column_indices = np.arange(n_class).reshape(1, -1).repeat(n_obj, axis=0)
        mask = all_column_indices != y[:, np.newaxis]
        remaining_col_indices = all_column_indices[mask]

        ''' Random factor loading for the remaining classes'''
        # remaining_vals = 1 - G[np.arange(n_obj), y]
        # dirichlet_samples = np.random.dirichlet(np.ones(n_class - 1), size=n_obj) * remaining_vals[:, np.newaxis]
        # G[np.repeat(np.arange(n_obj), n_class-1), remaining_col_indices] = dirichlet_samples.flatten()
        ''' Equal factor loading for the remaining classes'''
        G[np.repeat(np.arange(n_obj), n_class-1), remaining_col_indices] = (1 - class_separability[g])/n_class

        ''' Factor matrix '''
        F = np.random.normal(0, 1, (n_class, group_sizes[g]))
        ''' Group Synthetic Data'''
        X = signal_strength[g]*np.matmul(G, F) + np.sqrt(1-signal_strength[g]**2)*np.random.normal(0, 1,(n_obj, group_sizes[g]))
        xmats.append(X)
        fmats.append(F)
        gmats.append(G)
    data = np.concatenate(xmats, axis=1)
    return data, y, feat_labels

# n_objects = 10000
# clusters_number = 5
# timeseries_length = 1000
# data, key = onefactor_timeseries(n_objects, clusters_number,timeseries_length,coupling_parameter = .9, model='normal')
# cor = np.corrcoef(data)
