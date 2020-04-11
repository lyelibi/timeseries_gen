import numpy as np

def onefactor_timeseries(n_objects, clusters_number,timeseries_length,coupling_parameter = 1, model='normal'):
    
    ''' create the cluster classification key:
        just making sure that given a number of clusters all objects are assigned
        to one cluster'''
    rem = n_objects % clusters_number
    quo = n_objects // clusters_number
    key = np.repeat(range(clusters_number),quo).tolist()
    key.extend([clusters_number-1]*rem)

    '''one factor model requires:
        eta as the cluster random variable
        epsilon as the object random variable.
        We allow for the selection of gaussian or student-t models.
        i.e. stock market returns have fat tails, and aren't gaussian'''
    if model == 'normal':
        eta = np.random.normal(loc=0,scale=1,size=(clusters_number,timeseries_length))
        epsilon = np.random.normal(loc=0,scale=1,size = (n_objects,timeseries_length))
    elif model =='student':
        eta = np.random.standard_t(3,size = (clusters_number,timeseries_length))
        epsilon = np.random.standard_t(3,size = (n_objects,timeseries_length))
        
    ''' coupling paramater gs: varies from 0 to 1, and determines how much
    an object's features are tied to the cluster. Clusters where gs = 0 are very
    low density, and very noisy, whereas gs->1 means more correlated clusters'''
    if type(coupling_parameter) == int:
        gs = np.ones(clusters_number)*coupling_parameter
    else:
        gs = coupling_parameter
    
    ''' 1-factor model: as seen in Giada-Marsili 2001'''
    xi = np.zeros( (n_objects,timeseries_length))
    for i in range(n_objects):
        xi[i] = (np.sqrt(gs[key[i]])*eta[key[i]]+epsilon[i]) / np.sqrt(1 + gs[key[i]])

    return xi, key

n_objects = 10000
clusters_number = 3
timeseries_length = 100
data, key = onefactor_timeseries(n_objects, clusters_number,timeseries_length,coupling_parameter = 1, model='student')
cor = np.corrcoef(data)
