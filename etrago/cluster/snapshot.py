# -*- coding: utf-8 -*-
""" This module contains functions for calculating representative days/weeks
based on a pyPSA network object. It is designed to be used the the `lopf`
method.

Use:
    clusters = cluster(network, n_clusters=10)
    medoids = medoids(clusters)
    update_data_frames(network, medoids)

Remaining questions/tasks:

- Does it makes sense to cluster normed values?
- Include scaling method for yearly sums
"""

__copyright__ = "tba"
__license__ = "tba"
__author__ = "Simon Hilpert"

import os
import pandas as pd
import pyomo.environ as po
from pypsa.opf import network_lopf
from etrago.tools.utilities import results_to_csv
import tsam.timeseriesaggregation as tsam

write_results = True
home = os.path.expanduser('C:/eTraGo/etrago')
resultspath = os.path.join(home, 'snapshot-clustering-results-k10-cyclic-tsam',) # args['scn_name'])

def snapshot_clustering(network, how='daily', clusters= []):

#==============================================================================
#     # This will calculate the original problem
#     run(network=network.copy(), path=resultspath,
#     write_results=write_results, n_clusters=None)
#==============================================================================

    for c in clusters:
        path = os.path.join(resultspath, how)

        run(network=network.copy(), path=path,
            write_results=write_results, n_clusters=c,
            how=how, normed=False)

    return network


def tsam_cluster(timeseries_df, typical_periods=10, how='daily'):
    """
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timeseries to cluster

    Returns
    -------
    timeseries : pd.DataFrame
        Clustered timeseries
    """

    if how == 'daily':
        hours = 24
    if how == 'weekly':
        hours = 168

    aggregation = tsam.TimeSeriesAggregation(
        timeseries_df,
        noTypicalPeriods=typical_periods,
        rescaleClusterPeriods=False, 
        hoursPerPeriod=hours,
        clusterMethod='hierarchical')
    
    timeseries = aggregation.createTypicalPeriods()
    cluster_weights = aggregation.clusterPeriodNoOccur
    
    # get the medoids/ the clusterCenterIndices
    clusterCenterIndices= aggregation.clusterCenterIndices 
    
    # get all index for every hour of that day of the clusterCenterIndices
    start=[]
    # get the first hour of the clusterCenterIndices (days start with 0)
    for i in clusterCenterIndices:
        start.append(i*hours)
    
    # get a list with all hours belonging to the clusterCenterIndices
    nrhours=[]
    for j in start:
        nrhours.append(j)
        x=1
        while x < hours: 
            j=j+1
            nrhours.append(j)
            x=x+1
            
    # get the origial Datetimeindex
    dates = timeseries_df.iloc[nrhours].index 
        
    return timeseries, cluster_weights, dates, hours


def run(network, path, write_results=False, n_clusters=None, how='daily',
        normed=False):
    """
    """
    # reduce storage costs due to clusters

    if n_clusters is not None:
        path = os.path.join(path, str(n_clusters))

        network.cluster = True

        # calculate clusters
        tsam_ts, cluster_weights,dates,hours = tsam_cluster(prepare_pypsa_timeseries(network),
                               typical_periods=n_clusters,
                               how='daily')       
               
        update_data_frames(network, cluster_weights, dates, hours)                 
        
        
    else:
        network.cluster = False
        path = os.path.join(path, 'original')

    snapshots = network.snapshots
    
    # start powerflow calculations
    network_lopf(network, snapshots, extra_functionality = daily_bounds,
                 solver_name='gurobi')
    
    # write results to csv
    if write_results:
        results_to_csv(network, path)
        write_lpfile(network, path=os.path.join(path, "file.lp"))

    return network

def prepare_pypsa_timeseries(network, normed=False):
    """
    """
    
    if normed:
        normed_loads = network.loads_t.p_set / network.loads_t.p_set.max()
        normed_renewables = network.generators_t.p_max_pu

        df = pd.concat([normed_renewables,
                        normed_loads], axis=1)
    else:
        loads = network.loads_t.p_set
        renewables = network.generators_t.p_set
        df = pd.concat([renewables, loads], axis=1)
    
    return df


def update_data_frames(network,cluster_weights, dates,hours):
    """ Updates the snapshots, snapshots weights and the dataframes based on
    the original data in the network and the medoids created by clustering
    these original data.

    Parameters
    -----------
    network : pyPSA network object
    cluster_weights: dictionary 
    dates: Datetimeindex 


    Returns
    -------
    network

    """ 
    network.snapshot_weightings= network.snapshot_weightings.loc[dates]
    network.snapshots = network.snapshot_weightings.index
    
    #set new snapshot weights from cluster_weights
    snapshot_weightings=[]
    for i in cluster_weights.values():
        x=0
        while x<hours: 
            snapshot_weightings.append(i)
            x+=1
    for i in range(len(network.snapshot_weightings)):
        network.snapshot_weightings[i] = snapshot_weightings[i]   
    
    #put the snapshot in the right order
    network.snapshots.sort_values()
    network.snapshot_weightings.sort_index()
    
    return network

def snapshot_cluster_constraints(network, snapshots):
    """

    Notes
    ------
    Adding arrays etc. to `network.model` as attribute is not required but has
    been done as it belongs to the model as sets for constraints and variables

    """
    if network.cluster:
        sus = network.storage_units

        network.model.storages = sus.index

        if True:
        # TODO: replace condition by somthing like:
        # if network.cluster['intertemporal']:
            # somewhere get 1...365, e.g in network.cluster['candidates']
            # should be array-like
            candidates = network.cluster['candidates']

            # mapper for finding representative period (from clusterd data) for
            # every candidate
            candidate_period_mapper = network.cluster['candidate_period_mapper']

            # create set for inter-temp contraints and variables
            network.model.candidates = po.Set(initialize=candidates,
                                              ordered=True)

            # create inter soc variable for each storage and each candidate
            # (e.g. day of year for daily clustering)
            network.model.state_of_charge_inter = po.Var(
                network.model.storages, network.model.candidates
                within=po.NonNegativeReals)

            def inter_storage_soc_rule(m, s, i):
                """
                """
                if i == network.model.canadidates[-1]:
                    # if last candidate: build 'cyclic' constraint instead normal
                    # normal one (would cause error anyway as t+1 does not exist for
                    # last timestep)
                    (m.state_of_charge_inter[s, i] ==
                     m.state_of_charge_inter[s, network.model.canadidates[0]])
                else:
                    expr = (
                        m.state_of_charge_inter[s, i + 1] ==
                        m.state_of_charge_inter[s, i] *
                        (1 - network.storage_units[s].standing_loss)^24 +
                        # TODO:
                        # candidate_period_mapper needs to map to last timestep of
                        # representative period for canadidate i. which shoul match
                        # the snapshot index of course
                        m.state_of_charge[s, candidate_period_mapper[i]])
                return expr
            network.model.inter_storage_soc_constraint = po.Constraint(
                network.model.storages, network.model.candidates,
                rule=inter_storage_soc_rule)

            def inter_storage_capacity_rule(m, s, i):
                """
                """
                return (
                    m.state_of_charge_inter[s, i] *
                    (1 - network.storage_units[s].standing_loss)^24 +
                    m.state_of_charge[s, candidate_period_mapper[i]] <=
                    m.storage_p_nom[s] * network.storage_units.at[s, 'max_hours'])
            network.model.inter_storage_capacity_constraint = po.Constraint(
                network.model.storages, network.model.candidates,
                rule=inter_storage__capacity_rule)

        # take every first hour of the clustered days
        network.model.period_starts = network.snapshot_weightings.index[0::24]

        def day_rule(m, s, p):
            """
            Sets the soc of the every first hour to the soc of the last hour
            of the day (i.e. + 23 hours)
            """
            return (
                m.state_of_charge[s, p] ==
                m.state_of_charge[s, p + pd.Timedelta(hours=23)])

        network.model.period_bound = po.Constraint(
            network.model.storages, network.model.period_starts, rule=day_rule)


####################################
def manipulate_storage_invest(network, costs=None, wacc=0.05, lifetime=15):
    # default: 4500 € / MW, high 300 €/MW
    crf = (1 / wacc) - (wacc / ((1 + wacc) ** lifetime))
    network.storage_units.capital_cost = costs / crf

def write_lpfile(network=None, path=None):
    network.model.write(path,
                        io_options={'symbolic_solver_labels':True})

def fix_storage_capacity(network,resultspath, n_clusters): ###"network" dazugefügt
    path = resultspath.strip('daily')
    values = pd.read_csv(path + 'storage_capacity.csv')[n_clusters].values
    network.storage_units.p_nom_max = values
    network.storage_units.p_nom_min = values
    resultspath = 'compare-'+resultspath

