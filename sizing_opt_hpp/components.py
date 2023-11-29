'''
    Module components
    Contains classes to describe the components of an hybrid power plant

    Class Storage:
        Represent a battery or hydrogen storage system that can be charg
        ed or discharged

    Class Production:
        Represent a power plant (wind or pv) production electricity from
        a given resource
'''

import matplotlib.pyplot as plt
import numpy as np

from sizing_opt_hpp.timeseries import TimeSeries

class Storage:
    '''
        class Storage used to represent battery or hydrogen storage systems

        Class members:
            - e_cap: Energy capacity [MWh]
            - p_cap: Power capacity [MW]
            - eff_in: efficiency to charge the storage [-]
            - eff_out: efficiency to discharge the storage [-]
            - e_cost: cost per unit of energy capacity [kEur/MWh]
            - p_cost: cost per unit of power capacity [kEur/MW]
    '''

    def __init__(self, e_cap=0, p_cap=0, eff_in=1, eff_out=1, e_cost=0,
                 p_cost=0) -> None:
        self.e_cap = e_cap
        self.p_cap = p_cap
        self.eff_in = eff_in
        self.eff_out = eff_out
        self.e_cost = e_cost
        self.p_cost = p_cost

    def get_av_eff(self):
        '''Returns the average efficiency'''
        return 0.5*(self.eff_in + self.eff_out)

    def get_rt_eff(self):
        '''Returns the round trip efficiency'''
        return self.eff_in*self.eff_out


class Production:
    '''
        Class Production represents wind or PV production systems

        Class members:
            - cp: Power coefficient [-]
            - p_max: Maximum power [MW]
            - p_cost: cost per unit of power capacity [kEur/MW]
    '''
    def __init__(self, cp=0, p_max=0, p_cost=0) -> None:
        self.cp = cp
        self.p_max = p_max
        self.p_cost = p_cost

    def get_production(self, resource: TimeSeries) -> TimeSeries:
        ''' 
        Calculates power production from resource timeseries (wind, ..)
        '''
        prod = self.cp * resource.data

        prod[prod>= self.p_max] = self.p_max

        return TimeSeries(data=prod, dt=resource.dt)

    def get_tot_costs(self):
        '''Returns total costs for the production'''
        return self.p_max * self.p_cost

class OpSchedule:
    '''
        Class OpSchedule describe a realization of an energy schedule 
        (or Operation Schedule) corresponding to a given list of storage 
        and renewable electric production units

        Class members:
            - storage_list: list of Storage (typically BESS and H2)
            - production_list: list of Production (Wind and PV)
            - production_p: list of TimeSeries for the power output of
            Production units
            - storage_p: list of TimeSeries for the power output of
            Storage units
            - storage_e: list of TimeSeries for the energy level (SoC)
            of Storage units            
    '''

    def __init__(self,
                 production_list, storage_list,
                 production_p, storage_p,
                 storage_e) -> None:
        '''
            Initialization function for OpSchedule
        '''
        self.production_list = production_list
        self.storage_list = storage_list
        self.production_p = production_p
        self.storage_p = storage_p
        self.storage_e = storage_e

    def plot_powerflow(self, label_list = None, xlabel = 'Time [day]',
                       ylabel1 = 'Power [MW]', ylabel2 = 'Energy [MWh]' ):
        '''
            Function to plot the power flow of the operation schedule

            Arguments:
                - label_list: list of labels to appear on the legend
                - xlabel: label of the x-axis
                - ylabel1: label of the y-axis (left) for power
                - ylabel2: label of the y-axis (right) for energy
        '''
        if label_list is None:
            cnt = 0
            label_list = []
            for storage_item in self.storage_p:
                label_list.append('Storage P ' + str(cnt))
                cnt+=1
            cnt = 0
            for production_item in self.production_p:
                label_list.append('Production P' + str(cnt))
                cnt+=1
            cnt = 0
            for storage_item in self.storage_e:
                label_list.append('Storage E ' + str(cnt))
                cnt+=1

        cnt = 0

        for storage_item in self.storage_p:
            if storage_item.data is not None:
                plt.plot(storage_item.time() * 1/24, storage_item.data,
                         label = label_list[cnt])
                cnt+=1

        for production_item in self.production_p:
            if production_item.data is not None:
                plt.plot(production_item.time() * 1/24, production_item.data,
                         label = label_list[cnt])
                cnt+=1

        plt.ylabel(ylabel1)
        plt.xlabel(xlabel)
        plt.twinx()

        for storage_item in self.storage_e:
            if storage_item.data is not None:
                plt.plot(storage_item.time()* 1/24, storage_item.data, '--',
                         label = label_list[cnt])
                cnt+=1

        plt.ylabel(ylabel2)

        # Manually specify the labels and lines for the legend
        lines = []
        labels = []

        for ax in plt.gcf().get_axes():
            for line in ax.get_lines():
                lines.append(line)
                labels.append(line.get_label())

        # Create a single legend that includes both sets of labels
        plt.legend(lines, labels, loc = "best")

    def plot_powerout(self, label_list = None, xlabel = 'Time [day]',
                       ylabel = 'Power [MW]'):
        '''
            Function to plot the power the operation schedule, focusing 
            on the power sent to the grid (power "out").

            Arguments:
                - label_list: list of labels to appear on the legend
                - xlabel: label of the x-axis
                - ylabel: label of the y-axis for power
        '''
        if label_list is None:
            cnt = 0
            label_list = []
            for storage_item in self.storage_p:
                label_list.append('Storage P ' + str(cnt))
                cnt+=1
            cnt = 0
            for production_item in self.production_p:
                label_list.append('Production P' + str(cnt))
                cnt+=1

        cnt = 0
        dt_all = self.storage_p[0].dt
        power_acc = np.zeros_like(self.storage_p[0].data)
        for storage_item in self.storage_p:
            assert storage_item.dt == dt_all
            plt.bar(storage_item.time()* 1/24, np.maximum(0,storage_item.data),
                    label = label_list[cnt], width =  dt_all/24,
                    bottom = np.maximum(0,power_acc))
            power_acc += storage_item.data
            cnt += 1

        for production_item in self.production_p:
            assert production_item.dt == dt_all
            plt.bar(production_item.time()* 1/24,
                    np.maximum(0 ,production_item.data + power_acc)
                    - np.maximum(0, power_acc),
                    label = label_list[cnt], width =  dt_all/24,
                    bottom = np.maximum(0,power_acc))
            power_acc += production_item.data
            cnt+=1

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
