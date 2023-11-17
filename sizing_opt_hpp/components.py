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
        class Production used to represent wind or PV production system

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

        return TimeSeries(data=prod, dt=resource.dt)

    def get_tot_costs(self):
        '''Returns total costs for the production'''
        return self.p_max * self.p_cost
