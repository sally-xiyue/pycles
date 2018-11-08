#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True
cimport mpi4py.libmpi as mpi
cimport Grid
cimport ReferenceState
cimport ParallelMPI
cimport TimeStepping
cimport Radiation
cimport Surface
from NetCDFIO cimport NetCDFIO_Stats
import cython
import cPickle
cimport numpy as np
import numpy as np
include "parameters.pxi"

import cython

def SurfaceBudgetFactory(namelist):
    if namelist['meta']['casename'] == 'ZGILS':
        return SurfaceBudget(namelist)
    elif namelist['meta']['casename'] == 'GCMFixed':
        return SurfaceBudget(namelist)
    elif namelist['meta']['casename'] == 'GCMVarying' or namelist['meta']['casename'] == 'GCMMean':
        if namelist['surface_budget']['sea_ice']:
            return SurfaceBudgetSeaice(namelist)
        else:
            return SurfaceBudgetVarying(namelist)
    else:
        return SurfaceBudgetNone()

cdef class SurfaceBudgetNone:
    def __init__(self):
        return

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef update(self,Grid.Grid Gr, Radiation.RadiationBase Ra, Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef stats_io(self, Surface.SurfaceBase Sur, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return


cdef class SurfaceBudget:
    def __init__(self, namelist):


        try:
            self.ocean_heat_flux = namelist['surface_budget']['ocean_heat_flux']
        except:
            file=namelist['gcm']['file']
            fh = open(file, 'r')
            tv_input_data = cPickle.load(fh)
            fh.close()

            lat_in = tv_input_data['lat']
            lat_idx = (np.abs(lat_in - namelist['gcm']['latitude'])).argmin()

            self.ocean_heat_flux = tv_input_data['qflux'][lat_idx]
            print 'Ocean heat flux set to: ', self.ocean_heat_flux

        try:
            self.water_depth_initial = namelist['surface_budget']['water_depth_initial']
        except:
            self.water_depth_initial = 1.0
        try:
            self.water_depth_final = namelist['surface_budget']['water_depth_final']
        except:
            self.water_depth_final = 1.0
        try:
            self.water_depth_time = namelist['surface_budget']['water_depth_time']
        except:
            self.water_depth_time = 0.0
        # Allow spin up time with fixed sst
        try:
            self.fixed_sst_time = namelist['surface_budget']['fixed_sst_time']
        except:
            self.fixed_sst_time = 0.0



        self.water_depth = self.water_depth_initial

        return



    cpdef initialize(self, Grid.Grid Gr,  NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        NS.add_ts('surface_temperature', Gr, Pa)
        return

    cpdef update(self, Grid.Grid Gr, Radiation.RadiationBase Ra, Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        cdef:
            int root = 0
            int count = 1
            double rho_liquid = 1000.0
            double mean_shf = Pa.HorizontalMeanSurface(Gr, &Sur.shf[0])
            double mean_lhf = Pa.HorizontalMeanSurface(Gr, &Sur.lhf[0])
            double net_flux, tendency



        if TS.rk_step != 0:
            return
        if TS.t < self.fixed_sst_time:
            return

        if Pa.sub_z_rank == 0:

            if TS.t > self.water_depth_time:
                self.water_depth = self.water_depth_final
            else:
                self.water_depth = self.water_depth_initial


            net_flux =  -self.ocean_heat_flux - Ra.srf_lw_up - Ra.srf_sw_up - mean_shf - mean_lhf + Ra.srf_lw_down + Ra.srf_sw_down
            tendency = net_flux/cl/rho_liquid/self.water_depth
            Sur.T_surface += tendency *TS.dt

        mpi.MPI_Bcast(&Sur.T_surface,count,mpi.MPI_DOUBLE,root, Pa.cart_comm_sub_z)

        return
    cpdef stats_io(self, Surface.SurfaceBase Sur, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        NS.write_ts('surface_temperature', Sur.T_surface, Pa)
        return

cdef class SurfaceBudgetVarying:
    def __init__(self, namelist):

        try:
            self.file=str(namelist['gcm']['file'])
        except:
            self.file = None

        try:
            self.ocean_heat_flux = namelist['surface_budget']['ocean_heat_flux']
        except:
            try:
                fh = open(self.file, 'r')
                tv_input_data = cPickle.load(fh)
                fh.close()

                self.ocean_heat_flux = tv_input_data['qflux'][0]
                print 'Ocean heat flux set to: ', self.ocean_heat_flux
            except:
                print('No ocean heat flux specified. Set it to zero!')
                self.ocean_heat_flux = 0.0
        try:
            self.water_depth_initial = namelist['surface_budget']['water_depth_initial']
        except:
            self.water_depth_initial = 1.0
        try:
            self.water_depth_final = namelist['surface_budget']['water_depth_final']
        except:
            self.water_depth_final = 1.0
        try:
            self.water_depth_time = namelist['surface_budget']['water_depth_time']
        except:
            self.water_depth_time = 0.0
        # Allow spin up time with fixed sst
        try:
            self.fixed_sst_time = namelist['surface_budget']['fixed_sst_time']
        except:
            self.fixed_sst_time = 0.0

        try:
            self.flux_ice = tv_input_data['flux_ice'][0]
            self.h_ice = tv_input_data['h_ice'][0]
        except:
            self.flux_ice = 0.0
            self.h_ice = 0.0

        try:
            self.prescribe_sst = namelist['surface']['gcm_sst']
        except:
            self.prescribe_sst = False

        self.t_indx = 0


        self.water_depth = self.water_depth_initial

        return



    cpdef initialize(self, Grid.Grid Gr,  NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        if self.prescribe_sst:
            Pa.root_print('Prescribing SST from GCM!')

        NS.add_ts('surface_temperature', Gr, Pa)
        NS.add_ts('conductive_flux_ice', Gr, Pa)
        NS.add_ts('ice_thickness', Gr, Pa)
        return

    cpdef update(self, Grid.Grid Gr, Radiation.RadiationBase Ra, Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        cdef:
            int root = 0
            int count = 1
            double rho_liquid = 1000.0
            double mean_shf
            double mean_lhf
            double net_flux, tendency

        if TS.rk_step != 0:
            return
        if TS.t < self.fixed_sst_time:
            return

        if Pa.sub_z_rank == 0:

            if TS.t > self.water_depth_time:
                self.water_depth = self.water_depth_final
            else:
                self.water_depth = self.water_depth_initial


            if self.prescribe_sst:
                if int(TS.t // (3600.0 * 6.0)) > self.t_indx:
                    self.t_indx = int(TS.t // (3600.0 * 6.0))
                    fh = open(self.file, 'r')
                    tv_input_data = cPickle.load(fh)
                    fh.close()

                    Sur.T_surface = tv_input_data['ts'][self.t_indx]

                    try:
                        self.h_ice = tv_input_data['h_ice'][self.t_indx]
                        self.flux_ice = tv_input_data['flux_ice'][self.t_indx]
                    except:
                        pass

            else:

                mean_shf = Pa.HorizontalMeanSurface(Gr, &Sur.shf[0])
                mean_lhf = Pa.HorizontalMeanSurface(Gr, &Sur.lhf[0])

                net_flux =  -self.ocean_heat_flux - Ra.srf_lw_up - Ra.srf_sw_up - \
                            mean_shf - mean_lhf + Ra.srf_lw_down + \
                            Ra.srf_sw_down

                tendency = net_flux/cl/rho_liquid/self.water_depth
                Sur.T_surface += tendency *TS.dt

        mpi.MPI_Bcast(&Sur.T_surface,count,mpi.MPI_DOUBLE,root, Pa.cart_comm_sub_z)

        return
    cpdef stats_io(self, Surface.SurfaceBase Sur, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        NS.write_ts('surface_temperature', Sur.T_surface, Pa)
        NS.write_ts('conductive_flux_ice', self.flux_ice, Pa)
        NS.write_ts('ice_thickness', self.h_ice, Pa)
        return


cdef class SurfaceBudgetSeaice:
    '''
    Sea ice model based on FMS implementation by Ian Eisenman
    "Zero-layer model" by Semtner
    Surface temperature is either sea ice top temperature (h_ice > 0) or mixed-layer temperature (h_ice = 0).
    Need to prescribe mixed-layer depth.
    Surface temperature is homogeneous in the domain.
    Work flow as follows:
    1) Calculate mixed-layer temperature tendency
    2) Calculate sea ice thickness tendency
    3) Adjust sea ice thickness if the updated T_ml < T_freeze (frazil growth at sea ice bottom)
    4) Calculate surface temperature: if updated h_ice > 0, T_surf = T_ice; otherwise, T_surf = T_ml (updated)
    '''
    def __init__(self, namelist):

        try:
            self.file=str(namelist['gcm']['file'])
        except:
            self.file = None

        try:
            self.ocean_heat_flux = namelist['surface_budget']['ocean_heat_flux']
        except:
            try:
                fh = open(self.file, 'r')
                tv_input_data = cPickle.load(fh)
                fh.close()

                self.ocean_heat_flux = tv_input_data['qflux'][0]
                print 'Ocean heat flux set to: ', self.ocean_heat_flux
            except:
                print('No ocean heat flux specified. Set it to zero!')
                self.ocean_heat_flux = 0.0
        try:
            self.water_depth_initial = namelist['surface_budget']['water_depth_initial']
        except:
            self.water_depth_initial = 1.0
        try:
            self.water_depth_final = namelist['surface_budget']['water_depth_final']
        except:
            self.water_depth_final = 1.0
        try:
            self.water_depth_time = namelist['surface_budget']['water_depth_time']
        except:
            self.water_depth_time = 0.0
        # Allow spin up time with fixed sst
        try:
            self.fixed_sst_time = namelist['surface_budget']['fixed_sst_time']
        except:
            self.fixed_sst_time = 0.0


        self.t_indx = 0


        self.water_depth = self.water_depth_initial


        self.ice_temperature = tv_input_data['ts'][0] # Sea ice surface temperature
        self.ice_thickness = tv_input_data['h_ice'][0]
        self.water_temperature = Tf #Initial ML temperature is at freezing


        return



    cpdef initialize(self, Grid.Grid Gr,  NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        # if self.prescribe_sst:
        #     Pa.root_print('Prescribing SST from GCM!')

        # if namelist['surface']['sea_ice']:
        Pa.root_print('Sea ice model is on!')

        NS.add_ts('surface_temperature', Gr, Pa)
        NS.add_ts('ice_conductive_flux', Gr, Pa)
        NS.add_ts('ice_thickness', Gr, Pa)
        return

    cpdef update(self, Grid.Grid Gr, Radiation.RadiationBase Ra, Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        cdef:
            int root = 0
            int count = 1
            double rho_liquid = 1000.0
            double mean_shf
            double mean_lhf
            double net_flux, tendency
            double delta_water_temperature, ice_thickness_tendency
            double delta_ice_thickness, delta_ice_temperature
            double mean_windspeed
            double stefan = 5.6734e-8
            double dlw_dt_surf = 0.0
            double F0 = 120.0 #Basal heat flux coefficient (W/m^2/K)
            double Lhf = 3.0e8 #Latent heat of fusion (J/m^3)
            double kice = 2.0 #Conductivity of ice (W/m/K)

        if TS.rk_step != 0:
            return
        if TS.t < self.fixed_sst_time:
            return

        if Pa.sub_z_rank == 0:

            if TS.t > self.water_depth_time:
                self.water_depth = self.water_depth_final
            else:
                self.water_depth = self.water_depth_initial


            mean_shf = Pa.HorizontalMeanSurface(Gr, &Sur.shf[0])
            mean_lhf = Pa.HorizontalMeanSurface(Gr, &Sur.lhf[0])
            dlw_dt_surf = 4.0 * stefan * Sur.T_surface ** 3.0


            #Radiative + turbulent fluxes (domain mean)
            net_flux =  -self.ocean_heat_flux - Ra.srf_lw_up - Ra.srf_sw_up - \
                        mean_shf - mean_lhf + Ra.srf_lw_down + \
                        Ra.srf_sw_down

            #Ice basal heat flux
            basal_flux = F0*(self.water_temperature - Tf)

            delta_water_temperature = (self.ocean_heat_flux - basal_flux)/self.water_depth/rho_liquid/cl*TS.dt

            ice_thickness_tendency = (- net_flux - basal_flux)/Lhf

            #Update ice thickness
            self.ice_thickness += ice_thickness_tendency*TS.dt

            #Frazil growth
            if (self.water_temperature + delta_water_temperature) < Tf:
                delta_ice_thickness = (Tf - self.water_temperature) * rho_liquid * cl * self.water_depth / Lhf
                # self.water_temperature = Tf
                delta_water_temperature = Tf - self.water_temperature
                self.ice_thickness += delta_ice_thickness

            if self.ice_thickness < 0.0: #Complete ablation
                delta_water_temperature += -self.ice_thickness * Lhf / (self.water_depth * rho_liquid * cl)
                self.ice_thickness = 0.0

            #Update mixed layer ocean temperature
            self.water_temperature += delta_water_temperature

            if self.ice_thickness > 0.0:
                #Implicitly calculate ice surface temperature
                self.ice_flux = kice*(Tf - self.ice_temperature)/self.ice_thickness
                delta_ice_temperature = (net_flux + self.ice_flux)/(kice/self.ice_thickness +
                                                                    (dlw_dt_surf + Sur.dshf_dt_surf + Sur.dlhf_dt_surf))

                self.ice_temperature += delta_ice_temperature
                Sur.T_surface = self.ice_temperature
            else:
                tendency = net_flux/cl/rho_liquid/self.water_depth
                Sur.T_surface += tendency *TS.dt

            # print('net flux ', net_flux)
            # print('basal flux ', basal_flux)
            # print('ice thickness tendency', ice_thickness_tendency)
            # print('ice thickness ', self.ice_thickness)
            # print('ice flux ', self.ice_flux)
            # print('delta ice temp ', delta_ice_temperature)
            # print('ice temperature ', self.ice_temperature)


        mpi.MPI_Bcast(&Sur.T_surface,count,mpi.MPI_DOUBLE,root, Pa.cart_comm_sub_z)

        return
    cpdef stats_io(self, Surface.SurfaceBase Sur, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        NS.write_ts('surface_temperature', Sur.T_surface, Pa)
        NS.write_ts('ice_conductive_flux', self.ice_flux, Pa)
        NS.write_ts('ice_thickness', self.ice_thickness, Pa)
        return