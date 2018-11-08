cimport Grid
cimport ReferenceState

cimport ParallelMPI
cimport TimeStepping
cimport Radiation
cimport Surface
from NetCDFIO cimport NetCDFIO_Stats


cdef class SurfaceBudgetNone:

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, Radiation.RadiationBase Ra, Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Surface.SurfaceBase Sur, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)


cdef class SurfaceBudget:
    cdef:
        double ocean_heat_flux
        double water_depth_initial
        double water_depth_final
        double water_depth_time
        double fixed_sst_time
        double water_depth

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, Radiation.RadiationBase Ra, Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Surface.SurfaceBase Sur, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class SurfaceBudgetVarying:
    cdef:
        double ocean_heat_flux
        double water_depth_initial
        double water_depth_final
        double water_depth_time
        double fixed_sst_time
        double water_depth
        str file
        double flux_ice
        double h_ice
        int t_indx
        bint prescribe_sst

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, Radiation.RadiationBase Ra, Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Surface.SurfaceBase Sur, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class SurfaceBudgetSeaice:
    cdef:
        double ocean_heat_flux
        double water_depth_initial
        double water_depth_final
        double water_depth_time
        double fixed_sst_time
        double water_depth
        str file
        int t_indx
        double ice_thickness
        double water_temperature
        double ice_temperature
        double ice_flux

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, Radiation.RadiationBase Ra, Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Surface.SurfaceBase Sur, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)