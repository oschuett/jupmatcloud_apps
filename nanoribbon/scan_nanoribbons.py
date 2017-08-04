import aiida.common
import numpy as np
from aiida.common import aiidalogger
from aiida.orm.workflow import Workflow
from aiida.orm import Code, Computer
from aiida.orm import DataFactory
from ase.io import read, write
from aiida.orm import load_node



#number of pools
def getpools(nodes,cpu_node,kpoints):
    npools=1
    return npools

class ScananoribbonWorkflow(Workflow):

    
    magnetic=True

    def __init__(self, **kwargs):
        super(ScananoribbonWorkflow, self).__init__(**kwargs)



    def get_structure(self):
        ParameterData = DataFactory('parameter')
        params = self.get_parameters()

        StructureData = DataFactory('structure')
        r=read(params['inpcoord'])  
        s = StructureData(ase=r)
        s.store()
        return s

    def get_pw_vc_parameters(self):
        ParameterData = DataFactory('parameter')
        parameters = ParameterData(dict={
                  'CONTROL': {
                      'calculation': 'vc-relax',
                      'restart_mode': 'from_scratch',
                      'wf_collect': True,
                      'forc_conv_thr':0.0001,
                      'nstep':500,
                      },
                  'SYSTEM': {
                      'ecutwfc': 50.,
                      'ecutrho': 400.,
                      'occupations':'smearing',
                      'degauss': 0.001,
                      },
                  'ELECTRONS': {
                      'conv_thr': 1.e-8,
                      'mixing_beta':0.25,
                      'electron_maxstep':50,
                      'scf_must_converge':False,
                      },
                  'CELL': {
                  'cell_dynamics':'bfgs',
                  'cell_dofree':'x'
                  }})
        return parameters
    def get_pw_relax_parameters(self):
        
        ParameterData = DataFactory('parameter')
        parameters = ParameterData(dict={
                  'CONTROL': {
                      'calculation': 'relax',
                      'restart_mode': 'from_scratch',
                      'wf_collect': True,
                      'tstress':True,
                      'forc_conv_thr':0.0001,
                      'nstep':500,
                      },
                  'SYSTEM': {
                      'ecutwfc': 50.,
                      'ecutrho': 400.,
                      'occupations':'smearing',
                      'degauss': 0.001,
                      },
                  'ELECTRONS': {
                      'conv_thr': 1.e-8,
                      'mixing_beta':0.25,
                      'electron_maxstep':50,
                      'scf_must_converge':False,
                      },
                  'IONS':{}})
        return parameters
    
    
    def get_pw_energy_parameters(self):
        ParameterData = DataFactory('parameter')
        parameters = ParameterData(dict={
                  'CONTROL': {
                      'calculation': 'scf',
                      'restart_mode': 'from_scratch',
                      'wf_collect': True,
                      'forc_conv_thr':0.0001,
                      'nstep':500,
                      },
                  'SYSTEM': {
                      'ecutwfc': 50.,
                      'ecutrho': 400.,
                      'occupations':'smearing',
                      'degauss': 0.001,
                      },
                  'ELECTRONS': {
                      'conv_thr': 1.e-8,
                      'mixing_beta':0.25,
                      'electron_maxstep':500,
                      'scf_must_converge':False,
                      }})
        if (self.magnetic):
            parameters.dict['SYSTEM']['nspin']=2
        return parameters

    def get_bands_parameters(self):
        ParameterData = DataFactory('parameter')
        parameters = ParameterData(dict={
                  'CONTROL': {
                      'calculation': 'bands',
                      'tstress': True,
                      'wf_collect': True,
                      'forc_conv_thr':0.0001,
                      'nstep':500,
                      },
                  'SYSTEM': {
                      'ecutwfc': 50.,
                      'ecutrho': 400.,
                      'occupations':'smearing',
                      'degauss': 0.001,
                      },
                  'ELECTRONS': {
                      'conv_thr': 1.e-8,
                      'mixing_beta':0.25,
                      'electron_maxstep':100,
                      'scf_must_converge':False,
                  }})
        if (self.magnetic):
            parameters.dict['SYSTEM']['nspin']=2
        return parameters

        
    def get_hartree_parameters(self):
        ParameterData = DataFactory('parameter')
        parameters = ParameterData(dict={
                  'inputpp':{
                  'plot_num':11,
                  },
                  'plot':{
                  'nfile':1,
                  'iflag':3,
                  'output_format':6,
                  'fileout':'hartree.cube',
                  }}).store()
        return parameters
    def get_sts_parameters(self,efermi,emin=-3,emax=-2,delta_sts=0.05,degauss_sts=0.075):
        ParameterData = DataFactory('parameter')
        parameters = ParameterData(dict={
                  'inputpp':{
#                  'filplot':'sts_'+str(emin)+'_'+str(emax)+'_',
                  'plot_num':22,
                  'emin':emin,  
                  'emax':emax,  
                  'delta_sts':delta_sts,
                  'degauss_sts':degauss_sts,
                  },
                  }).store()
        return parameters
    def get_KS_parameters(self,npoints=1,klist=[1],bandlist=[1]):
        ParameterData = DataFactory('parameter')
        mydict={'inputpp':{'npoints':npoints,
#                           'filplot':'KS',
                           'plot_num':23,
                          },
               }
        i=1
        for nb in bandlist:
            for nk in klist:
                mydict['inputpp']['kpoints({})'.format(i)]=nk
                mydict['inputpp']['kbands({})'.format(i)]=nb
                i+=1
        parameters = ParameterData(dict=mydict).store()
        return parameters
    def get_totmag_parameters(self):
        ParameterData = DataFactory('parameter')
        parameters = ParameterData(dict={
                  'inputpp':{
                  'plot_num':6,
                  },
                  'plot':{
                  'nfile':1,
                  'iflag':3,
                  'output_format':6,
                  'fileout':'totmag.cube',
                  }}).store()
        return parameters
    def get_bands_postproc_parameters(self):
        ParameterData = DataFactory('parameter')
        parameters = ParameterData(dict={
                  'bands': {
                  'filband':'mol.band',
                  'lsym':True,
                  }}).store()
        return parameters
                

    def get_kpoints(self,nx=1,ny=1,nz=1, set_list=False):
        if nx < 1.0:
            nx=1
        if ny < 1.0:
            ny=1
        if nz < 1.0:
            nz=1
        KpointsData = DataFactory('array.kpoints')
        kpoints = KpointsData()
        if set_list is False:
            kpoints.set_kpoints_mesh([nx,ny,nz], offset=[0.0,0.0,0.0] )
        else:
            points=[]
            for i in np.linspace(0,0.5,nx):
                points.append([i,0.0, 0.0])
            kpoints.set_kpoints(points)
        return kpoints




    def get_pw_calculation(self, pw_structure, rem_folder,
                           pw_parameters,pw_kpoints,
                           num_machines=1):
        ParameterData = DataFactory('parameter')
        params = self.get_parameters()


        pw_codename = params['pw_codename']
        max_wallclock_seconds  = params['max_wallclock_seconds']
        num_mpiprocs_per_machine   = params['num_mpiprocs_per_machine']
        pseudo_family = params['pseudo_family']


        code = Code.get_from_string(pw_codename)
        calc = code.new_calc()
        calc.set_max_wallclock_seconds(max_wallclock_seconds)
        calc.set_resources({"num_machines":num_machines,"num_mpiprocs_per_machine": num_mpiprocs_per_machine})
        if 'queue' in params:
            calc.set_queue_name(params['queue'])
        if rem_folder :
            calc.use_parent_folder(rem_folder)
        
        npools=getpools(nodes=num_machines,cpu_node=num_mpiprocs_per_machine,kpoints=pw_kpoints)

        settings_dict={'cmdline':["-npools",str(npools)]}
        
        if pw_parameters.dict.CONTROL['calculation']=='bands':
            settings_dict['also_bands']=True
            settings_dict['cmdline']=["-npools","2"]
        if  pw_parameters.dict.CONTROL['calculation']=='scf':
            start_mag={}
            for i in  pw_structure.kinds:
                if i.name.endswith("1"):
                    start_mag[i.name]=1.0
                    self.magnetic=True
                elif i.name.endswith("2"):
                    self.magnetic=True
                    start_mag[i.name]=-1.0
                else:
                    start_mag[i.name]=0.0
            if ( self.magnetic) :
                pw_parameters.dict['SYSTEM']['starting_magnetization']=start_mag

        pw_parameters.store()
        settings = ParameterData(dict=settings_dict)
        calc.use_settings(settings)
        calc.use_code(code)
        calc.use_structure(pw_structure)
        calc.use_pseudos_from_family(pseudo_family)
        calc.use_parameters(pw_parameters)
        calc.use_kpoints(pw_kpoints)
        
        calc.store_all()
        return calc

    def get_sts_calculation(self, rem_folder, sts_parameters):
        params = self.get_parameters()
        sts_codename = params['sts_codename']
        max_wallclock_seconds  = params['max_wallclock_seconds']
        num_mpiprocs_per_machine   = params['num_mpiprocs_per_machine']


        code = Code.get_from_string(sts_codename)
        calc = code.new_calc()
        calc.set_max_wallclock_seconds(max_wallclock_seconds)
        calc.set_resources({"num_machines":1,"num_mpiprocs_per_machine": num_mpiprocs_per_machine})
        calc.use_parent_folder(rem_folder)
        if 'queue' in params:
            calc.set_queue_name(params['queue'])
        calc.store()

        calc.use_code(code)
        calc.use_parameters(sts_parameters)
        return calc

    def get_KS_calculation(self, rem_folder, KS_parameters):
        params = self.get_parameters()
        KS_codename = params['sts_codename']
        max_wallclock_seconds  = params['max_wallclock_seconds']
        num_mpiprocs_per_machine   = params['num_mpiprocs_per_machine']


        code = Code.get_from_string(KS_codename)
        calc = code.new_calc()
        calc.set_max_wallclock_seconds(max_wallclock_seconds)
        calc.set_resources({"num_machines":1,"num_mpiprocs_per_machine": num_mpiprocs_per_machine})
        calc.use_parent_folder(rem_folder)
        if 'queue' in params:
            calc.set_queue_name(params['queue'])
        calc.store()

        calc.use_code(code)
        calc.use_parameters(KS_parameters)
        return calc

    def get_hartree_calculation(self, rem_folder, hartree_parameters):
        params = self.get_parameters()
        hartree_codename = params['hartree_codename']
        max_wallclock_seconds  = params['max_wallclock_seconds']
        num_mpiprocs_per_machine   = params['num_mpiprocs_per_machine']


        code = Code.get_from_string(hartree_codename)
        calc = code.new_calc()
        calc.set_max_wallclock_seconds(max_wallclock_seconds)
        calc.set_resources({"num_machines":1,"num_mpiprocs_per_machine": num_mpiprocs_per_machine})
        calc.use_parent_folder(rem_folder)
        if 'queue' in params:
            calc.set_queue_name(params['queue'])
        calc.store()

        calc.use_code(code)
        calc.use_parameters(hartree_parameters)
        return calc


    def get_pp_calculation(self, rem_folder, pp_parameters):
        params = self.get_parameters()
        pp_codename = params['pp_codename']
        max_wallclock_seconds  = params['max_wallclock_seconds']
        num_mpiprocs_per_machine   = params['num_mpiprocs_per_machine']


        code = Code.get_from_string(pp_codename)
        calc = code.new_calc()
        calc.set_max_wallclock_seconds(max_wallclock_seconds)
        calc.set_resources({"num_machines":1,"num_mpiprocs_per_machine": num_mpiprocs_per_machine})
        calc.use_parent_folder(rem_folder)
        if 'queue' in params:
            calc.set_queue_name(params['queue'])
        calc.store()

        calc.use_code(code)
        calc.use_parameters(pp_parameters)
        return calc

    def get_bands_postproc_calculation(self, rem_folder, bands_parameters, num_machines):
        params = self.get_parameters()
        bands_codename = params['bands_codename']
        max_wallclock_seconds  = params['max_wallclock_seconds']
        num_mpiprocs_per_machine   = params['num_mpiprocs_per_machine']


        code = Code.get_from_string(bands_codename)
        calc = code.new_calc()
        calc.set_max_wallclock_seconds(max_wallclock_seconds)
        calc.set_resources({"num_machines":num_machines,"num_mpiprocs_per_machine": num_mpiprocs_per_machine})
        calc.use_parent_folder(rem_folder)
        if 'queue' in params:
            calc.set_queue_name(params['queue'])
        calc.store()

        calc.use_code(code)
        calc.use_parameters(bands_parameters)
        return calc
        

    @Workflow.step
    def cell_opt(self):
       # from aiida.orm import Code, Computer, CalculationFactory
        #aiidalogger.info("Running a unit cell optimization job")
        self.append_to_report("Running a unit cell optimization job")
        params = self.get_parameters()
        maxkpt_cell = params['kpoints']['maxkpt_cell']
        refa0 = params['kpoints']['refa0']
        struct=self.get_structure()
        x_cell=struct.cell[0][0]
        calc =        self.get_pw_calculation(struct,
                                         rem_folder=None,
                                         pw_parameters=self.get_pw_vc_parameters(),
                                         pw_kpoints=self.get_kpoints(round(refa0/x_cell*maxkpt_cell)), num_machines=params['nodes'])
        calc.label=params['struclabel']+' CELL_OPT'
        calc.description=params['struclabel']+'  CELL_OPT'


        self.attach_calculation(calc)
        self.next(self.cell_opt2)
#       self.next(self.exit)
    @Workflow.step
    def cell_opt2(self):
       # from aiida.orm import Code, Computer, CalculationFactory
        #aiidalogger.info("Running a unit cell optimization job")
        self.append_to_report("Running a unit cell optimization job")
        params = self.get_parameters()
        maxkpt_cell = params['kpoints']['maxkpt_cell']
        refa0 = params['kpoints']['refa0']
        start_calc=self.get_step_calculations(self.cell_opt)[0]
        struct=start_calc.out.output_structure
        x_cell=struct.cell[0][0]
        calc =        self.get_pw_calculation(struct,
                                         rem_folder=None,
                                         pw_parameters=self.get_pw_vc_parameters(),
                                         pw_kpoints=self.get_kpoints(round(refa0/x_cell*maxkpt_cell)), num_machines=params['nodes'])
        calc.label=params['struclabel']+' CELL_OPT2'
        calc.description=params['struclabel']+'   CELL_OPT2'
        self.attach_calculation(calc)
        self.next(self.relax)
    @Workflow.step
    def relax(self):
        self.append_to_report("Reoptimizing the geometry with more k-points")
        params = self.get_parameters()
        maxkpt_geo = params['kpoints']['maxkpt_geo']
        refa0 = params['kpoints']['refa0']
        start_calc=self.get_step_calculations(self.cell_opt2)[0]
        struct=start_calc.out.output_structure
        x_cell=struct.cell[0][0]
        calc =        self.get_pw_calculation(struct,
                                         rem_folder=None,
                                         pw_parameters=self.get_pw_relax_parameters(),
                                         pw_kpoints=self.get_kpoints(round(refa0/x_cell*maxkpt_geo)), num_machines=params['nodes'])
        calc.label=params['struclabel']+' RELAX'
        calc.description=params['struclabel']+'  RELAX'

        self.attach_calculation(calc)
        self.next(self.scf)
    @Workflow.step
    def scf(self):
        self.append_to_report("Recomputing charge-density with more k-points in the optimized geometry")
        params = self.get_parameters()
        maxkpt_scf = params['kpoints']['maxkpt_scf']
        refa0 = params['kpoints']['refa0']
#Uncomment this for a production run:
        if not params['restart']['torestart']:
          start_calc=self.get_step_calculations(self.relax)[0]
        else:
          if not params['restart']['relax_pk']:
            start_calc=self.get_step_calculations(self.relax)[0]
          else:
            start_calc=load_node(params['restart']['relax_pk'])

        struct=start_calc.out.output_structure
        x_cell=struct.cell[0][0]
        calc =        self.get_pw_calculation(struct,
                                         rem_folder=None,
                                         pw_parameters=self.get_pw_energy_parameters(),
                                         pw_kpoints=self.get_kpoints(round(refa0/x_cell*maxkpt_scf)), num_machines=params['nodes'])
        calc.label=params['struclabel']+' SCF'
        calc.description=params['struclabel']+'  SCF after RELAX'


        self.attach_calculation(calc)
        self.next(self.plot_hartree)

    @Workflow.step
    def plot_hartree(self):
        params = self.get_parameters()
        self.append_to_report("Plotting hartree potential")
        if not params['restart']['torestart']:
          start_calc=self.get_step_calculations(self.scf)[0]
        else:
          if not params['restart']['scf_pk']:
            start_calc=self.get_step_calculations(self.scf)[0]
          else:
            start_calc=load_node(params['restart']['scf_pk'])

        calc= self.get_hartree_calculation(rem_folder=start_calc.out.remote_folder,
                                      hartree_parameters=self.get_hartree_parameters() )
        calc.label=params['struclabel']+' HARTREE'
        calc.description=params['struclabel']+'  HARTREE'
        calc.set_append_text( "../../../hartree.py") 
        self.attach_calculation(calc)
        self.next(self.bands)
    @Workflow.step
    def bands(self):
        self.append_to_report("Band structure calculations")
        params = self.get_parameters()
        maxkpt_bands = params['kpoints']['maxkpt_bands']
        refa0 = params['kpoints']['refa0']
#Uncomment this for a production run:
        if not params['restart']['torestart']:
          start_calc=self.get_step_calculations(self.scf)[0]
        else:
          if not params['restart']['scf_pk']:
            start_calc=self.get_step_calculations(self.scf)[0]
          else:
            start_calc=load_node(params['restart']['scf_pk'])


        struct=start_calc.inp.structure
        x_cell=struct.cell[0][0]
        calc =        self.get_pw_calculation(struct,
                                         rem_folder=start_calc.out.remote_folder,
                                         pw_parameters=self.get_bands_parameters(),
                                         pw_kpoints=self.get_kpoints(nx=round(refa0/x_cell*maxkpt_bands),
                                         set_list=True), num_machines=params['nodes'])
        calc.label=params['struclabel']+' BANDS'
        calc.description=params['struclabel']+'  BANDS'

        self.attach_calculation(calc)
        self.next(self.KS)

    @Workflow.step
    def KS(self):
        params = self.get_parameters()
        self.append_to_report("KS STATES")
#Uncomment this for a production run:
        if not params['restart']['torestart']:
          start_calc=self.get_step_calculations(self.bands)[0]
#Uncomment this for a test run:
#       start_calc=load_node(434)
        else:

          if params['restart']['bands_pk']:
            start_calc=load_node(params['restart']['bands_pk'])
          else:
            start_calc=self.get_step_calculations(self.bands)[0]

        nel=start_calc.res.number_of_electrons
        nkpt=start_calc.res.number_of_k_points
        nspin=start_calc.res.number_of_spin_components
        midz=start_calc.inp.structure.cell[2][2]/2

        b1=int(nel/2 )
        bandlist=[]
        bandlist.append(b1-1)
        bandlist.append(b1)
        bandlist.append(b1+1)
        bandlist.append(b1+2)
        if nkpt > 10 :
            klist=[int(round(e)) for e in np.arange(1,nkpt+1-0.1*nkpt,0.1*nkpt) ]
            klist.append(nkpt)
        else:
            klist=range(1,nkpt+1)
        npoints=len(klist)*4
        calc= self.get_KS_calculation(rem_folder=start_calc.out.remote_folder,
                                      KS_parameters=self.get_KS_parameters(npoints=npoints,klist=klist,bandlist=bandlist) )
        calc.label=params['struclabel']+' KS'
        calc.description=params['struclabel']+'  KS'
        calc.set_append_text( "../../../cube-plot.py --qe_cubes aiida.filplot* --positions {} --plot --plotrange 0 0.001 \n ".format(midz+1))
        self.attach_calculation(calc)
        self.next(self.stsf)

    @Workflow.step
    def stsf(self):
        params = self.get_parameters()
        self.append_to_report("STS  filled states")
#Uncomment this for a production run:
        if not params['restart']['torestart']:
          start_calc=self.get_step_calculations(self.bands)[0]
          scf_calc=self.get_step_calculations(self.scf)[0]
#Uncomment this for a test run:
        else:
          if params['restart']['bands_pk']:
            start_calc=load_node(params['restart']['bands_pk'])
          else:
            start_calc=self.get_step_calculations(self.bands)[0]
          if params['restart']['scf_pk']:
            scf_calc=load_node(params['restart']['scf_pk'])
          else:
            scf_calc=self.get_step_calculations(self.scf)[0]

        efermi=scf_calc.res.fermi_energy
# TO DEFINE STS RANGE FROM HOMO  to HOMO-rang
#       start_calc=load_node(434)
        out=find_bandgap(start_calc.out.output_band, fermi_energy=efermi)
        if out[0] != False:
            homo=out[2]
            lumo=out[3]
        else:
            homo=efermi
            lumo=efermi
        hartree_calc=self.get_step_calculations(self.plot_hartree)[0]
        vacuum_level=hartree_calc.res.vacuum_level*27.211385/2
        rang=params['sts']['range']
        delta_sts=params['sts']['delta_sts']
        degauss_sts=params['sts']['degauss_sts']
        emin=homo-rang
        emax=homo

        calc= self.get_sts_calculation(rem_folder=start_calc.out.remote_folder,
                                      sts_parameters=self.get_sts_parameters(efermi=efermi,
                                      emin=emin,emax=emax, delta_sts=delta_sts,degauss_sts=degauss_sts) )
        calc.label=params['struclabel']+' STS FILLED'
        calc.description=params['struclabel']+'  STS FILLED'
        calc.set_append_text( "../../../qe-plot-sts.py --heights 2.0 3.0 "
        "--energy_shift {}  --qe_cubes aiida.filplot* \n ".format(-1.0*vacuum_level))
        self.attach_calculation(calc)
        self.next(self.stse)

    @Workflow.step
    def stse(self):
        params = self.get_parameters()
        self.append_to_report("STS  filled states")
#Uncomment this for a production run:
        if not params['restart']['torestart']:
          start_calc=self.get_step_calculations(self.bands)[0]
          scf_calc=self.get_step_calculations(self.scf)[0]
#Uncomment this for a test run:
        else:
          if params['restart']['bands_pk']:
            start_calc=load_node(params['restart']['bands_pk'])
          else:
            start_calc=self.get_step_calculations(self.bands)[0]
          if params['restart']['scf_pk']:
            scf_calc=load_node(params['restart']['scf_pk'])
          else:
            scf_calc=self.get_step_calculations(self.scf)[0]


        efermi=scf_calc.res.fermi_energy
# TO DEFINE STS RANGE FROM HOMO  to HOMO-rang
#       start_calc=load_node(434)
        out=find_bandgap(start_calc.out.output_band, fermi_energy=efermi)
        if out[0] != False:
            homo=out[2]
            lumo=out[3]
        else:
            homo=efermi
            lumo=efermi
        hartree_calc=self.get_step_calculations(self.plot_hartree)[0]
        vacuum_level=hartree_calc.res.vacuum_level*27.211385/2
        nel=start_calc.res.number_of_electrons
        nkpt=start_calc.res.number_of_k_points
        nspin=start_calc.res.number_of_spin_components
        rang=params['sts']['range']
        delta_sts=params['sts']['delta_sts']
        degauss_sts=params['sts']['degauss_sts']
        emin=lumo
        emax=lumo+rang

        calc= self.get_sts_calculation(rem_folder=start_calc.out.remote_folder,
                                      sts_parameters=self.get_sts_parameters(efermi=efermi,
                                      emin=emin,emax=emax, delta_sts=delta_sts,degauss_sts=degauss_sts) )
        calc.label=params['struclabel']+' STS EMPTY'
        calc.description=params['struclabel']+'  STS EMPTY'
        calc.set_append_text( "../../../qe-plot-sts.py --heights 2.0 3.0 "
        "--energy_shift {} --qe_cubes aiida.filplot* \n ".format(-1.0*vacuum_level))
        self.attach_calculation(calc)
        self.next(self.totmag)

    @Workflow.step 
    def totmag(self):
        params = self.get_parameters()
        self.append_to_report("Plotting total magnetization")
#Uncomment this for a production run:
        if not params['restart']['torestart']:
          start_calc=self.get_step_calculations(self.scf)[0]
#Uncomment this for a test run:
#       start_calc=load_node(554)
        else:
          if not params['restart']['scf_pk']:
            start_calc=self.get_step_calculations(self.scf)[0]
          else:
            start_calc=load_node(params['restart']['scf_pk'])

        midz=start_calc.inp.structure.cell[2][2]/2
        efermi=start_calc.res.fermi_energy
        calc= self.get_pp_calculation(rem_folder=start_calc.out.remote_folder,
                                      pp_parameters=self.get_totmag_parameters() )
        calc.label=params['struclabel']+' TOTMAG'
        calc.description=params['struclabel']+'  TOTMAG'
        calc.set_append_text("#convert cube to igor  \n"
                             "../../../cube-plot.py --cubes totmag.cube "
                             " --positions {} --format igor --plotrange -0.0005 0.0005 \n".format(midz+1))
        self.attach_calculation(calc)
        self.next(self.exit)


def find_bandgap(bandsdata, number_electrons=None, fermi_energy=None):
    """
    Tries to guess whether the bandsdata represent an insulator.
    This method is meant to be used only for electronic bands (not phonons)
    By default, it will try to use the occupations to guess the number of
    electrons and find the Fermi Energy, otherwise, it can be provided
    explicitely.
    Also, there is an implicit assumption that the kpoints grid is
    "sufficiently" dense, so that the bandsdata are not missing the
    intersection between valence and conduction band if present.
    Use this function with care!

    :param (float) number_electrons: (optional) number of electrons in the unit cell
    :param (float) fermi_energy: (optional) value of the fermi energy.

    :note: By default, the algorithm uses the occupations array
      to guess the number of electrons and the occupied bands. This is to be
      used with care, because the occupations could be smeared so at a
      non-zero temperature, with the unwanted effect that the conduction bands
      might be occupied in an insulator.
      Prefer to pass the number_of_electrons explicitly

    :note: Only one between number_electrons and fermi_energy can be specified at the
      same time.

    :return: (is_insulator, gap), where is_insulator is a boolean, and gap a
             float. The gap is None in case of a metal, zero when the homo is
             equal to the lumo (e.g. in semi-metals).
    """

    def nint(num):
        """
        Stable rounding function
        """
        if (num > 0):
            return int(num + .5)
        else:
            return int(num - .5)

    if fermi_energy and number_electrons:
        raise ValueError("Specify either the number of electrons or the "
                         "Fermi energy, but not both")

    try:
        stored_bands = bandsdata.get_bands()
    except KeyError:
        raise KeyError("Cannot do much of a band analysis without bands")

    if len(stored_bands.shape) == 3:
        # I write the algorithm for the generic case of having both the
        # spin up and spin down array

        # put all spins on one band per kpoint
        bands = np.concatenate([_ for _ in stored_bands], axis=1)
    else:
        bands = stored_bands

    # analysis on occupations:
    if fermi_energy is None:

        num_kpoints = len(bands)

        if number_electrons is None:
            try:
                _, stored_occupations = bandsdata.get_bands(also_occupations=True)
            except KeyError:
                raise KeyError("Cannot determine metallicity if I don't have "
                               "either fermi energy, or occupations")

            # put the occupations in the same order of bands, also in case of multiple bands
            if len(stored_occupations.shape) == 3:
                # I write the algorithm for the generic case of having both the
                # spin up and spin down array

                # put all spins on one band per kpoint
                occupations = np.concatenate([_ for _ in stored_occupations], axis=1)
            else:
                occupations = stored_occupations

            # now sort the bands by energy
            # Note: I am sort of assuming that I have an electronic ground state

            # sort the bands by energy, and reorder the occupations accordingly
            # since after joining the two spins, I might have unsorted stuff
            bands, occupations = [np.array(y) for y in zip(*[zip(*j) for j in
                                                                [sorted(zip(i[0].tolist(), i[1].tolist()),
                                                                        key=lambda x: x[0])
                                                                 for i in zip(bands, occupations)]])]
            number_electrons = int(round(sum([sum(i) for i in occupations]) / num_kpoints))

            homo_indexes = [np.where(np.array([nint(_) for _ in x]) > 0)[0][-1] for x in occupations]
            if len(set(homo_indexes)) > 1:  # there must be intersections of valence and conduction bands
                return False, None
            else:
                homo = [_[0][_[1]] for _ in zip(bands, homo_indexes)]
                try:
                    lumo = [_[0][_[1] + 1] for _ in zip(bands, homo_indexes)]
                except IndexError:
                    raise ValueError("To understand if it is a metal or insulator, "
                                     "need more bands than n_band=number_electrons")

        else:
            bands = np.sort(bands)
            number_electrons = int(number_electrons)

            # find the zero-temperature occupation per band (1 for spin-polarized
            # calculation, 2 otherwise)
            number_electrons_per_band = 4 - len(stored_bands.shape)  # 1 or 2
            # gather the energies of the homo band, for every kpoint
            homo = [i[number_electrons / number_electrons_per_band - 1] for i in bands]  # take the nth level
            try:
                # gather the energies of the lumo band, for every kpoint
                lumo = [i[number_electrons / number_electrons_per_band] for i in bands]  # take the n+1th level
            except IndexError:
                raise ValueError("To understand if it is a metal or insulator, "
                                 "need more bands than n_band=number_electrons")

        if number_electrons % 2 == 1 and len(stored_bands.shape) == 2:
            # if #electrons is odd and we have a non spin polarized calculation
            # it must be a metal and I don't need further checks
            return False, None

        # if the nth band crosses the (n+1)th, it is an insulator
        gap = min(lumo) - max(homo)
        if gap == 0.:
            return False, 0.
        elif gap < 0.:
            return False, gap
        else:
            return True, gap, max(homo), min(lumo)

    # analysis on the fermi energy
    else:
        # reorganize the bands, rather than per kpoint, per energy level

        # I need the bands sorted by energy
        bands.sort()

        levels = bands.transpose()
        max_mins = [(max(i), min(i)) for i in levels]

        if fermi_energy > bands.max():
            raise ValueError("The Fermi energy is above all band energies, "
                             "don't know what to do")
        if fermi_energy < bands.min():
            raise ValueError("The Fermi energy is below all band energies, "
                             "don't know what to do.")

        # one band is crossed by the fermi energy
        if any(i[1] < fermi_energy and fermi_energy < i[0] for i in max_mins):
            return False, 0.

        # case of semimetals, fermi energy at the crossing of two bands
        # this will only work if the dirac point is computed!
        elif (any(i[0] == fermi_energy for i in max_mins) and
                  any(i[1] == fermi_energy for i in max_mins)):
            return False, 0.
        # insulating case
        else:
            # take the max of the band maxima below the fermi energy
            homo = max([i[0] for i in max_mins if i[0] < fermi_energy])
            # take the min of the band minima above the fermi energy
            lumo = min([i[1] for i in max_mins if i[1] > fermi_energy])

            gap = lumo - homo
            if gap <= 0.:
                raise Exception("Something wrong has been implemented. "
                                "Revise the code!")
            return True, gap, homo, lumo


