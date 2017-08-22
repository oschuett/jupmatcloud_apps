from aiida.orm.data.structure import StructureData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.base import Int, Float, Str
from aiida.orm.data.structure import StructureData
from aiida.orm.data.upf import get_pseudos_dict, get_pseudos_from_structure

from aiida.orm.calculation.job.quantumespresso.pw import PwCalculation
from aiida.orm.calculation.job.quantumespresso.pp import PpCalculation

from aiida.orm import load_node
from aiida.orm.code import Code
from aiida.orm.computer import Computer
from aiida.orm.querybuilder import QueryBuilder

from aiida.work.workchain import WorkChain, ToContext, Calc
from aiida.work.run import run, async, submit, restart

class NanoribbonWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(NanoribbonWorkChain, cls).define(spec)
        spec.input("pw_code", valid_type=Code)
        spec.input("pp_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("precision", valid_type=Float, default=Float(1.0), required=False)
        spec.outline(
            #cls.run_cell_opt1,
            #cls.run_cell_opt2,
            cls.run_scf,
            cls.run_export_orbitals,
            #cls.run_export_hartree,
            #cls.calc_vacuum_level,
            #cls.run_bands,
        )
        spec.dynamic_output()

    #============================================================================================================
    def run_cell_opt1(self):
        structure = self.inputs.structure
        return self._submit_pw_calc(structure, label="cell_opt1", runtype='vc-relax', precision=0.5, min_kpoints=1)

    #============================================================================================================
    def run_cell_opt2(self):
        prev_calc = self.ctx.cell_opt1
        assert(prev_calc.get_state() == 'FINISHED')
        structure = prev_calc.out.output_structure
        return self._submit_pw_calc(structure, label="cell_opt2", runtype='vc-relax', precision=1.0, min_kpoints=1)

    #============================================================================================================
    def run_scf(self):
        #prev_calc = self.ctx.cell_opt2
        #assert(prev_calc.get_state() == 'FINISHED')
        #structure = prev_calc.out.output_structure
        structure = self.inputs.structure
        return self._submit_pw_calc(structure, label="scf", runtype='scf', precision=2.0, min_kpoints=5)

    #============================================================================================================
    def run_export_orbitals(self):
        self.report("Running pp.x to export KS orbitals")

        inputs = {}
        inputs['code'] = self.inputs.pp_code

        prev_calc = self.ctx.scf
        assert(prev_calc.get_state() == 'FINISHED')
        inputs['parent_folder'] = prev_calc.out.remote_folder

        structure = prev_calc.inp.structure
        cell_a = structure.cell[0][0]
        cell_b = structure.cell[1][1]
        cell_c = structure.cell[2][2]
        midz = cell_c/2 + 1

        nel = prev_calc.res.number_of_electrons
        nkpt = prev_calc.res.number_of_k_points
        kband1 = int(nel/2) - 1
        kband2 = int(nel/2) + 2
        kpoint1 = round(0.1*nkpt)
        kpoint2 = round(nkpt+1-0.1*nkpt)

        parameters = ParameterData(dict={
                  'inputpp':{
                      'plot_num': 7, # contribution of a selected wavefunction to charge density
                      'kpoint(1)': kpoint1,
                      'kpoint(2)': kpoint2,
                      'kband(1)': kband1,
                      'kband(2)': kband2,
                  },
                  'plot':{
                      'iflag': 2, # 2D plot
                      'output_format': 7, # format suitable for gnuplot   (2D) x, y, f(x,y)
                      'x0(1)': 0.0, #3D vector, origin of the plane (in alat units)
                      'x0(2)': 0.0,
                      'x0(3)': midz/cell_a,
                      'e1(1)': cell_a/cell_a, #3D vectors which determine the plotting plane (in alat units)
                      'e1(2)': 0.0,
                      'e1(3)': 0.0,
                      'e2(1)': 0.0,
                      'e2(2)': cell_b/cell_a,
                      'e2(3)': 0.0,
                      'nx': 10, # Number of points in the plane
                      'ny': 10,
                      'fileout': '_orbital_midz.dat',
                  },
        })
        inputs['parameters'] = parameters

 #       settings = ParameterData(dict={'additional_retrieve_list':['vacuum_hartree.dat']})
 #       inputs['settings'] = settings

        inputs['_options'] = {
            "resources": {"num_machines": 1},
            "max_wallclock_seconds": 10 * 60,
            # workaround for bug in PpCalculator. We don't want to retrive this huge intermediate file.
#            "append_text": u"rm -v aiida.filplot\n",
        }

        future = submit(PpCalculation.process(), **inputs)
        return ToContext(hartree=Calc(future))


    #============================================================================================================
    def run_export_hartree(self):
        self.report("Running pp.x to export hartree potential")

        inputs = {}
        inputs['code'] = self.inputs.pp_code

        prev_calc = self.ctx.scf
        assert(prev_calc.get_state() == 'FINISHED')
        inputs['parent_folder'] = prev_calc.out.remote_folder

        structure = prev_calc.out.output_structure
        cell_a = structure.cell[0][0]
        cell_b = structure.cell[1][1]
        cell_c = structure.cell[2][2]

        parameters = ParameterData(dict={
                  'inputpp':{
                      'plot_num': 11, # the V_bare + V_H potential
                  },
                  'plot':{
                      'iflag': 2, # 2D plot
                      'output_format': 7, # format suitable for gnuplot   (2D) x, y, f(x,y)
                      'x0(1)': 0.0, #3D vector, origin of the plane (in alat units)
                      'x0(2)': 0.0,
                      'x0(3)': cell_c/cell_a,
                      'e1(1)': cell_a/cell_a, #3D vectors which determine the plotting plane (in alat units)
                      'e1(2)': 0.0,
                      'e1(3)': 0.0,
                      'e2(1)': 0.0,
                      'e2(2)': cell_b/cell_a,
                      'e2(3)': 0.0,
                      'nx': 10, # Number of points in the plane
                      'ny': 10,
                      'fileout': 'vacuum_hartree.dat',
                  },
        })
        inputs['parameters'] = parameters

        settings = ParameterData(dict={'additional_retrieve_list':['vacuum_hartree.dat']})
        inputs['settings'] = settings

        inputs['_options'] = {
            "resources": {"num_machines": 1},
            "max_wallclock_seconds": 10 * 60,
            # workaround for bug in PpCalculator. We don't want to retrive this huge intermediate file.
            "append_text": u"rm -v aiida.filplot\n",
        }

        future = submit(PpCalculation.process(), **inputs)
        return ToContext(hartree=Calc(future))


    #============================================================================================================
    def calc_vacuum_level(self):
        self.report("Calculating vacuum level")
        prev_calc = self.ctx.hartree
        assert(prev_calc.get_state() == 'FINISHED')

        fn = prev_calc.out.retrieved.get_abs_path("vacuum_hartree.dat")
        data = np.loadtxt(fn)
        vacuum_level = np.mean(data[:,2])
        self.report("Found vacuum level: %f"%vacuum_level)

        output_parameters = ParameterData(dict={'vacuum_level':vacuum_level})
        self.out("output_parameters", output_parameters)


    #============================================================================================================
    def run_bands(self):
        prev_calc = self.ctx.scf
        assert(prev_calc.get_state() == 'FINISHED')
        structure = prev_calc.inp.structure
        parent_folder = prev_calc.out.remote_folder
        return self._submit_pw_calc(structure, label="bands", parent_folder=parent_folder, runtype='bands', precision=4.0, min_kpoints=10)

    #============================================================================================================
    def _submit_pw_calc(self, structure, label, runtype, precision, min_kpoints, parent_folder=None):
        self.report("Running pw.x for "+label)

        inputs = {}
        inputs['code'] = self.inputs.pw_code
        inputs['structure'] = structure
        inputs['parameters'] =  self._get_parameters(structure, runtype)
        inputs['pseudo'] = self._get_pseudos(structure, family_name="SSSP_acc_PBE")
        if parent_folder:
            inputs['parent_folder'] = parent_folder

        # kpoints
        cell_a = inputs['structure'].cell[0][0]
        precision *= self.inputs.precision.value
        nkpoints = max(min_kpoints, int(30 * cell_a/2.5 * precision))
        kpoints = self._get_kpoints(nkpoints)
        inputs['kpoints'] = kpoints

        # parallelization settings
        npools = min(1+nkpoints/5, 5)
        natoms = len(structure.sites)
        nnodes = (1 + natoms/30) * npools
        inputs['_options'] = {
            "resources": {"num_machines": nnodes},
            "max_wallclock_seconds": 60 * 60, # one hour
        }
        settings = {'cmdline': ["-npools",str(npools)]}

        if runtype == "bands":
            settings['also_bands'] = True # instruction for output parser

        inputs['settings'] = ParameterData(dict=settings)

#         self.report("precision %f"%precision)
#         self.report("nkpoints %d"%nkpoints)
#         self.report("npools %d"%npools)
#         self.report("natoms %d"%natoms)
#         self.report("nnodes %d"%nnodes)

        future = submit(PwCalculation.process(), **inputs)
        return ToContext(**{label:Calc(future)})

    #============================================================================================================
    def _get_parameters(self, structure, runtype):
        params = {'CONTROL': {
                     'calculation': runtype,
                     'wf_collect': True,
                     'forc_conv_thr': 0.0001,
                     'nstep': 500,
                     },
                 'SYSTEM': {
                     'ecutwfc': 50.,
                     'ecutrho': 400.,
                     'occupations': 'smearing',
                     'degauss': 0.001,
                     },
                 'ELECTRONS': {
                     'conv_thr': 1.e-8,
                     'mixing_beta': 0.25,
                     'electron_maxstep': 50,
                     'scf_must_converge': False,
                     },
        }

        if runtype == "vc-relax":
            params['CELL'] = {'cell_dofree': 'x'} # in y and z direction there is only vacuum

        #if runtype == "bands":
        #    params['CONTROL']['restart_mode'] = 'restart'

        start_mag = self._get_magnetization(structure)
        if any([m!=0 for m in start_mag.values()]):
            params['SYSTEM']['nspin'] = 2
            params['SYSTEM']['starting_magnetization'] = start_mag

        return ParameterData(dict=params)

    #============================================================================================================
    def _get_kpoints(self,nx=1,ny=1,nz=1, set_list=False):
        if nx < 1.0:
            nx=1
        if ny < 1.0:
            ny=1
        if nz < 1.0:
            nz=1

        kpoints = KpointsData()
        if set_list is False:
            kpoints.set_kpoints_mesh([nx,ny,nz], offset=[0.0,0.0,0.0] )
        else:
            points=[]
            for i in np.linspace(0,0.5,nx):
                points.append([i,0.0, 0.0])
            kpoints.set_kpoints(points)
        return kpoints

    #============================================================================================================
    def _get_pseudos(self, structure, family_name):
        kind_pseudo_dict = get_pseudos_from_structure(structure, family_name)
        pseudos = {}
        for p in kind_pseudo_dict.values():
            kinds = "_".join([k for k, v in kind_pseudo_dict.items() if v==p])
            pseudos[kinds] = p

        return pseudos

    #============================================================================================================
    def _get_magnetization(self, structure):
        start_mag={}
        for i in structure.kinds:
            if i.name.endswith("1"):
                start_mag[i.name] = 1.0
            elif i.name.endswith("2"):
                start_mag[i.name] = -1.0
            else:
                start_mag[i.name] = 0.0
        return start_mag

#EOF