from aiida.orm.data.structure import StructureData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.base import Int, Float, Str, Bool
from aiida.orm.data.singlefile import SinglefileData
from aiida.orm.code import Code

from aiida.work.workchain import WorkChain, ToContext, Calc
from aiida.work.run import run, async, submit

from aiida_cp2k.calculations import Cp2kCalculation

import tempfile
import shutil
import numpy as np
from os import path

class SlabGeoOptWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(SlabGeoOptWorkChain, cls).define(spec)
        spec.input("cp2k_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("max_force", valid_type=Float, default=Float(0.001), required=False)
        spec.input("dftb_switch", valid_type=Bool, default=Bool(False), required=True)
        spec.input("vdw_switch", valid_type=Bool, default=Bool(False), required=False)
        spec.input("mgrid_cutoff", valid_type=Int, default=Int(600), required=False)

        spec.outline(
            cls.run_geopt
            # cls.init,
            # while_(cls.not_finished)(
            #     cls.run_geopt
            # )
        )
        spec.dynamic_output()

    # ==============================================================================================
    def init(self):
        # structure = self.inputs.structure
        # return self._submit_pw_calc(structure, label="cell_opt1", runtype='vc-relax', precision=0.5, min_kpoints=1)
        pass

    # ==============================================================================================
    def not_finished(self):
        pass
        # structure = self.inputs.structure
        # return self._submit_pw_calc(structure, label="cell_opt1", runtype='vc-relax', precision=0.5, min_kpoints=1)

    # ==============================================================================================
    def run_geopt(self):
        self.report("Running CP2K geometry optimization")

        inputs = self.build_calc_inputs(self.inputs.structure, self.inputs.cp2k_code, self.inputs.max_force, self.inputs.dftb_switch, self.inputs.mgrid_cutoff, self.inputs.vdw_switch)

        future = submit(Cp2kCalculation.process(), **inputs)
        return ToContext(geo_opt=Calc(future))

    # ==============================================================================================
    @classmethod
    def build_calc_inputs(cls, structure, code, max_force, dftb_switch, mgrid_cutoff, vdw_switch):

        inputs = {}
        inputs['_label'] = "slab_geo_opt"
        inputs['code'] = code
        inputs['file'] = {}

        # make sure we're really dealing with a gold slab
        atoms = structure.get_ase()  # slow
        try:
            first_slab_atom = np.argwhere(atoms.numbers == 79)[0, 0] + 1
            is_H = atoms.numbers[first_slab_atom-1:] == 1
            is_Au = atoms.numbers[first_slab_atom-1:] == 79
            assert np.all(np.logical_or(is_H, is_Au))
            assert np.sum(is_Au) / np.sum(is_H) == 4
        except:
            raise Exception("Structure is not a proper slab.")

        # structure
        molslab_f, mol_f = cls.mk_coord_files(atoms, first_slab_atom)
        inputs['file']['molslab_coords'] = molslab_f
        inputs['file']['mol_coords'] = mol_f

        # Au potential
        pot_f = SinglefileData(file='/project/apps/surfaces/slab/Au.pot')
        inputs['file']['au_pot'] = pot_f

        # parameters
        cell_abc = "%f  %f  %f" % (atoms.cell[0, 0], atoms.cell[1, 1], atoms.cell[2, 2])
        machine_cores = code.get_remote_computer().get_default_mpiprocs_per_machine()
        if int(dftb_switch) == 1:
            num_machines = int(np.ceil(1. + first_slab_atom/30.))
            walltime = int((1. + first_slab_atom/30.) * 3600.)
        else:
            num_machines = int(np.ceil(1. + first_slab_atom/120.))
            walltime = 24 * 3600

        inp = cls.get_cp2k_input(cell_abc, first_slab_atom, len(atoms), max_force, dftb_switch, mgrid_cutoff, vdw_switch, machine_cores*num_machines)
        inputs['parameters'] = ParameterData(dict=inp)

        # settings
        settings = ParameterData(dict={'additional_retrieve_list': ['*.pdb']})
        inputs['settings'] = settings

        # resources
        inputs['_options'] = {
            "resources": {"num_machines": num_machines},
            "max_wallclock_seconds": walltime,  # 60 min
        }

        return inputs

    # ==============================================================================================
    @classmethod
    def mk_coord_files(cls, atoms, first_slab_atom):
        mol = atoms[:first_slab_atom-1]

        tmpdir = tempfile.mkdtemp()
        molslab_fn = tmpdir + '/mol_on_slab.xyz'
        mol_fn = tmpdir + '/mol.xyz'

        atoms.write(molslab_fn)
        mol.write(mol_fn)

        molslab_f = SinglefileData(file=molslab_fn)
        mol_f = SinglefileData(file=mol_fn)

        shutil.rmtree(tmpdir)

        return molslab_f, mol_f

    # ==============================================================================================
    @classmethod
    def get_cp2k_input(cls, cell_abc, first_slab_atom, last_slab_atom, max_force, dftb_switch, mgrid_cutoff, vdw_switch, machine_cores):
        inp = {
            'MULTIPLE_FORCE_EVALS': {
                'FORCE_EVAL_ORDER': '2 3',
                'MULTIPLE_SUBSYS': 'T'
            },
            'GLOBAL': {
                'RUN_TYPE': 'GEO_OPT'
            },
            'MOTION': cls.get_motion(first_slab_atom, last_slab_atom, max_force),
            'FORCE_EVAL': [cls.force_eval_mixed(cell_abc, first_slab_atom, last_slab_atom, machine_cores),
                           cls.force_eval_fist(cell_abc),
                           ],
        }

        if int(dftb_switch) == 0:
            inp['FORCE_EVAL'].append(cls.get_force_eval_qs_dftb(cell_abc, vdw_switch))
        else:
            inp['FORCE_EVAL'].append(cls.get_force_eval_qs_dft(cell_abc, mgrid_cutoff, vdw_switch))

        return inp

    # ==============================================================================================
    @classmethod
    def get_motion(cls, first_slab_atom, last_slab_atom, max_force):
        motion = {
            'CONSTRAINT': {
                'FIXED_ATOMS': {
                    'LIST': '%d..%d' % (first_slab_atom, last_slab_atom),
                }
            },
            'GEO_OPT': {
                'MAX_FORCE': '%f' % (max_force),
                'MAX_ITER': '5000'
            },
        }

        return motion

    # ==============================================================================================
    @classmethod
    def force_eval_mixed(cls, cell_abc, first_slab_atom, last_slab_atom, machine_cores):
        first_mol_atom = 1
        last_mol_atom = first_slab_atom - 1

        force_eval = {
            'METHOD': 'MIXED',
            'MIXED': {
                'MIXING_TYPE': 'GENMIX',
                'GROUP_PARTITION': '2 %d' % (machine_cores-2),  # IN THE HYPOTHESIS OF 40 cores
                'GENERIC': {
                    'ERROR_LIMIT': '1.0E-10',
                    'MIXING_FUNCTION': 'E1+E2',
                    'VARIABLES': 'E1 E2'
                },
                'MAPPING': {
                    'FORCE_EVAL_MIXED': {
                        'FRAGMENT': [{'_': '1', ' ': '%d  %d' % (first_mol_atom, last_mol_atom)},
                                     {'_': '2', ' ': '%d  %d' % (first_slab_atom, last_slab_atom)}],
                    },
                    'FORCE_EVAL': [{'_': '1', 'DEFINE_FRAGMENTS': '1 2'},
                                   {'_': '2', 'DEFINE_FRAGMENTS': '1'}],
                }
            },
            'SUBSYS': {
                'CELL': {'ABC': cell_abc},
                'TOPOLOGY': {
                    'COORD_FILE_NAME': 'mol_on_slab.xyz',
                    'COORDINATE': 'XYZ',
                    'CONNECTIVITY': 'OFF',
                }
            }
        }

        return force_eval

    # ==============================================================================================
    @classmethod
    def force_eval_fist(cls, cell_abc):
        ff = {
            'SPLINE': {
                'EPS_SPLINE': '1.30E-5',
                'EMAX_SPLINE': '0.8',
            },
            'CHARGE': [],
            'NONBONDED': {
                'GENPOT': [],
                'LENNARD-JONES': [],
                'EAM': {
                    'ATOMS': 'Au Au',
                    'PARM_FILE_NAME': 'Au.pot',
                },
            },
        }

        for x in ('Au', 'H', 'C', 'O', 'N'):
            ff['CHARGE'].append({'ATOM': x, 'CHARGE': 0.0})

        for x in ('C', 'N', 'O', 'H'):
            ff['NONBONDED']['GENPOT'].append(
                {'ATOMS': 'Au ' + x,
                 'FUNCTION': 'A*exp(-av*r)+B*exp(-ac*r)-C/(r^6)/( 1+exp(-20*(r/R-1)) )',
                 'VARIABLES': 'r',
                 'PARAMETERS': 'A av B ac C R',
                 'VALUES': '4.13643 1.33747 115.82004 2.206825 113.96850410723008483218 5.84114',
                 'RCUT': '15'}
            )

        for x in ('C H', 'H H', 'H N', 'C C', 'C O', 'C N', 'N N', 'O H', 'O N', 'O O'):
            ff['NONBONDED']['LENNARD-JONES'].append(
                {'ATOMS': x,
                 'EPSILON': '0.0',
                 'SIGMA': '3.166',
                 'RCUT': '15'}
            )

        force_eval = {
            'METHOD': 'FIST',
            'MM': {
                'FORCEFIELD': ff,
                'POISSON': {
                    'EWALD': {
                      'EWALD_TYPE': 'none',
                    },
                },
            },
            'SUBSYS': {
                'CELL': {
                    'ABC': cell_abc,
                },
                'TOPOLOGY': {
                    'COORD_FILE_NAME': 'mol_on_slab.xyz',
                    'COORDINATE': 'XYZ',
                    'CONNECTIVITY': 'OFF',
                },
            },
        }
        return force_eval

    # ==============================================================================================
    @classmethod
    def get_force_eval_qs_dftb(cls, cell_abc, vdw_switch):
        force_eval = {
            'METHOD': 'Quickstep',
            'DFT': {
                'QS': {
                    'METHOD': 'DFTB',
                    'EXTRAPOLATION': 'ASPC',
                    'EXTRAPOLATION_ORDER': '3',
                    'DFTB': {
                        'SELF_CONSISTENT': 'T',
                        'DISPERSION': '%s' % (str(vdw_switch)[0]),
                        'ORTHOGONAL_BASIS': 'F',
                        'DO_EWALD': 'F',
                        'PARAMETER': {
                            'PARAM_FILE_PATH': 'DFTB/scc',
                            'PARAM_FILE_NAME': 'scc_parameter',
                            'UFF_FORCE_FIELD': '../uff_table',
                        },
                    },
                },
                'SCF': {
                    'MAX_SCF': '30',
                    'SCF_GUESS': 'RESTART',
                    'EPS_SCF': '1.0E-6',
                    'OT': {
                        'PRECONDITIONER': 'FULL_SINGLE_INVERSE',
                        'MINIMIZER': 'CG',
                    },
                    'OUTER_SCF': {
                        'MAX_SCF': '20',
                        'EPS_SCF': '1.0E-6',
                    },
                    'PRINT': {
                        'RESTART': {
                            'EACH': {
                                'QS_SCF': '0',
                                'GEO_OPT': '1',
                            },
                            'ADD_LAST': 'NUMERIC',
                            'FILENAME': 'RESTART'
                        },
                        'RESTART_HISTORY': {'_': 'OFF'}
                    }
                }
            },
            'SUBSYS': {
                'CELL': {'ABC': cell_abc},
                'TOPOLOGY': {
                    'COORD_FILE_NAME': 'mol.xyz',
                    'COORDINATE': 'xyz'
                }
            }
        }

        return force_eval

    # ==============================================================================================
    @classmethod
    def get_force_eval_qs_dft(cls, cell_abc, mgrid_cutoff, vdw_switch):
        force_eval = {
            'METHOD': 'Quickstep',
            'DFT': {
                'BASIS_SET_FILE_NAME': 'BASIS_MOLOPT',
                'POTENTIAL_FILE_NAME': 'POTENTIAL',
                'RESTART_FILE_NAME': './PROJ-RESTART.wfn',
                'QS': {
                    'METHOD': 'GPW',
                    'EXTRAPOLATION': 'ASPC',
                    'EXTRAPOLATION_ORDER': '3',
                    'EPS_DEFAULT': '1.0E-14',
                },
                'MGRID': {
                    'CUTOFF': '%d' % (mgrid_cutoff),
                    'NGRID': '5',
                },
                'SCF': {
                    'MAX_SCF': '20',
                    'SCF_GUESS': 'RESTART',
                    'EPS_SCF': '1.0E-7',
                    'OT': {
                        'PRECONDITIONER': 'FULL_SINGLE_INVERSE',
                        'MINIMIZER': 'CG',
                    },
                    'OUTER_SCF': {
                        'MAX_SCF': '15',
                        'EPS_SCF': '1.0E-7',
                    },
                    'PRINT': {
                        'RESTART': {
                            'EACH': {
                                'QS_SCF': '0',
                                'GEO_OPT': '1',
                            },
                            'ADD_LAST': 'NUMERIC',
                            'FILENAME': 'RESTART'
                        },
                        'RESTART_HISTORY': {'_': 'OFF'}
                    }
                },
                'XC': {
                    'XC_FUNCTIONAL': {'_': 'PBE'},
                },
            },
            'SUBSYS': {
                'CELL': {'ABC': cell_abc},
                'TOPOLOGY': {
                    'COORD_FILE_NAME': 'mol.xyz',
                    'COORDINATE': 'xyz',
                    'CENTER_COORDINATES': {'_': ''},
                },
            }
        }

        if vdw_switch is not None:
            force_eval['DFT']['XC']['VDW_POTENTIAL'] = {
                'DISPERSION_FUNCTIONAL': 'PAIR_POTENTIAL',
                'PAIR_POTENTIAL': {
                    'TYPE': 'DFTD3',
                    'CALCULATE_C9_TERM': '.TRUE.',
                    'PARAMETER_FILE_NAME': 'dftd3.dat',
                    'REFERENCE_FUNCTIONAL': 'PBE',
                    'R_CUTOFF': '15',
                }
            }

        for x in ('Au', 'C', 'Br', 'H'):
            force_eval['SUBSYS']['KIND '+x] = {
                'BASIS_SET': 'TZV2P-MOLOPT-GTH',
                'POTENTIAL': 'GTH-PBE-q4'
            }

        return force_eval

    # ==============================================================================================
    def _check_prev_calc(self, prev_calc):
        error = None
        if prev_calc.get_state() != 'FINISHED':
            error = "Previous calculation in state: "+prev_calc.get_state()
        elif "aiida.out" not in prev_calc.out.retrieved.get_folder_list():
            error = "Previous calculation did not retrive aiida.out"
        else:
            fn = prev_calc.out.retrieved.get_abs_path("aiida.out")
            content = open(fn).read()
            if "JOB DONE." not in content:
                error = "Previous calculation's aiida.out does not contain JOB DONE."
        if error:
            self.report("ERROR: "+error)
            self.abort(msg=error)
            raise Exception(error)

# EOF
