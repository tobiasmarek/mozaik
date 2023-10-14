import mozaik
import numpy as np
from parameters import ParameterSet
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet
from pyNN import space
from collections import OrderedDict
from string import Template
from pyNN.errors import NothingToWriteError
import pyNN.nest
import quantities as pq
from mozaik.tools.distribution_parametrization import PyNNDistribution
from mozaik.sheets.direct_stimulator import DirectStimulator

import pytest
from unittest.mock import MagicMock
from unittest.mock import patch
from unittest.mock import call



@pytest.fixture(scope="module")
def mock_model():
    """
    Mocking the Model class to separate the testing of Sheets and the Model itself.
    Scope being equal to module means creating the mocked object only once per module.
    """
    model = MagicMock(sim = pyNN.nest)
    model.sim.state = MagicMock(dt = MagicMock())
    
    yield model




from mozaik.sheets import Sheet

class TestSheet():

    @pytest.fixture(scope="class", params=["sheet_0", "sheet_1"])
    def params(self, request):
        """A fixture for getting the current sheet parameters"""
        yield MozaikExtendedParameterSet(f"SheetsTests/param/{request.param}")


    @pytest.fixture(scope="class")
    def init_sheet(self, mock_model, params):
        """A fixture for the initialization of the Sheet object, created once"""
        yield Sheet(mock_model, 1, 1, params), params # 1, 1 - not specified in params because Sheet's required parameters do not want sx nor sy


    # __INIT__ 

    def test_init_assertions(self, init_sheet, mock_model):
        sheet, params = init_sheet

        expected_values = [mock_model.sim, mock_model.sim.state.dt, params.name, None, 1, 1, 0]
        actual_values = [sheet.sim, sheet.dt, sheet.name, sheet._pop, sheet.size_x, sheet.size_y, sheet.msc]

        assert expected_values == actual_values


    def test_init_base_component_call(self, mock_model, params):
        def set_vals(sheet, mock_model, params):
            sheet.model = mock_model
            sheet.parameters = MagicMock(cell = MagicMock(params = {}))

        with patch("mozaik.core.BaseComponent.__init__", side_effect=set_vals) as mock_parent_call, patch.object(mock_model, 'register_sheet') as mock_register_sheet:
            sheet = Sheet(mock_model, None, None, params)

            mock_parent_call.assert_called_once_with(sheet, mock_model, params)
            mock_register_sheet.assert_called_once_with(sheet)


    def test_init_dist_params(self, init_sheet):
        sheet, params = init_sheet

        for key in params.cell.params:
            if isinstance(params.cell.params[key], PyNNDistribution):
                assert key not in sheet.parameters.cell.params.keys()
            else:
                assert params.cell.params[key] == sheet.parameters.cell.params[key]


    # SETUP_TO_RECORD_LIST

    def test_setup_to_record_list(self, init_sheet, _pop_mock):
        sheet, params = init_sheet
        sheet._pop, sheet.pop = None, _pop_mock

        mozaik.setup_mpi() # initialize mozaik.rng for shuffling
        sheet.setup_to_record_list()

        if params.recorders == {}:
            assert len(sheet.to_record) == 0
        else:
            unique_variables = {}
            for recorder in params.recorders:
                for var in params.recorders[recorder].variables:
                    unique_variables[var] = True

            assert len(sheet.to_record) == len(unique_variables)
            # more tests


    # SIZE_IN_DEGREES

    def test_size_in_degrees(self, init_sheet):
        sheet, _ = init_sheet

        with pytest.raises(NotImplementedError):
            sheet.size_in_degrees()


    @pytest.fixture(params=[np.array([2,3,15.212,0.5,-2.5]), np.array([])])
    def _pop_mock(self, request):
        """Mocking the population, created once per function call"""
        all_cells = request.param

        _pop_mock = MagicMock(all_cells = all_cells)
        _pop_mock.record = MagicMock()
        _pop_mock.__getitem__.return_value = MagicMock(record = None)
        _pop_mock.__len__.return_value = len(all_cells)

        yield _pop_mock


    # POP

    def test_pop_assign(self, init_sheet, _pop_mock):
        sheet, _ = init_sheet

        with patch.object(sheet, 'setup_artificial_stimulation') as mock_arti_stim, patch.object(sheet, 'setup_initial_values') as mock_init_val:
            sheet._pop, sheet.pop = None, _pop_mock
            mock_arti_stim.assert_called_once()
            mock_init_val.assert_called_once()

        assert sheet.pop == _pop_mock
        assert len(sheet._neuron_annotations) == len(_pop_mock.all_cells)
        assert all(isinstance(sheet._neuron_annotations[i], OrderedDict) for i in range(len(sheet._neuron_annotations)))

    
    def test_pop_not_set(self, init_sheet):
        sheet, _ = init_sheet
        sheet._pop = None

        assert sheet.pop == None

    
    def test_pop_set_again(self, init_sheet, _pop_mock):
        sheet, _ = init_sheet
        sheet._pop, sheet.pop = None, _pop_mock

        with pytest.raises(Exception):
            sheet.pop = "set_value_again"


    @pytest.fixture(params=[2, 10, 0])
    def _pop_mock_and_neuron_annotations(self, request, _pop_mock):
        """
        Mocking neuron_annotations, created once per function call.
        For the creation of neuron_annotations - Population object is needed.
        Neuron_annotations with the corresponding population are returned.
        """
        width = request.param
        
        _neuron_annotations = []
        for i in range(len(_pop_mock)):
            ord_dict = OrderedDict()
            
            for j in range(width):
                ord_dict[f"key_{i},{j}"] = (False, f"ann_{i},{j}")
            
            _neuron_annotations.append(ord_dict)
        
        yield _pop_mock, _neuron_annotations


    # ADD_NEURON_ANNOTATION

    @pytest.mark.parametrize("neuron_number,key,value,protected", [(1,"key_1", "annotation", False), (2,"key_2", "annotation", True)])
    def test_add_neuron_annotation_assign(self, init_sheet, _pop_mock_and_neuron_annotations, neuron_number, key, value, protected):
        sheet, _ = init_sheet
        sheet._pop = None
        sheet.pop, sheet._neuron_annotations = _pop_mock_and_neuron_annotations
        if neuron_number >= len(sheet._neuron_annotations):
            return

        sheet.add_neuron_annotation(neuron_number, key, value, protected)

        assert sheet._neuron_annotations[neuron_number][key] == (protected, value)


    def test_add_neuron_annotation_to_protected(self, init_sheet, _pop_mock):
        sheet, _ = init_sheet
        sheet._pop, sheet.pop, sheet._neuron_annotations = None, _pop_mock, [OrderedDict([("key", (True, "old_annotation"))])]

        sheet.add_neuron_annotation(0, "key", "new_annotation", False)

        assert sheet._neuron_annotations[0]["key"] == (True, "old_annotation")


    def test_add_neuron_annotation_neuron_missing(self, init_sheet, _pop_mock):
        sheet, _ = init_sheet
        sheet._pop, sheet.pop, sheet._neuron_annotations = None, _pop_mock, []

        non_existent_neuron_number = 0

        # with pytest.raises(Exception): # fails on IndexError - not handled
        sheet.add_neuron_annotation(non_existent_neuron_number, "key", "value", False)


    def test_add_neuron_annotation_pop_not_set(self, init_sheet):
        sheet, _ = init_sheet
        sheet._pop = None
        
        #with pytest.raises(Exception): # fails on AttributeError -> only loggs the error and proceeds to another if statement
        sheet.add_neuron_annotation(0, "key", "value", False)


    # GET_NEURON_ANNOTATION

    def test_get_neuron_annotation_value(self, init_sheet, _pop_mock_and_neuron_annotations):
        sheet, _ = init_sheet
        _pop_mock, _neuron_annotations = _pop_mock_and_neuron_annotations
        sheet._pop, sheet.pop, sheet._neuron_annotations = None, _pop_mock, _neuron_annotations

        neuron_number = len(_neuron_annotations)//2

        if neuron_number >= len(_neuron_annotations):
            return
        
        keys = list(_neuron_annotations[neuron_number].keys())
        
        if len(keys) == 0:
            return
        
        key = keys[0]
        
        assert sheet.get_neuron_annotation(neuron_number, key) == _neuron_annotations[neuron_number][key][1]

    
    def test_get_neuron_annotation_key_missing(self, init_sheet, _pop_mock):
        sheet, _ = init_sheet
        sheet._pop, sheet.pop, sheet._neuron_annotations = None, _pop_mock, [OrderedDict()]

        #with pytest.raises(Exception):
        sheet.get_neuron_annotation(0, "non-existent_key")


    def test_get_neuron_annotation_neuron_missing(self, init_sheet, _pop_mock):
        sheet, _ = init_sheet
        sheet._pop, sheet.pop, sheet._neuron_annotations = None, _pop_mock, []

        non_existent_neuron_number = 0

        #with pytest.raises(Exception): # fails on IndexError - not handled
        sheet.get_neuron_annotation(non_existent_neuron_number, "key")

    
    def test_get_neuron_annotation_pop_not_set(self, init_sheet):
        sheet, _ = init_sheet
        sheet._pop = None

        #with pytest.raises(Exception): # fails on AttributeError -> only loggs the error and proceeds to another if statement
        sheet.get_neuron_annotation("neuron_number", "key")


    # GET_NEURON_ANNOTATIONS

    def test_get_neuron_annotations_list(self, init_sheet, _pop_mock_and_neuron_annotations):
        sheet, _ = init_sheet
        sheet._pop = None
        sheet.pop, _neuron_annotations = _pop_mock_and_neuron_annotations
        sheet._neuron_annotations = _neuron_annotations

        returned_neuron_annotations = []
        for ord_dict in _neuron_annotations:
            result_dict = OrderedDict()

            for key in ord_dict:
                result_dict[key] = ord_dict[key][1]

            returned_neuron_annotations.append(result_dict)
        
        assert sheet.get_neuron_annotations() == returned_neuron_annotations


    def test_get_neuron_annotations_pop_not_set(self, init_sheet):
        sheet, _ = init_sheet
        sheet._pop = None

        #with pytest.raises(Exception): # fails on TypeError but proceeds -> logger does not raise
        sheet.get_neuron_annotations()


    # DESCRIBE

    @pytest.mark.parametrize("template,render,result", [('default', lambda t, c: Template(t).safe_substitute(c), None), (None, lambda t, c: c, {'name': 'Sheet'})])
    def test_describe(self, init_sheet, template, render, result): # test more than just Sheet class
        sheet, _ = init_sheet

        assert sheet.describe(template, render) == result


    # RECORD
    
    @pytest.mark.parametrize("all_keys, not_all_keys", [(['key_0', 'key_1', 'key_3'], ['key_2', 'key_4']), ([], []), ([], ['key_0']), (['key_0'], [])])
    def test_record(self, init_sheet, _pop_mock, all_keys, not_all_keys):
        sheet, params = init_sheet
        sheet._pop, sheet.pop = None, _pop_mock    
        
        ord_dict, expected_all_calls, expected_not_all_calls = OrderedDict(), [], []
        for key in all_keys:
            ord_dict[key] = 'all'
            expected_all_calls.append(call(key, sampling_interval=params.recording_interval))
        for key in not_all_keys:
            ord_dict[key] = 'not_all'
            expected_not_all_calls.append(call(key, sampling_interval=params.recording_interval))

        sheet.to_record = ord_dict

        with patch.object(sheet, 'setup_to_record_list') as mock_setup_record, patch.object(_pop_mock, 'record') as mock_pop_record, patch.object(_pop_mock['not_all'], 'record') as mock_not_all_record:
            sheet.record()
            mock_pop_record.assert_has_calls(expected_all_calls)
            mock_not_all_record.assert_has_calls(expected_not_all_calls)
            mock_setup_record.assert_called_once()


    # GET_DATA

    @pytest.mark.parametrize("stimulus_duration", [None, 4.2, 0, -1]) # unused?
    def test_get_data_assertions(self, init_sheet, _pop_mock, stimulus_duration):
        def get_mocked_st():
            st = MagicMock(value = 50, t_start = 10, t_stop = 20, annotations = OrderedDict([('source_id', 1)])) # parametrize?
            st.__len__.return_value = 15
            return st

        def isub_side_effect(obj_ref, x):
            obj_ref.value -= x
            return obj_ref

        sheet, params = init_sheet

        segment = MagicMock(annotations = OrderedDict(), spiketrains = [get_mocked_st() for i in range(5)], analogsignals = [get_mocked_st() for i in range(5)])
        block_mock = MagicMock(segments = [segment])
        _pop_mock.get_data.return_value = block_mock

        sheet._pop, sheet.pop = None, _pop_mock

        patches = []
        for mocked_st in segment.spiketrains: # patching every 'st -= t_start' call of every mocked_st object
            patcher = patch.object(mocked_st, '__isub__', new = isub_side_effect)
            patcher.start()
            patches.append(patcher)

        new_segment = sheet.get_data(stimulus_duration = stimulus_duration)
        
        for patcher in patches:
                patcher.stop()

        assert sheet.msc > 0 or (len(segment.spiketrains) == 0 and np.isnan(sheet.msc))
        assert new_segment.annotations["sheet_name"] == params['name']
        
        if (stimulus_duration != None):
            assert all(new_segment.spiketrains[i].t_start == 0 * pq.ms for i in range(len(segment.spiketrains)))
            assert all(new_segment.spiketrains[i].t_stop == 10 for i in range(len(segment.spiketrains)))
            assert all(new_segment.spiketrains[i].value == 40 for i in range(len(segment.spiketrains)))
            assert all(new_segment.analogsignals[i].t_start == 0 * pq.ms for i in range(len(segment.analogsignals)))

    
    def test_get_data_call(self, init_sheet, _pop_mock):
        sheet, _ = init_sheet
        sheet._pop, sheet.pop = None, _pop_mock
        
        with patch.object(_pop_mock, 'get_data') as mock_get_data:
            sheet.get_data()
            mock_get_data.assert_called_once_with(['spikes', 'v', 'gsyn_exc', 'gsyn_inh'], clear=True)


    def test_get_data_pop_not_set(self, init_sheet):
        sheet, _ = init_sheet
        sheet._pop = None

        #with pytest.raises(Exception): # fails on NameError: name 'errmsg' is not defined
        sheet.get_data()
    

    def test_get_data_nothing_to_write_error(self, init_sheet, _pop_mock):
        sheet, _ = init_sheet

        _pop_mock.get_data = MagicMock(side_effect = NothingToWriteError("msg"))
        sheet._pop, sheet.pop = None, _pop_mock
        logger = MagicMock(debug = None)

        with patch.object(logger, 'debug') as mock_debug:
            sheet.get_data()
            mock_debug.assert_called_once()


    # MEAN_SPIKE_COUNT

    @pytest.mark.parametrize("value", [10, -10, 0.0000001])
    def test_mean_spike_count(self, init_sheet, value):
        sheet, _ = init_sheet
        sheet.msc = value

        assert sheet.mean_spike_count() == sheet.msc


    # PREPARE_ARTIFICIAL_STIMULATION

    @pytest.mark.parametrize("duration,offset,num_of_artificial_stim,num_of_additional_stim", [(0.6, 0.4, 1, 0), (0, 0, 0, 0), (1, 1, 0, 2)])
    def test_prepare_artificial_stimulation(self, init_sheet, duration, offset, num_of_artificial_stim, num_of_additional_stim):
        sheet, _ = init_sheet

        ds = MagicMock() # mocking stimulators
        sheet.artificial_stimulators = [ds for i in range(num_of_artificial_stim)]
        additional_stimulators = [ds for i in range(num_of_additional_stim)]
        total_calls = num_of_artificial_stim + num_of_additional_stim
        
        with patch.object(ds, 'prepare_stimulation') as mock_prep_stim:
            sheet.prepare_artificial_stimulation(duration, offset, additional_stimulators)
            if total_calls != 0:
                mock_prep_stim.assert_called_with(duration, offset)

            assert mock_prep_stim.call_count == total_calls


    # SETUP_ARTIFICIAL_STIMULATION

    def test_setup_artificial_stimulation(self, init_sheet):
        sheet, params = init_sheet

        sheet.setup_artificial_stimulation()

        for ds in sheet.artificial_stimulators:
            assert isinstance(ds, DirectStimulator)

        assert len(sheet.artificial_stimulators) == len(params.artificial_stimulators)


    # SETUP_INITIAL_VALUES

    def test_setup_initial_values_calls(self, init_sheet, _pop_mock):
        sheet, params = init_sheet
        sheet._pop, sheet.pop = None, _pop_mock

        with patch.object(_pop_mock, 'initialize') as mock_pop_init, patch.object(_pop_mock, 'set') as mock_pop_set:
            sheet.setup_initial_values()
            mock_pop_init.assert_called_once_with(**params.cell.initial_values)
            mock_pop_set.assert_called_once_with(**sheet.dist_params)


    def test_setup_initial_values_pine(self, init_sheet):
        pass # pyNN population with initialize() and set()




from mozaik.sheets.vision import RetinalUniformSheet

class TestRetinalUniformSheet:

    @pytest.fixture(scope="class", params=["retinal_sheet"])
    def params(self, request):
        yield MozaikExtendedParameterSet(f"SheetsTests/param/{request.param}")


    @pytest.fixture(scope="class")
    def init_sheet(self, mock_model, params):
        yield RetinalUniformSheet(mock_model, params), params


    # __INIT__

    def test_init_assertions(self, init_sheet):
        sheet, params = init_sheet
        
        assert isinstance(sheet.pop, pyNN.nest.Population)
        assert len(sheet.pop) == int(params.sx * params.sy * params.density)

        space_parameters = sheet.pop.celltype.parameter_space._parameters
        assert all(key in space_parameters and space_parameters[key] == value for key, value in params.cell.params.items())

        boundary = sheet.pop.structure.boundary
        assert isinstance(sheet.pop.structure, space.RandomStructure)
        assert isinstance(boundary, space.Cuboid)
        assert (boundary.width, boundary.height, boundary.depth) == (params.sx, params.sy, 0)
        
        # sheet.pop.initial_values['v'][0] # is checking initial_values required?

        assert sheet.pop.label == params.name

        assert sheet.pop._positions is not None


    def test_init_sheet_call(self, mock_model, params):
        def set_vals(sheet, mock_model, sx, sy, params):
            sheet.model = mock_model
            sheet.sim, sheet.parameters, sheet.name = sheet.model.sim, params, params.name
            sheet.size_x, sheet.size_y, sheet._pop, sheet.dist_params = sx, sy, None, OrderedDict()

        with patch.object(Sheet, '__init__', side_effect=set_vals) as mock_parent_call:
            sheet = RetinalUniformSheet(mock_model, params)
            mock_parent_call.assert_called_once_with(sheet, mock_model, params.sx, params.sy, params)


    # SIZE_IN_DEGREES

    def test_size_in_degrees(self, init_sheet):
        sheet, params = init_sheet

        assert sheet.size_in_degrees() == (params.sx, params.sy)




from mozaik.sheets.vision import SheetWithMagnificationFactor

class MockedSheetWithMagnificationFactor(SheetWithMagnificationFactor):
    """Because of missing 'density' in required_parameters of SheetWithMagnificationFactor class, a helper child class had to be created"""
    required_parameters = ParameterSet({'density': float,})

    def __init__(self, model, parameters):
        SheetWithMagnificationFactor.__init__(self, model, parameters) 

class TestSheetWithMagnificationFactor:   

    @pytest.fixture(scope="class", params=["sheet_w_mag_factor"])
    def params(self, request):
        yield MozaikExtendedParameterSet(f"SheetsTests/param/{request.param}")


    @pytest.fixture(scope="class") # (magnification_factor, sx, sy)
    def init_sheet(self, mock_model, params):
        yield MockedSheetWithMagnificationFactor(mock_model, params), params


    # __INIT__

    def test_init_assertions(self, init_sheet):
        sheet, params = init_sheet

        assert sheet.magnification_factor == params.magnification_factor


    def test_init_sheet_call(self, mock_model, params):
        mag = params.magnification_factor

        with patch.object(Sheet, '__init__') as mock_parent_call:
            sheet = SheetWithMagnificationFactor(mock_model, params)
            mock_parent_call.assert_called_once_with(sheet, mock_model, params.sx / mag, params.sy / mag, params)


    # VF_2_CS
    
    @pytest.mark.parametrize("degree_x,degree_y", [(10, 20), (5, 5)])
    def test_vf_2_cs(self, init_sheet, degree_x, degree_y):
        sheet, params = init_sheet
        mag = params.magnification_factor
        
        assert sheet.vf_2_cs(degree_x, degree_y) == (mag * degree_x, mag * degree_y)


    # CS_2_VF

    @pytest.mark.parametrize("micro_meters_x,micro_meters_y", [(10, 20), (5, 5)])
    def test_cs_2_vf(self, init_sheet, micro_meters_x, micro_meters_y):
        sheet, params = init_sheet
        mag = params.magnification_factor
        
        assert sheet.cs_2_vf(micro_meters_x, micro_meters_y) == (micro_meters_x / mag, micro_meters_y / mag)


    # DVF_2_DCS

    @pytest.mark.parametrize("distance_vf", [1,2,3])
    def test_dvf_2_dcs(self, init_sheet, distance_vf):
        sheet, params = init_sheet

        assert sheet.dvf_2_dcs(distance_vf) == distance_vf * params.magnification_factor


    # SIZE_IN_DEGREES

    def test_size_in_degrees(self, init_sheet):
        sheet, params = init_sheet
        mag = params.magnification_factor
 
        assert sheet.size_in_degrees() == (params.sx / mag, params.sy / mag)




from mozaik.sheets.vision import VisualCorticalUniformSheet

class TestVisualCorticalUniformSheet:

    @pytest.fixture(scope="class", params=["visual_cortical_sheet"])
    def params(self, request):
        yield MozaikExtendedParameterSet(f"SheetsTests/param/{request.param}")


    @pytest.fixture(scope="class") # (magnification_factor, sx, sy, density)
    def init_sheet(self, mock_model, params): # fails for smaller values (1.0, 80, 90, 3), (1, 60, 50, 6), (1, 180, 45, 5)])
        yield VisualCorticalUniformSheet(mock_model, params), params


    # __INIT__

    def test_init_assertions(self, init_sheet):
        sheet, params = init_sheet
        
        assert isinstance(sheet.pop, pyNN.nest.Population)
        assert len(sheet.pop) == int(params.sx * params.sy / 1000000 * params.density)

        space_parameters = sheet.pop.celltype.parameter_space._parameters
        assert all(key in space_parameters and space_parameters[key] == value for key, value in params.cell.params.items())

        boundary = sheet.pop.structure.boundary
        assert isinstance(sheet.pop.structure, space.RandomStructure)
        assert isinstance(boundary, space.Cuboid)
        assert (boundary.width, boundary.height, boundary.depth) == (params.sx / params.magnification_factor, params.sy / params.magnification_factor, 0)

        assert sheet.pop.label == params.name

        assert sheet.pop._positions is not None


    def test_init_sheet_w_mag_factor_call(self, mock_model, params):
        def set_vals(sheet, mock_model, params):
            sheet.model = mock_model
            sheet.sim, sheet.parameters, sheet.name = sheet.model.sim, params, params.name
            sheet._pop, sheet.dist_params = None, OrderedDict()
            sheet.magnification_factor = params.magnification_factor

        with patch.object(SheetWithMagnificationFactor, '__init__', side_effect=set_vals) as mock_parent_call:
            sheet = VisualCorticalUniformSheet(mock_model, params)
            mock_parent_call.assert_called_once_with(sheet, mock_model, params)




from mozaik.sheets.vision import VisualCorticalUniformSheet3D

class TestVisualCorticalUniformSheet3D:

    @pytest.fixture(scope="class", params=["visual_cortical_sheet_3d"])
    def params(self, request):
        yield MozaikExtendedParameterSet(f"SheetsTests/param/{request.param}")


    @pytest.fixture(scope="class") # (magnification_factor, sx, sy, density, min_depth, max_depth) # What happens if min_depth > max depth
    def init_sheet(self, mock_model, params): # fails for smaller values (1.2, 2.1, 3, 6, 3, 3), (4., .5, 6, 5, 5, 10)]) 
        yield VisualCorticalUniformSheet3D(mock_model, params), params


    # __INIT__

    def test_init_assertions(self, init_sheet):
        sheet, params = init_sheet
        
        assert isinstance(sheet.pop, pyNN.nest.Population)
        assert len(sheet.pop) == int(params.sx * params.sy / 1000000 * params.density)

        space_parameters = sheet.pop.celltype.parameter_space._parameters
        assert all(key in space_parameters and space_parameters[key] == value for key, value in params.cell.params.items())

        boundary = sheet.pop.structure.boundary
        assert isinstance(sheet.pop.structure, space.RandomStructure)
        assert isinstance(boundary, space.Cuboid)
        mag = params.magnification_factor
        assert (boundary.width, boundary.height, boundary.depth) == (params.sx / mag, params.sy/ mag, params.max_depth-params.min_depth)
        assert sheet.pop.structure.origin == (0.0, 0.0, (params.min_depth+params.max_depth)/2.0) 

        assert sheet.pop.label == params.name

        assert sheet.pop._positions is not None


    def test_init_sheet_w_mag_factor_call(self, mock_model, params):
        def set_vals(sheet, mock_model, params):
            sheet.model = mock_model
            sheet.sim, sheet.parameters, sheet.name = sheet.model.sim, params, params.name
            sheet._pop, sheet.dist_params = None, OrderedDict()
            sheet.magnification_factor = params.magnification_factor

        with patch.object(SheetWithMagnificationFactor, '__init__', side_effect=set_vals) as mock_parent_call:
            sheet = VisualCorticalUniformSheet3D(mock_model, params)
            mock_parent_call.assert_called_once_with(sheet, mock_model, params)