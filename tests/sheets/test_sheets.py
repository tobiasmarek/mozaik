import mozaik
import numpy as np
from parameters import ParameterSet
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet
from pyNN import space
from collections import OrderedDict
from string import Template
from pyNN.errors import NothingToWriteError
import pyNN.nest
from mozaik.tools.distribution_parametrization import PyNNDistribution
from mozaik.sheets.direct_stimulator import DirectStimulator

import pytest
from unittest.mock import MagicMock
from unittest.mock import patch
from unittest.mock import call



@pytest.fixture
def mock_model():
    model = MagicMock(sim = pyNN.nest)
    model.sim.state = MagicMock(dt = MagicMock())
    
    yield model


@pytest.fixture(scope="module", params=["sheet_0", "sheet_1"])
def params(request):
    yield MozaikExtendedParameterSet(f"SheetsTests/param/{request.param}")




from mozaik.sheets import Sheet

class TestSheet():

    @pytest.fixture
    def init_sheet(self, mock_model, params):
        yield Sheet(mock_model, None, None, params), params # instead of None, parametrize sx, sy


    # __INIT__ 

    def test_init_assertions(self, init_sheet, mock_model):
        sheet, params = init_sheet

        expected_values = np.array([mock_model.sim, mock_model.sim.state.dt, params.name, "", None, None, None, 0])
        actual_values = np.array([sheet.sim, sheet.dt, sheet.name, "", sheet._pop, sheet.size_x, sheet.size_y, sheet.msc])

        assert np.array_equal(expected_values, actual_values)


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

    def test_setup_to_record_list(self, init_sheet):
        sheet, params = init_sheet

        sheet.setup_to_record_list() # fails if recorders are None, has to be {} - a ParameterSet

        if params.recorders == {}:
            assert len(sheet.to_record) == 0
        else:
            assert len(sheet.to_record) == len(sheet.params.recorders)
            # more tests


    # SIZE_IN_DEGREES

    def test_size_in_degrees(self, init_sheet):
        sheet, _ = init_sheet

        with pytest.raises(NotImplementedError):
            sheet.size_in_degrees()


    @pytest.fixture(scope="function", params=[np.array([2,3,15.212,0.5,-2.5]), np.array([2,3,0.5,-2.5])]) # add more weird parameters
    def _pop_mock(self, request):
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
            sheet.pop = _pop_mock
            mock_arti_stim.assert_called_once()
            mock_init_val.assert_called_once()

        assert sheet.pop == _pop_mock and len(sheet._neuron_annotations) == len(_pop_mock.all_cells) # check if neuron annotations contain OrderedDict type

    
    def test_pop_not_set(self, init_sheet):
        sheet, _ = init_sheet

        assert sheet.pop == None

    
    def test_pop_set_again(self, init_sheet, _pop_mock):
        sheet, _ = init_sheet
        sheet.pop = _pop_mock

        with pytest.raises(Exception):
            sheet.pop = "set_value_again"      


    def test_pop_locals(self):
        pass # test return value of pop() - locals()


    @pytest.fixture(scope="function", params=[2, 1, 10, 0])
    def _pop_mock_and_neuron_annotations(self, request, _pop_mock):
        width = request.param
        
        _neuron_annotations = []
        for i in range(len(_pop_mock)):
            ord_dict = OrderedDict()
            
            for j in range(width):
                ord_dict[f"key_{i},{j}"] = (False, f"ann_{i},{j}")
            
            _neuron_annotations.append(ord_dict)
        
        if width == 0:
            print(_neuron_annotations)
        
        yield _pop_mock, _neuron_annotations


    # ADD_NEURON_ANNOTATION

    @pytest.mark.parametrize("neuron_number,key,value,protected", [(1,"annotation_name", "annotation", False), (2,"annotation_name", "annotation", True)])
    def test_add_neuron_annotation_assign(self, init_sheet, _pop_mock_and_neuron_annotations, neuron_number, key, value, protected):
        sheet, _ = init_sheet
        sheet.pop, sheet._neuron_annotations = _pop_mock_and_neuron_annotations

        sheet.add_neuron_annotation(neuron_number, key, value, protected) # if neuron_number or key is missing -> KeyError

        assert sheet._neuron_annotations[neuron_number][key] == (protected, value) # or logger Annotation protected


    def test_add_neuron_annotation_pop_not_set(self, init_sheet):
        sheet, _ = init_sheet
        pass
        # assert sheet.add_neuron_annotation(neuron_number, key, value, protected) # logger Pop not have been set yet THIS FAILS


    @pytest.fixture(params=["normal", "upper_bound", "lower_bound"]) #, "protected", "missing_key", "missing_neuron"]) # which scope? parametrize instead of fixture?
    def sheet_neuron_num_key_result(self, request, init_sheet, _pop_mock_and_neuron_annotations):
        sheet, _ = init_sheet
        _pop_mock, _neuron_annotations = _pop_mock_and_neuron_annotations
        sheet.pop, sheet._neuron_annotations = _pop_mock, _neuron_annotations

        i = len(_neuron_annotations)//2
        key = "_"

        match request.param:
            case "normal":
                keys = list(_neuron_annotations[i].keys())
                if len(keys) != 0:
                    key = keys[0]
                yield sheet, i, key, _neuron_annotations[i][key][1]
            case "upper_bound":
                i = len(_neuron_annotations) - 1
                keys = list(_neuron_annotations[i].keys())
                if len(keys) != 0:
                    key = keys[0]
                yield sheet, i, key, _neuron_annotations[i][key][1]
            case "lower_bound":
                i = 0
                keys = list(_neuron_annotations[i].keys())
                if len(keys) != 0:
                    key = keys[0]
                yield sheet, i, key, _neuron_annotations[i][key][1]
            case "protected":
                keys = list(_neuron_annotations[i].keys())
                if len(keys) != 0:
                    key = keys[0]
                _neuron_annotations[i][key] = (True, _neuron_annotations[i][key][1])
                yield sheet, i, key, "msg_protected" # how
            case "missing_key":
                yield sheet, i, "non-existent_key", "msg_does_not_exist" 
            case "missing_neuron":
                i = len(_neuron_annotations) # todo
                yield sheet, i, "key", "msg_out_of_range"


    # GET_NEURON_ANNOTATION

    def test_get_neuron_annotation_value(self, sheet_neuron_num_key_result):
        sheet, neuron_number, key, result = sheet_neuron_num_key_result

        assert sheet.get_neuron_annotation(neuron_number, key) == result # check error msg logger smh

    
    def test_get_neuron_annotation_pop_not_set(self, init_sheet):
        sheet, _ = init_sheet
        pass
        # assert sheet.get_neuron_annotation(neuron_number, key) # logger Pop have not been set yet, but i get return value None or check error


    # GET_NEURON_ANNOTATIONS

    def test_get_neuron_annotations_list(self, init_sheet, _pop_mock_and_neuron_annotations):
        sheet, _ = init_sheet
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
        pass
        # assert sheet.get_neuron_annotations() # check logger.error


    # DESCRIBE

    @pytest.mark.parametrize("template,render,result", [('default', lambda t, c: Template(t).safe_substitute(c), None), (None, lambda t, c: c, {'name': 'Sheet'})])
    def test_describe(self, init_sheet, template, render, result): # test more than just Sheet class
        sheet, _ = init_sheet

        assert sheet.describe(template, render) == result


    # RECORD
    
    @pytest.mark.parametrize("all_keys, not_all_keys", [(['key_0', 'key_1', 'key_3'], ['key_2', 'key_4']), ([], []), ([], ['key_0']), (['key_0'], [])])
    def test_record(self, init_sheet, _pop_mock, all_keys, not_all_keys):
        sheet, params = init_sheet
        sheet.pop = _pop_mock    
        
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

    @pytest.mark.parametrize("stimulus_duration", [None, 1, 4.2, 0, -1, 0.00001])
    def test_get_data_assertions(self, init_sheet, _pop_mock, stimulus_duration):
        sheet, params = init_sheet
        
        segment = MagicMock(annotations = OrderedDict(), spiketrains = [], analogsignals = [])
        block_mock = MagicMock(segments = [segment])
        _pop_mock.get_data = block_mock

        sheet.pop = _pop_mock

        new_segment = sheet.get_data(stimulus_duration)
        print(new_segment.annotations["sheet_name"])

        assert sheet.msc >= 0 or (len(segment.spiketrains) == 0 and np.isnan(sheet.msc)) # can it be 0?
        # assert new_segment.annotations["sheet_name"] == params['name']
        # assert  new_segment... == segment... / params... / ...

    
    def test_get_data_call(self, init_sheet, _pop_mock):
        sheet, _ = init_sheet
        sheet.pop = _pop_mock
        
        with patch.object(_pop_mock, 'get_data') as mock_get_data:
            sheet.get_data()
            mock_get_data.assert_called_once_with(['spikes', 'v', 'gsyn_exc', 'gsyn_inh'], clear=True)


    def test_get_data_pop_not_set(self, init_sheet):
        sheet, _ = init_sheet

        with pytest.raises(Exception): # be more specific because -> NameError: name 'errmsg' is not defined
            sheet.get_data()
    

    def test_get_data_nothing_to_write_error(self, init_sheet, _pop_mock):
        sheet, _ = init_sheet

        _pop_mock.get_data = MagicMock(side_effect = NothingToWriteError("error_msg"))
        sheet.pop = _pop_mock
        logger = MagicMock(degub = None)

        with patch.object(_pop_mock, 'get_data'), patch.object(logger, 'debug') as mock_debug:
            sheet.get_data()
            mock_debug.assert_called_once_with("error_msg")


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

        ds = MagicMock()
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
        sheet.pop = _pop_mock

        with patch.object(_pop_mock, 'initialize') as mock_pop_init, patch.object(_pop_mock, 'set') as mock_pop_set:
            sheet.setup_initial_values()
            mock_pop_init.assert_called_once_with(**params.cell.initial_values)
            mock_pop_set.assert_called_once_with(**sheet.dist_params)


    def test_setup_initial_values_pine(self, init_sheet):
        pass # pyNN population with initialize() and set()




from mozaik.sheets.vision import RetinalUniformSheet

class TestRetinalUniformSheet:

    @pytest.fixture(params=[(11.2, 20.1, 3), (14.0, 5.5, 6)])
    def _params(self, request, params):
        args = request.param
        params['sx'], params['sy'], params['density'] = args[0], args[1], args[2]

        yield params


    @pytest.fixture
    def init_sheet(self, mock_model, _params):
        yield RetinalUniformSheet(mock_model, _params), _params


    # __INIT__

    def test_init_assertions(self, init_sheet):
        sheet, params = init_sheet
        
        assert isinstance(sheet.pop, pyNN.nest.Population)
        # assert sheet.pop... == int(parameters.sx * parameters.sy * parameters.density)
        # assert sheet.pop... == getattr(sheet.model.sim, params.cell.model)
        # assert sheet.pop...params? == params.cell.params
        assert isinstance(sheet.pop.structure, space.RandomStructure)
        # assert sheet.pop.structure.boundary == space.Cuboid(params.sx, params.sy, 0)
        # assert sheet.pop.structure... ==
        # assert sheet.pop.initial_values == params.cell.initial_values
        assert sheet.pop.label == params.name

        # test if self.pop.positions was called


    def test_init_sheet_call(self, mock_model, _params):
        params = _params

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

class TestSheetWithMagnificationFactor:

    @pytest.fixture(params=[(1.2, 2.1, 3), (4., .5, 6)]) # (magnification_factor, sx, sy) DENSITY MISSING!
    def _params(self, request, params):
        args = request.param
        params['magnification_factor'], params['sx'], params['sy'] = args[0], args[1], args[2]
        # params['density'] = 0.5

        yield params


    @pytest.fixture
    def init_sheet(self, mock_model, _params):
        yield SheetWithMagnificationFactor(mock_model, _params), _params


    # __INIT__

    def test_init_assertions(self, init_sheet):
        sheet, params = init_sheet

        assert sheet.magnification_factor == params.magnification_factor


    def test_init_sheet_call(self, mock_model, _params):
        params = _params
        mag = params.magnification_factor

        with patch.object(Sheet, '__init__') as mock_parent_call:
            sheet = SheetWithMagnificationFactor(mock_model, params)
            mock_parent_call.assert_called_once_with(sheet, mock_model, params.sx / mag, params.sy / mag, params)


    # VF_2_CS
    
    @pytest.mark.parametrize("degree_x,degree_y", [(10, 20), (5, 5)]) # Non-positive?
    def test_vf_2_cs(self, init_sheet, degree_x, degree_y):
        sheet, params = init_sheet
        mag = params.magnification_factor
        
        assert sheet.vf_2_cs(degree_x, degree_y) == (mag * degree_x, mag * degree_y)


    # CS_2_VF

    @pytest.mark.parametrize("micro_meters_x,micro_meters_y", [(10, 20), (5, 5)]) # Non-positive?
    def test_cs_2_vf(self, init_sheet, micro_meters_x, micro_meters_y):
        sheet, params = init_sheet
        mag = params.magnification_factor
        
        assert sheet.cd_2_vf(micro_meters_x, micro_meters_y) == (micro_meters_x / mag, micro_meters_y / mag)


    # DVF_2_DCS

    @pytest.mark.parametrize("distance_vf", [1,2,3]) # Non-positive?
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

    @pytest.fixture(params=[(1.0, 10000, 2000, 4), (1.2, 20000, 500, 6)]) # (magnification_factor, sx, sy, density)
    def _params(self, request, params): # fails for smaller values (1.0, 80, 90, 3), (1, 60, 50, 6), (1, 180, 45, 5)])
        args = request.param
        params['magnification_factor'], params['sx'], params['sy'], params['density'] = args[0], args[1], args[2], args[3]

        yield params


    @pytest.fixture
    def init_sheet(self, mock_model, _params):
        yield VisualCorticalUniformSheet(mock_model, _params), _params


    # __INIT__

    def test_init_assertions(self, init_sheet):
        sheet, params = init_sheet
        
        assert isinstance(sheet.pop, pyNN.nest.Population)
        # assert sheet.pop... == int(parameters.sx * parameters.sy * parameters.density)
        # assert sheet.pop... == getattr(sheet.model.sim, params.cell.model)
        # assert sheet.pop...params? == params.cell.params
        # assert isinstance(sheet.pop.structure, space.RandomStructure)
        # assert sheet.pop.structure.boundary == space.Cuboid(params.sx, params.sy, 0)
        # assert sheet.pop.structure... == !!!
        # assert sheet.pop.initial_values == params.cell.initial_values
        # assert sheet.pop.label == params.name

        # test if self.pop.positions was called


    def test_init_sheet_w_mag_factor_call(self, mock_model, _params):
        params = _params
        
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

    @pytest.fixture(params=[(1.0, 10000, 2000, 4, 3, 3), (1.2, 20000, 500, 6, 5, 10)]) # (magnification_factor, sx, sy, density, min_depth, max_depth) What happens if min_depth > max depth
    def _params(self, request, params): # fails for smaller values (1.2, 2.1, 3, 6, 3, 3), (4., .5, 6, 5, 5, 10)]) 
        args = request.param
        params['magnification_factor'], params['sx'], params['sy'], params['density'] = args[0], args[1], args[2], args[3]
        params['min_depth'], params['max_depth'] = args[4], args[5]

        yield params


    @pytest.fixture
    def init_sheet(self, mock_model, _params):
        yield VisualCorticalUniformSheet3D(mock_model, _params), _params


    # __INIT__

    def test_init_assertions(self, init_sheet):
        sheet, params = init_sheet
        
        assert isinstance(sheet.pop, pyNN.nest.Population)
        # assert sheet.pop... == int(parameters.sx * parameters.sy * parameters.density)
        # assert sheet.pop... == getattr(sheet.model.sim, params.cell.model)
        # assert sheet.pop...params? == params.cell.params
        # assert isinstance(sheet.pop.structure, space.RandomStructure)
        # assert sheet.pop.structure.boundary == space.Cuboid(params.sx, params.sy, 0)
        # assert sheet.pop.structure... == !!!
        # assert sheet.pop.initial_values == params.cell.initial_values
        # assert sheet.pop.label == params.name

        # test if self.pop.positions was called


    def test_init_sheet_w_mag_factor_call(self, mock_model, _params):
        params = _params
        
        def set_vals(sheet, mock_model, params):
            sheet.model = mock_model
            sheet.sim, sheet.parameters, sheet.name = sheet.model.sim, params, params.name
            sheet._pop, sheet.dist_params = None, OrderedDict()
            sheet.magnification_factor = params.magnification_factor

        with patch.object(SheetWithMagnificationFactor, '__init__', side_effect=set_vals) as mock_parent_call:
            sheet = VisualCorticalUniformSheet3D(mock_model, params)
            mock_parent_call.assert_called_once_with(sheet, mock_model, params)
