import mozaik
from parameters import ParameterSet
from pyNN import space

import pytest
import numpy as np
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet

from unittest.mock import MagicMock
from unittest.mock import patch
from collections import OrderedDict

from mozaik.sheets import Sheet

class TestSheet():

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.sim = MagicMock()
        model.sim.state = MagicMock()
        model.sim.state.dt = 0.1
        
        yield model

    
    @pytest.fixture(scope="function", params=["sheet_0", "sheet_1"])
    def params(self, request):
        params = MozaikExtendedParameterSet(f"SheetsTests/param/{request.param}")
        
        yield params


    @pytest.fixture
    def init_sheet(self, mock_model, params):
        sheet = Sheet(mock_model, None, None, params)

        yield sheet, params


    # __INIT__    

    def test_init_assertions(self, init_sheet, mock_model):
        sheet, params = init_sheet

        expected_values = np.array([mock_model.sim, mock_model.sim.state.dt, params.name, "", None, None, None, 0])
        actual_values = np.array([sheet.sim, sheet.dt, sheet.name, "", sheet._pop, sheet.size_x, sheet.size_y, sheet.msc])

        assert np.array_equal(expected_values, actual_values)


    def test_init_base_component_call(self, mock_model, params):
        #todo

        with patch.object(BaseComponent, '__init__') as mock_parent_call, patch.object(mock_model, 'register_sheet') as mock_register_sheet:
            Sheet(mock_model, None, None, params)
            mock_parent_call.assert_called_once_with(mock_model, params)
            mock_register_sheet.assert_called_once()


    def test_init_dist_params(self, init_sheet):
        sheet, params = init_sheet

        for key in params.cell.params: # tadyto samozřejmě není dostatečnej test tyhle funkce
            assert params.cell.params[key] == sheet.parameters.cell.params[key]


    # SETUP_TO_RECORD_LIST

    def test_setup_to_record_list(self, init_sheet):
        sheet, params = init_sheet

        sheet.setup_to_record_list() # padá to pokud jsou recorders prázdný !

        if params.recorders == None or params.recorders == {}:
            assert len(sheet.to_record) == 0
        else:
            assert len(sheet.to_record) == len(sheet.params.recorders)
            # jestě něco


    # SIZE_IN_DEGREES

    def test_size_in_degrees(self, init_sheet):
        sheet, _ = init_sheet

        with pytest.raises(NotImplementedError):
            sheet.size_in_degrees()


    # POP
    # rozdělit na více funkcí?
    @pytest.mark.parametrize("all_cells", [np.array([2,3,15.212,0.5,-2.5]), np.array([2,3,15.212,0.5,-2.5])])
    def test_pop(self, init_sheet, all_cells):
        sheet, _ = init_sheet

        assert sheet.pop == None

        _pop_mock = MagicMock()
        _pop_mock.all_cells = all_cells
        _pop_mock.__len__.return_value = len(all_cells)

        with patch.object(sheet, 'setup_artificial_stimulation') as mock_arti_stim, patch.object(sheet, 'setup_initial_values') as mock_init_val:
            sheet.pop = _pop_mock
            mock_arti_stim.assert_called_once()
            mock_init_val.assert_called_once()
            # je rozdil mezi sheet.pop a sheet._pop z vnějšku?
        assert sheet.pop == _pop_mock and len(sheet._neuron_annotations) == len(_pop_mock.all_cells) # check jestli to v neuron annotations je OrderedDict

        with pytest.raises(Exception):
            sheet.pop = "set_value_again"

        # jestě smh testovat return value pop() locals()


    # ADD_NEURON_ANNOTATION

    @pytest.mark.parametrize("neuron_number,key,value,protected", [(1,"annotation_name", "annotation", False), (2,"annotation_name", "annotation", True)])
    def test_add_neuron_annotation(self, init_sheet, neuron_number, key, value, protected): # mel bych dodat jeste parametr 'result'
        sheet, _ = init_sheet

        #sheet.add_neuron_annotation(neuron_number, key, value, protected) # logger Pop not have been set yet TOHLE SPADNE

        _pop_mock = MagicMock()
        _pop_mock.all_cells = np.array([-1,0,42])
        sheet.pop = _pop_mock

        sheet._neuron_annotations = [OrderedDict([('key0', (False, 'ann0'))]),
                                    OrderedDict([('key0', (False, 'ann0')), ('key1', (False, 'ann1'))]),
                                    OrderedDict([('key0', (True, 'ann0'))])]


        sheet.add_neuron_annotation(neuron_number, key, value, protected) # pokud tam neuron_number není, nebo není key, tak to spadne (KeyError)?

        assert sheet._neuron_annotations[neuron_number][key] == (protected, value) # or logger Annotation protected


    # GET_NEURON_ANNOTATION

    @pytest.mark.parametrize("neuron_number,key,result", [(0, 'key0', 'ann0'), (1, 'key1', 'ann1'), (2, 'key3', "logger msg"), (5, 'key0', "OutOfRangeError")])
    def test_get_neuron_annotation(self, init_sheet, neuron_number, key, result):
        sheet, _ = init_sheet

        # sheet.get_neuron_annotation(neuron_number, key) # logger Pop not have been set yet, ale dostanu return value None nebo muzu tky error

        _pop_mock = MagicMock()
        _pop_mock.all_cells = np.array([-1,0,42])
        sheet.pop = _pop_mock

        sheet._neuron_annotations = [OrderedDict([('key0', (False, 'ann0'))]),
                                    OrderedDict([('key0', (False, 'ann0')), ('key1', (False, 'ann1'))]),
                                    OrderedDict([('key0', (True, 'ann0'))])]

        assert sheet.get_neuron_annotation(neuron_number, key) == result


    # GET_NEURON_ANNOTATIONS

    def test_get_neuron_annotations(self, init_sheet):
        sheet, _ = init_sheet

        #sheet.get_neuron_annotations() # a měl bych dostat od logger.error, nebo taky navratovou hodnotu []? (nebo logger.error mě raisne)

        _pop_mock = MagicMock()
        _pop_mock.all_cells = np.array([-1,0,42])
        _pop_mock.__len__.return_value = len(_pop_mock.all_cells)
        sheet.pop = _pop_mock

        _neuron_annotations = [OrderedDict([('key0', (False, 'ann0'))]),        # CO FIXTURE _NEURON_ANNOTATIONS
                                OrderedDict([('key0', (False, 'ann0')), ('key1', (False, 'ann1'))]),
                                OrderedDict([('key0', (True, 'ann0'))])]

        returned_neuron_annotations = [OrderedDict([('key0', 'ann0')]),
                                OrderedDict([('key0', 'ann0'), ('key1', 'ann1')]),
                                OrderedDict([('key0', 'ann0')])]
        
        sheet._neuron_annotations = _neuron_annotations
        
        assert sheet.get_neuron_annotations() == returned_neuron_annotations


    # DESCRIBE

    @pytest.mark.parametrize("template,render", [('default', lambda t, c: f"{t}_{c}"), (None, lambda t, c: f"{t}_{c}")]) #lambda t, c: Template(t).safe_substitute(c)), ()]) # from string import Template
    def test_describe(self, init_sheet, template, render):
        sheet, params = init_sheet

        assert sheet.describe(template) == { 'name': 'Sheet' } ## ale co s renderem a taky netestuju vubec kdyz to je neco jinyho nez normal Sheet


    # RECORD

    def test_record(self, init_sheet):
        sheet, _ = init_sheet

        _pop_mock = MagicMock()
        _pop_mock.record = MagicMock()
        _pop_mock.all_cells = np.array([-1,0,42])
        sheet.pop = _pop_mock

        sheet.to_record = OrderedDict([('key0', 'all'), ('key1', 'all'), ('key2', '')])

        with patch.object(sheet, 'setup_to_record_list') as mock_setup_record, patch.object(_pop_mock, 'record') as mock_record:
            sheet.record()
            mock_record.assert_called() # called ale jak, kolikrát a s čím -> a ještě záleží nějak na 'all'
            mock_setup_record.assert_called_once()


    # GET_DATA

    @pytest.mark.parametrize("stimulus_duration", [None, 1, 4.2, 0, -1, 0.00001])
    def test_get_data(self, init_sheet, stimulus_duration):
        sheet, params = init_sheet

        _pop_mock = MagicMock()
        _pop_mock.all_cells = np.array([-1,0,42])
        
        segment = MagicMock()
        segment.annotations = OrderedDict()
        segment.spiketrains, segment.analogsignals = MagicMock(), MagicMock() # tohle by měly být arrays

        block_mock = MagicMock()
        block_mock.segments = [segment]

        _pop_mock.get_data = block_mock

        sheet.pop = _pop_mock

        assert sheet.get_data(stimulus_duration) == segment # ale změněnej


    # MEAN_SPIKE_COUNT

    def test_mean_spike_count(self, init_sheet):
        sheet, _ = init_sheet

        assert sheet.mean_spike_count() == sheet.msc # parametrizovat nějak, má tohle vůbec smysl testovat?


    # PREPARE_ARTIFICIAL_STIMULATION

    @pytest.mark.parametrize("duration,offset,additional_stimulators", [(0.6, 0.4, [])])
    def test_prepare_artificial_stimulation(self, init_sheet, duration, offset, additional_stimulators):
        sheet, _ = init_sheet

        ds = MagicMock()
        sheet.artificial_stimulators = [ds,ds,ds] # mozna dobrý parametrizovat jak pocet kolik jich je

        with patch.object(ds, 'prepare_stimulation') as mock_prep_stim:
            sheet.prepare_artificial_stimulation(duration, offset, additional_stimulators)
            mock_prep_stim.assert_called_with(duration, offset) # nemusí ale vůbec být called, pokud nejsou stimulators
            assert mock_prep_stim.call_count == len(sheet.artificial_stimulators) + len(additional_stimulators)


    # SETUP_ARTIFICIAL_STIMULATION

    def test_setup_artificial_stimulation(self, init_sheet):
        sheet, params = init_sheet

        sheet.setup_artificial_stimulation()

        assert len(sheet.artificial_stimulators) == len(params.artificial_stimulators) # ještě něco


    # SETUP_INITIAL_VALUES

    def test_setup_initial_values(self, init_sheet):
        sheet, params = init_sheet

        # nebo jen mock objekt popu
        sheet.setup_initial_values() # pyNN populace s initialize a set

        pass # zkontrolovat jestli odpovídají tomu co je v args




from mozaik.sheets.vision import RetinalUniformSheet

class TestRetinalUniformSheet:

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.sim = MagicMock()
        
        yield model

    
    @pytest.fixture(scope="function", params=["sheet_0", "sheet_1"])
    def base_params(self, request):
        base_params = MozaikExtendedParameterSet(f"SheetsTests/param/{request.param}")
        
        yield base_params


    @pytest.fixture(params=[(11.2, 20.1, 3), (14.0, 5.5, 6)])
    def params(self, request, base_params):
        args = request.param
        base_params['sx'], base_params['sy'], base_params['density'] = args[0], args[1], args[2]

        yield base_params


    @pytest.fixture
    def init_sheet(self, mock_model, params):
        sheet = RetinalUniformSheet(mock_model, params)

        yield sheet, params


    # INIT

    def test_init_assertions(self, init_sheet):
        # otestovat ze se assignlo správně self.pop

        # nejak otestovat ze se volalo self.pop.positions
        pass


    def test_init_sheet_call(self, mock_model, params):

        with patch.object(Sheet, '__init__') as mock_parent_call:
            sheet = RetinalUniformSheet(mock_model, params)
            mock_parent_call.assert_called_once_with(mock_model, params.sx, params.sy, params)


    # SIZE_IN_DEGREES

    def test_size_in_degrees(self, init_sheet):
        sheet, params = init_sheet

        assert sheet.size_in_degrees() == (params.sx, params.sy)




from mozaik.sheets.vision import SheetWithMagnificationFactor

class TestSheetWithMagnificationFactor:

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.sim = MagicMock()
        
        yield model

    
    @pytest.fixture(scope="function", params=["sheet_0", "sheet_1"])
    def base_params(self, request):
        base_params = MozaikExtendedParameterSet(f"SheetsTests/param/{request.param}")
        
        yield base_params


    @pytest.fixture(params=[(1.2, 2.1, 3), (4., .5, 6)]) # (magnification_factor, sx, sy) CHYBÍ DENSITY!!
    def params(self, request, base_params):
        args = request.param
        base_params['magnification_factor'], base_params['sx'], base_params['sy'] = args[0], args[1], args[2]
        # base_params['density'] = 0.5

        yield base_params


    @pytest.fixture
    def init_sheet(self, mock_model, params):
        sheet = SheetWithMagnificationFactor(mock_model, params)

        yield sheet, params


    # INIT

    def test_init_assertions(self, init_sheet):
        sheet, params = init_sheet

        expected_values, actual_values = np.array([]), np.array([])

        assert np.array_equal(expected_values, actual_values)


    def test_init_sheet_call(self, mock_model, params):
        
        with patch.object(Sheet, '__init__') as mock_parent_call:
            sheet = SheetWithMagnificationFactor(mock_model, params)
            mock_parent_call.assert_called_once_with(mock_model, params.sx, params.sy, params)


    # VF_2_CS
    
    @pytest.mark.parametrize("degree_x,degree_y", [(10, 20), (5, 5)]) # co zápor, nemělo by spadnout?
    def test_vf_2_cs(self, init_sheet, degree_x, degree_y):
        sheet, params = init_sheet
        mag = params.magnification_factor
        
        assert sheet.vf_2_cs(degree_x, degree_y) == (mag * degree_x, mag * degree_y)


    # CS_2_VF

    @pytest.mark.parametrize("micro_meters_x,micro_meters_y", [(10, 20), (5, 5)]) # zápor? nula?
    def test_cs_2_vf(self, init_sheet, micro_meters_x, micro_meters_y):
        sheet, params = init_sheet
        mag = params.magnification_factor
        
        assert sheet.cd_2_vf(micro_meters_x, micro_meters_y) == (micro_meters_x / mag, micro_meters_y / mag)


    # DVF_2_DCS

    @pytest.mark.parametrize("distance_vf", [1,2,3]) # zápor? nula?
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

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.sim = MagicMock()
        
        yield model

    
    @pytest.fixture(scope="function", params=["sheet_0", "sheet_1"])
    def base_params(self, request):
        base_params = MozaikExtendedParameterSet(f"SheetsTests/param/{request.param}")
        
        yield base_params


    @pytest.fixture(params=[(1.2, 2.1, 3, 6), (4., .5, 6, 5)]) # (magnification_factor, sx, sy, density)
    def params(self, request, base_params):
        args = request.param
        base_params['magnification_factor'], base_params['sx'], base_params['sy'], base_params['density'] = args[0], args[1], args[2], args[3]

        yield base_params


    @pytest.fixture
    def init_sheet(self, mock_model, params):
        sheet = VisualCorticalUniformSheet(mock_model, params)

        yield sheet, params

    # INIT

    def test_init_assertions(self, init_sheet):
        pass


    def test_init_sheet_w_mag_factor_call(self, mock_model, params):
        
        with patch.object(SheetWithMagnificationFactor, '__init__') as mock_parent_call:
            sheet = VisualCorticalUniformSheet(mock_model, params)
            mock_parent_call.assert_called_once_with(mock_model, params.sx, params.sy, params)




from mozaik.sheets.vision import VisualCorticalUniformSheet3D

class TestVisualCorticalUniformSheet3D:

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.sim = MagicMock()
        
        yield model

    
    @pytest.fixture(scope="function", params=["sheet_0", "sheet_1"])
    def base_params(self, request):
        base_params = MozaikExtendedParameterSet(f"SheetsTests/param/{request.param}")
        
        yield base_params


    @pytest.fixture(params=[(1.2, 2.1, 3, 6), (4., .5, 6, 5)]) # (magnification_factor, sx, sy, density)
    def params(self, request, base_params):
        args = request.param
        base_params['magnification_factor'], base_params['sx'], base_params['sy'], base_params['density'] = args[0], args[1], args[2], args[3]

        yield base_params


    @pytest.fixture
    def init_sheet(self, mock_model, params):
        sheet = VisualCorticalUniformSheet3D(mock_model, params)

        yield sheet, params


    # INIT

    def test_init_assertions(self, init_sheet): # specific pro 3D
        pass


    def test_init_sheet_w_mag_factor_call(self, mock_model, params):
        
        with patch.object(SheetWithMagnificationFactor, '__init__') as mock_parent_call:
            sheet = VisualCorticalUniformSheet3D(mock_model, params)
            mock_parent_call.assert_called_once_with(mock_model, params.sx, params.sy, params)
    
