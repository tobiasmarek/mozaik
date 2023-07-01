import mozaik
from parameters import ParameterSet
from pyNN import space
from mozaik.sheets import Sheet

import pytest
import numpy as np
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet
import pyNN.nest as sim
from tests.sheets.SheetsTests.model import ModelForSheets

from unittest.mock import MagicMock
from unittest.mock import patch
from collections import OrderedDict


class TestSheet():

    @pytest.fixture(scope="module", params=["sheet_0", "sheet_1"])
    def init_sheet(self, request):
        params = MozaikExtendedParameterSet("SheetsTests/param/defaults")
        params.sheets.sheet = f'url("{request.param}")'

        model = ModelForSheets(sim, 1, params) # prvni na chybu a vyzkoušet více než jen 1
        sheet = model.sheets.sheet

        yield sheet, params


    def test_init(self, init_sheet):
        sheet, params = init_sheet

        # check ze byla zavolana init BaseComponent se spravnyma parametrama např num_of_threads = 1
                                                                   
        expected_values = np.array([sim, sim.state.dt, params.sheets.sheet.params.name, sheet, None, None, None, 0]) # size_x a size_y co budou?
                                                                # tohle je weird.. netestuju přece nic z modelu
        actual_values = np.array([sheet.sim, sheet.dt, sheet.name, "model.sheets[sheet.name]", sheet._pop, sheet.size_x, sheet.size_y, sheet.msc])

        pytest.assert_array_equal(expected_values, actual_values)


    def test_setup_to_record_list(self, init_sheet):
        sheet, _ = init_sheet

        sheet.setup_to_record_list()

        if sheet.params.recorders == None or sheet.params.recorders == {}:
            assert len(sheet.to_record) == 0
        else:
            assert len(sheet.to_record) == len(sheet.params.recorders)
            # jestě něco


    def test_size_in_degrees(self, init_sheet):
        sheet, _ = init_sheet

        with pytest.raises(NotImplementedError):
            sheet.size_in_degrees()


    @pytest.mark.parametrize("all_cells", [np.array([2,3,15.212,0.5,-2.5]), np.array([2,3,15.212,0.5,-2.5])])
    def test_pop(self, init_sheet, all_cells):
        sheet, _ = init_sheet

        assert sheet.pop == None

        _pop_mock = MagicMock()
        _pop_mock.all_cells = all_cells

        with patch.object(sheet, 'setup_artificial_stimulation') as mock_arti_stim, patch.object(sheet, 'setup_initial_values') as mock_init_val:
            sheet.pop = _pop_mock
            mock_arti_stim.assert_called_once()
            mock_init_val.assert_called_once()
            # je rozdil mezi sheet.pop a sheet._pop z vnějšku?
        assert sheet.pop == _pop_mock and len(sheet._neuron_annotations) == len(_pop_mock.all_cells) # check jestli to v neuron annotations je OrderedDict

        with pytest.raises(Exception):
            sheet.pop = "set_value_again"

        # jestě smh testovat return value pop() locals()


    @pytest.mark.parametrize("neuron_number,key,value,protected", [(1,"annotation_name", "annotation", False), (2,"annotation_name", "annotation", True)])
    def test_add_neuron_annotation(self, init_sheet, neuron_number, key, value, protected): # mel bych dodat jeste parametr 'result'
        sheet, _ = init_sheet

        sheet.add_neuron_annotation(neuron_number, key, value, protected) # logger Pop not have been set yet

        _pop_mock = MagicMock()
        _pop_mock.all_cells = np.array([-1,0,42])
        sheet.pop = _pop_mock

        sheet._neuron_annotations = [OrderedDict([('key0', (False, 'ann0'))]),
                                    OrderedDict([('key0', (False, 'ann0')), ('key1', (False, 'ann1'))]),
                                    OrderedDict([('key0', (True, 'ann0'))])]


        sheet.add_neuron_annotation(neuron_number, key, value, protected) # pokud tam neuron_number není, nebo není key, tak to spadne (KeyError)?

        assert sheet._neuron_annotations[neuron_number][key] == (protected, value) # or logger Annotation protected


    @pytest.mark.parametrize("neuron_number,key,result", [(0, 'key0', 'ann0'), (1, 'key1', 'ann1'), (2, 'key3', "logger msg"), (5, 'key0', "OutOfRangeError")]) #?
    def test_get_neuron_annotation(self, init_sheet, neuron_number, key, result):
        sheet, _ = init_sheet

        sheet.add_neuron_annotation(neuron_number, key) # logger Pop not have been set yet, ale dostanu return value None nebo muzu tky error

        _pop_mock = MagicMock()
        _pop_mock.all_cells = np.array([-1,0,42])
        sheet.pop = _pop_mock

        sheet._neuron_annotations = [OrderedDict([('key0', (False, 'ann0'))]),
                                    OrderedDict([('key0', (False, 'ann0')), ('key1', (False, 'ann1'))]),
                                    OrderedDict([('key0', (True, 'ann0'))])]

        assert sheet.get_neuron_annotation(neuron_number, key) == result


    def test_get_neuron_annotations(self, init_sheet):
        sheet, _ = init_sheet

        sheet.get_neuron_annotations() # a měl bych dostat od logger.error, nebo taky navratovou hodnotu []? (nebo logger.error mě raisne)

        _pop_mock = MagicMock()
        _pop_mock.all_cells = np.array([-1,0,42])
        sheet.pop = _pop_mock

        _neuron_annotations = [OrderedDict([('key0', (False, 'ann0'))]),        # CO FIXTURE _NEURON_ANNOTATIONS
                                OrderedDict([('key0', (False, 'ann0')), ('key1', (False, 'ann1'))]),
                                OrderedDict([('key0', (True, 'ann0'))])]
        
        sheet._neuron_annotations = _neuron_annotations
        
        assert sheet.get_neuron_annotations() == _neuron_annotations


    @pytest.mark.parametrize("template,render", [('default', lambda t, c: f"{t}_{c}")]) #lambda t, c: Template(t).safe_substitute(c)), ()]) # from string import Template
    def test_describe(self, init_sheet, template, render):
        sheet, _ = init_sheet

        sheet.describe(template) == "Sheet" ## ale co s renderem


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


    def test_mean_spike_count(self, init_sheet):
        sheet, _ = init_sheet

        assert sheet.mean_spike_count() == sheet.msc


    @pytest.mark.parametrize("duration,offset,additional_stimulators", [(0.6, 0.4, [])])
    def test_prepare_artificial_stimulation(self, init_sheet, duration, offset, additional_stimulators):
        sheet, _ = init_sheet

        # ds by měl být stimulator, který obsahuje additional_stimulator třeba
        with patch.object(ds, prepare_stimulation) as mock_prep_stim:
            sheet.prepare_artificial_stimulation(duration, offset, additional_stimulators)
            mock_prep_stim.assert_called_with(duration, offset) # nemusí ale vůbec být called, pokud nejsou stimulators
            # checknout kolikrát for loop běžel len(additional_stimulators)+len(sheet.artificial_stimulators)


    def test_setup_artificial_stimulation(self, init_sheet):
        sheet, params = init_sheet

        sheet.setup_artificial_stimulation()

        assert len(sheet.artificial_stimulators) == len(params.sheets.sheet.params.artificial_stimulators) # ještě něco


    def test_setup_initial_values(self, init_sheet):
        sheet, params = init_sheet

        sheet.setup_initial_values() # pyNN populace s initialize a set

        pass # zkontrolovat jestli odpovídají tomu co je v args




class TestRetinalUniformSheet:

    @pytest.fixture(params=[(11.2, 20.1, 3), (14., 5.5, 6)]) # (sx, sy, density)
    def init_sheet(self, request):
        args = request.param
    
        params = MozaikExtendedParameterSet("SheetsTests/param/defaults")
        sh_params = params.sheets.sheet.params
        sh_params['sx'], sh_params['sy'], sh_params['density'] = args[0], args[1], args[2]
        
        model = ModelForSheets(sim, 1, params)
        sheet = model.sheets.sheet

        return sheet, args

    
    @pytest.mark.parametrize("sx,sy,density", [(11.2, 20.1, 3), (14.0, 5.5, 6)]) # (sx, sy, density)
    def test_init(self, sx, sy, density):
        params = MozaikExtendedParameterSet("SheetsTests/param/defaults")
        sh_params = params.sheets.sheet.params
        sh_params['sx'], sh_params['sy'], sh_params['density'] = sx, sy, density # rekl bych ze je problem s tim, ze v TestSheet nezkousim i jiny parametry jako sx,sy..
        
        with patch.object(Sheet, '__init__') as mock_init:
            model = ModelForSheets(sim, 1, params)
            sheet = model.sheets.sheet
            mock_init.assert_called_once()
        
        # otestovat ze se assignlo správně self.pop

        # nejak otestovat ze se volalo self.pop.positions


    def test_size_in_degrees(self, init_sheet):
        sheet, args = init_sheet

        assert sheet.test_size_in_degrees() == (args[0], args[1])




class TestSheetWithMagnificationFactor:

    @pytest.fixture(params=[(1.2, 2.1, 3), (4., .5, 6)]) # (magnification_factor, sx, sy)
    def init_sheet(self, request):
        args = request.param
        params = MozaikExtendedParameterSet("SheetsTests/param/defaults")  
        model = ModelForSheets(sim, 1, params)
        sheet = model.sheets.sheet

        return sheet, args


    def test_init(self, init_sheet):
        sheet, args = init_sheet

        expected_values, actual_values = np.array([]), np.array([])

        pytest.assert_array_equal(expected_values, actual_values)

    
    @pytest.mark.parametrize("degree_x,degree_y", [(10, 20), (5, 5)])
    def test_vf_2_cs(self, init_sheet, degree_x, degree_y):
        sheet, args = init_sheet
        mag = args[0]
        
        assert sheet.vf_2_cs(degree_x, degree_y) == (mag * degree_x, mag * degree_y)


    @pytest.mark.parametrize("micro_meters_x,micro_meters_y", [(10, 20), (5, 5)])
    def test_cs_2_vf(self, init_sheet, micro_meters_x, micro_meters_y):
        sheet, args = init_sheet
        mag = args[0]
        
        assert sheet.cd_2_vf(micro_meters_x, micro_meters_y) == (micro_meters_x / mag, micro_meters_y / mag)


    @pytest.mark.parametrize("distance_vf", [1,2,3])
    def test_dvf_2_dcs(self, init_sheet, distance_vf):
        sheet, args = init_sheet

        assert sheet.dvf_2_dcs(distance_vf) == distance_vf * args[0]


    def test_size_in_degrees(self, init_sheet):
        sheet, params = init_sheet
        sh_params = params.sheets.sheet.params
 
        assert sheet.size_in_degrees() == (sh_params.sx / sh_params.magnification_factor, sh_params.sy / sh_params.magnification_factor)




class TestVisualCorticalUniformSheet:

    @pytest.fixture(params=[5, 10]) # density
    def init_sheet(self, request):
        args = request.param
        params = MozaikExtendedParameterSet("SheetsTests/param/defaults") 
        model = ModelForSheets(sim, 1, params)
        sheet = model.sheets.sheet

        return sheet


    def test_init(self, init_sheet):
        pass




class TestVisualCorticalUniformSheet3D:

    @pytest.fixture(params=[(1.2, 2.1), (4., .5)]) # (min_depth, max_depth)
    def init_sheet(self, request):
        args = request.param
        params = MozaikExtendedParameterSet("SheetsTests/param/defaults")    
        model = ModelForSheets(sim, 1, params)
        sheet = model.sheets.sheet

        return sheet
    

    def test_init(self, init_sheet):
        pass
    
