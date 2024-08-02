import mozaik
import numpy as np
from mozaik.core import ParametrizedObject
from parameters import ParameterSet
from mozaik import setup_mpi
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet
from mozaik.tools.circ_stat import circular_dist

import pytest
from unittest.mock import MagicMock
from unittest.mock import patch
from unittest.mock import call



@pytest.fixture(scope="module", params=[(1, 0.1), (11, 0.15), (42, 0.05), (100, 0.5)]) # (num_of_cells, magnification_factor)
def mock_sheet(request):
    """
    Mocking the Sheet class to separate the testing of PopulationSelector and the Sheet itself.
    Scope being equal to module means creating the mocked object only once per module.
    """
    sheet = MagicMock()

    num_of_cells, sheet.magnification_factor = request.param # what if its not a sheet with magnification_factor?

    sheet.pop.all_cells = np.array([i for i in range(0, num_of_cells)], dtype=object)
    sheet.pop.positions = np.array([np.linspace(0, 1, len(sheet.pop.all_cells)), np.linspace(0, 1, len(sheet.pop.all_cells))]) # better real pos?

    sheet.cs_2_vf = MagicMock(side_effect=lambda x, y: (x / sheet.magnification_factor, y / sheet.magnification_factor))
    sheet.vf_2_cs = MagicMock(side_effect=lambda x, y: (x * sheet.magnification_factor, y * sheet.magnification_factor))
    
    _neuron_annotations = [i for i in range(len(sheet.pop.all_cells))] # does not let me to have an object there (namely str f"ann_{i},0")
    sheet.get_neuron_annotation = MagicMock(side_effect=lambda i, ann: _neuron_annotations[i])

    yield sheet




from mozaik.sheets.population_selector import PopulationSelector

class TestPopulationSelector:

    @pytest.fixture(scope="class")
    def init_pop_selector(self, mock_sheet):
        """A fixture for the initialization of the PopulationSelector object, created once"""
        params = ParameterSet({})

        yield PopulationSelector(mock_sheet, params), params


    # __INIT__

    def test_init_assertions(self, init_pop_selector, mock_sheet):
        pop_sel, _ = init_pop_selector

        assert pop_sel.sheet == mock_sheet
        assert all(pop_sel.z == mock_sheet.pop.all_cells.astype(int))


    def test_init_parametrized_object_call(self, mock_sheet):
        params = ParameterSet({})

        with patch.object(ParametrizedObject, '__init__') as mock_parent_call:
            pop_sel = PopulationSelector(mock_sheet, params)

            mock_parent_call.assert_called_once_with(pop_sel, params)


    # GENERATE_IDD_LIST_OF_NEURONS

    def test_generate_idd_list_of_neurons(self, init_pop_selector):
        pop_sel, _ = init_pop_selector
        
        with pytest.raises(NotImplementedError):
            pop_sel.generate_idd_list_of_neurons()




from mozaik.sheets.population_selector import RCAll

class TestRCAll:

    @pytest.fixture(scope="class")
    def init_pop_selector(self, mock_sheet):
        params = ParameterSet({})

        yield RCAll(mock_sheet, params), params


    # GENERATE_IDD_LIST_OF_NEURONS

    def test_generate_idd_list_of_neurons(self, init_pop_selector):
        pop_sel, _ = init_pop_selector
        z = pop_sel.sheet.pop.all_cells.astype(int)
        sel_len = len(z)

        selected_pop = pop_sel.generate_idd_list_of_neurons()

        assert len(selected_pop) == sel_len
        assert len(selected_pop) == len(list(set(selected_pop))) # if unique
        assert all([id in z for id in selected_pop]) # if in the original population




from mozaik.sheets.population_selector import RCRandomN

class TestRCRandomN:
    
    @pytest.fixture(scope="class", params=[1000, 311, 1, 0, -1, -200])
    def init_pop_selector(self, request, mock_sheet):
        """A fixture for the initialization of the PopulationSelector object, created once"""
        params = ParameterSet({"num_of_cells": request.param})

        yield RCRandomN(mock_sheet, params), params


    # __INIT__

    def test_init_assertions(self, init_pop_selector):
        pop_sel, params = init_pop_selector
        
        assert params.num_of_cells == pop_sel.parameters.num_of_cells # int


    # GENERATE_IDD_LIST_OF_NEURONS

    def test_generate_idd_list_of_neurons(self, init_pop_selector):
        pop_sel, _ = init_pop_selector
        sel_len = pop_sel.parameters.num_of_cells
        z = pop_sel.sheet.pop.all_cells.astype(int)
        
        setup_mpi(mozaik_seed=513,pynn_seed=1023)
        selected_pop = pop_sel.generate_idd_list_of_neurons()

        if sel_len <= len(z): # if n <= population size
            if sel_len >= 0: 
                assert len(selected_pop) == sel_len
            else:
                assert len(selected_pop) == max(0, len(z) + sel_len)
        else:
            assert len(selected_pop) == len(z)

        if sel_len > 0 or (sel_len < 0 and max(0, len(z) + sel_len) != 0): # if at least one neuron is selected
            if len(selected_pop) > 1: # if able to shuffle
                assert (selected_pop != z[:sel_len]).any() # if shuffled # what if identity
        else:
            assert np.array_equal(selected_pop, np.array([]).astype(int))
        
        assert len(selected_pop) == len(list(set(selected_pop))) # if unique
        assert all([id in z for id in selected_pop]) # if in the original population




from mozaik.sheets.population_selector import RCRandomPercentage

class TestRCRandomPercentage:   
    
    @pytest.fixture(scope="class", params=[1, 11, 20.55, 100, 100.1, 150, 0, -1, -200])
    def init_pop_selector(self, request, mock_sheet):
        params = ParameterSet({"percentage": request.param})

        yield RCRandomPercentage(mock_sheet, params), params


    # __INIT__

    def test_init_assertions(self, init_pop_selector):
        pop_sel, params = init_pop_selector
        
        assert params.percentage == pop_sel.parameters.percentage # float


    # GENERATE_IDD_LIST_OF_NEURONS

    def test_generate_idd_list_of_neurons(self, init_pop_selector):
        pop_sel, _ = init_pop_selector
        z = pop_sel.sheet.pop.all_cells.astype(int)
        sel_len = int(len(z) * pop_sel.parameters.percentage/100)
        
        setup_mpi(mozaik_seed=513,pynn_seed=1023)
        selected_pop = pop_sel.generate_idd_list_of_neurons()

        if pop_sel.parameters.percentage <= 100: # if percentage <= 100
            if sel_len >= 0:
                assert len(selected_pop) == sel_len
            else:
                assert len(selected_pop) == max(0, len(z) + sel_len)
        else:
            assert len(selected_pop) == len(z)

        if sel_len > 0 or (sel_len < 0 and max(0, len(z) + sel_len) != 0): # if at least one neuron is selected
            if len(selected_pop) > 1: # if able to shuffle
                assert (selected_pop != z[:sel_len]).any() # if shuffled # what if identity
        else:
            assert np.array_equal(selected_pop, np.array([]).astype(int))

        assert len(selected_pop) == len(list(set(selected_pop))) # if unique
        assert all([id in z for id in selected_pop]) # if in the original population



from mozaik.sheets.population_selector import RCGrid

class TestRCGrid:

    @pytest.fixture(scope="class", params=["rc_grid_pop_sel"])
    def params(self, request):
        yield MozaikExtendedParameterSet(f"tests/sheets/PopulationSelectorTests/param/{request.param}")
    
    
    @pytest.fixture(scope="class")
    def init_pop_selector(self, mock_sheet, params):
        yield RCGrid(mock_sheet, params), params


    # __INIT__

    def test_init_assertions(self, init_pop_selector):
        pop_sel, params = init_pop_selector
        
        assert params.size == pop_sel.parameters.size # float
        assert params.spacing == pop_sel.parameters.spacing  # float
        assert params.offset_x == pop_sel.parameters.offset_x  # float
        assert params.offset_y == pop_sel.parameters.offset_y # float


    # GENERATE_IDD_LIST_OF_NEURONS

    def test_generate_idd_list_of_neurons(self, init_pop_selector):
        pop_sel, _ = init_pop_selector
        max_sel_len = (pop_sel.parameters.size / pop_sel.parameters.spacing)**2 # number of electrodes
        centered_electrodes = np.arange(0, pop_sel.parameters.size, pop_sel.parameters.spacing) - pop_sel.parameters.size/2.0
        xx = [x / pop_sel.sheet.magnification_factor for x in pop_sel.parameters.offset_x + centered_electrodes] # what if not sheet w mag factor
        yy = [y / pop_sel.sheet.magnification_factor for y in pop_sel.parameters.offset_y + centered_electrodes] # what if not sheet w mag factor

        selected_pop = pop_sel.generate_idd_list_of_neurons()
        
        assert len(selected_pop) <= max_sel_len
        assert all(pop_sel.z[np.argmin((pop_sel.sheet.pop.positions[0] - x)**2 +  (pop_sel.sheet.pop.positions[1] - y)**2)] in selected_pop for x in xx for y in yy)


    @pytest.mark.parametrize("size,spacing", [(3000, 2000), (1, 2), (11, 0.003)])
    def test_generate_idd_size_not_multiple_of_spacing(self, init_pop_selector, size, spacing):
        pop_sel, _ = init_pop_selector
        pop_sel.parameters.size, pop_sel.parameters.spacing = size, spacing
        
        with pytest.raises(AssertionError):
            pop_sel.generate_idd_list_of_neurons()




from mozaik.sheets.population_selector import RCGridDegree

class TestRCGridDegree:

    @pytest.fixture(scope="class", params=["rc_grid_degree_pop_sel"])
    def params(self, request):
        yield MozaikExtendedParameterSet(f"tests/sheets/PopulationSelectorTests/param/{request.param}")
    
    
    @pytest.fixture(scope="class")
    def init_pop_selector(self, mock_sheet, params):
        yield RCGridDegree(mock_sheet, params), params


    # __INIT__

    def test_init_assertions(self, init_pop_selector):
        pop_sel, params = init_pop_selector
        
        assert params.size == pop_sel.parameters.size # float
        assert params.spacing == pop_sel.parameters.spacing  # float
        assert params.offset_x == pop_sel.parameters.offset_x  # float
        assert params.offset_y == pop_sel.parameters.offset_y # float


    # GENERATE_IDD_LIST_OF_NEURONS

    def test_generate_idd_list_of_neurons(self, init_pop_selector):
        pop_sel, _ = init_pop_selector
        max_sel_len = (pop_sel.parameters.size / pop_sel.parameters.spacing)**2 # number of electrodes
        centered_electrodes = np.arange(0, pop_sel.parameters.size, pop_sel.parameters.spacing) - pop_sel.parameters.size/2.0
        xx = [x for x in pop_sel.parameters.offset_x + centered_electrodes]
        yy = [y for y in pop_sel.parameters.offset_y + centered_electrodes]

        selected_pop = pop_sel.generate_idd_list_of_neurons()
        
        assert len(selected_pop) <= max_sel_len
        assert all(pop_sel.z[np.argmin((pop_sel.sheet.pop.positions[0] - x)**2 +  (pop_sel.sheet.pop.positions[1] - y)**2)]\
            in selected_pop for x in xx for y in yy)


    @pytest.mark.parametrize("size,spacing", [(3000, 2000), (1, 2), (11, 0.003)])
    def test_generate_idd_size_not_multiple_of_spacing(self, init_pop_selector, size, spacing):
        pop_sel, _ = init_pop_selector
        pop_sel.parameters.size, pop_sel.parameters.spacing = size, spacing
        
        with pytest.raises(AssertionError):
            pop_sel.generate_idd_list_of_neurons()




from mozaik.sheets.population_selector import SimilarAnnotationSelector

class TestSimilarAnnotationSelector:

    @pytest.fixture(scope="class", params=["similar_annotation_selector"])
    def params(self, request):
        yield MozaikExtendedParameterSet(f"tests/sheets/PopulationSelectorTests/param/{request.param}")
    
    
    @pytest.fixture(scope="class", params=[1000, 311, 1, 0, -1, -200])
    def init_pop_selector(self, request, mock_sheet, params):
        params.num_of_cells = request.param

        yield SimilarAnnotationSelector(mock_sheet, params), params


    # __INIT__

    def test_init_assertions(self, init_pop_selector):
        pop_sel, params = init_pop_selector
        
        assert params.annotation == pop_sel.parameters.annotation  # str
        assert params.distance == pop_sel.parameters.distance  # float
        assert params.value == pop_sel.parameters.value  # float
        assert params.num_of_cells == pop_sel.parameters.num_of_cells  # int
        assert params.period == pop_sel.parameters.period  # float


    # PICK_CLOSE_TO_ANNOTATION

    def test_pick_close_to_annotation_zero_period(self, init_pop_selector):
        pop_sel, _ = init_pop_selector
        vals = [pop_sel.sheet.get_neuron_annotation(i, pop_sel.parameters.annotation) for i in range(len(pop_sel.z))]
        pop_sel.parameters.period = 0

        picked = pop_sel.pick_close_to_annotation()

        assert len(picked) <= len(pop_sel.z)
        assert len(picked) == len(list(set(picked))) # if unique
        assert all([id in np.arange(0, len(pop_sel.z)) for id in picked]) # if in the original population
        assert all([abs(vals[id] - pop_sel.parameters.value) <= pop_sel.parameters.distance for id in picked]) # check distance restriction with abs


    @pytest.mark.parametrize("period", [2.0, 4.2, 200, -1])
    def test_pick_close_to_annotation_non_zero_period(self, init_pop_selector, period):
        pop_sel, _ = init_pop_selector
        vals = [pop_sel.sheet.get_neuron_annotation(i, pop_sel.parameters.annotation) for i in range(len(pop_sel.z))]
        pop_sel.parameters.period = period

        picked = pop_sel.pick_close_to_annotation()

        assert len(picked) <= len(pop_sel.z)
        assert len(picked) == len(list(set(picked))) # if unique
        assert all([id in np.arange(0, len(pop_sel.z)) for id in picked]) # if in the original population
        assert all([circular_dist(vals[id], pop_sel.parameters.value, pop_sel.parameters.period) <= pop_sel.parameters.distance for id in picked]) # check distance restriction circular_dist


    # GENERATE_IDD_LIST_OF_NEURONS

    def test_generate_idd_list_of_neurons(self, init_pop_selector):
        pop_sel, _ = init_pop_selector
        sel_len = pop_sel.parameters.num_of_cells
        picked = sorted(pop_sel.pick_close_to_annotation()) # not mocked!

        selected_pop = pop_sel.generate_idd_list_of_neurons()
        
        if sel_len > 0:
            if sel_len <= len(pop_sel.z): # if n <= population size
                assert len(selected_pop) == sel_len
            else:
                assert len(selected_pop) == len(pop_sel.z)
            
            if sel_len > 0 and len(pop_sel.z) > 1: # if able to shuffle
                assert (selected_pop != pop_sel.z[picked[:sel_len]]).any() # if shuffled # what if identity
        else:
            if len(picked) > 0: # if n < 0 and there are neurons to select
                assert len(selected_pop) <= max(0, len(picked) + sel_len) # check upper bound
            else:
                assert np.array_equal(selected_pop, np.array([]).astype(int))
                




from mozaik.sheets.population_selector import SimilarAnnotationSelectorRegion

class TestSimilarAnnotationSelectorRegion:

    @pytest.fixture(scope="class", params=["similar_annotation_selector_region"])
    def params(self, request):
        yield MozaikExtendedParameterSet(f"tests/sheets/PopulationSelectorTests/param/{request.param}")
    
    
    @pytest.fixture(scope="class", params=[1000, 311, 1, 0, -1, -200])
    def init_pop_selector(self, request, mock_sheet, params):
        params.num_of_cells = request.param

        yield SimilarAnnotationSelectorRegion(mock_sheet, params), params


    # __INIT__

    def test_init_assertions(self, init_pop_selector):
        pop_sel, params = init_pop_selector
        
        assert params.annotation == pop_sel.parameters.annotation  # str
        assert params.distance == pop_sel.parameters.distance  # float
        assert params.value == pop_sel.parameters.value  # float
        assert params.num_of_cells == pop_sel.parameters.num_of_cells  # int
        assert params.period == pop_sel.parameters.period  # float

        assert params.size == pop_sel.parameters.size # float
        assert params.offset_x == pop_sel.parameters.offset_x  # float
        assert params.offset_y == pop_sel.parameters.offset_y # float


    # GENERATE_IDD_LIST_OF_NEURONS

    def test_generate_idd_list_of_neurons(self, init_pop_selector):
        pop_sel, _ = init_pop_selector
        sel_len = pop_sel.parameters.num_of_cells
        picked_or = set(pop_sel.pick_close_to_annotation()) # not mocked!
        xx, yy = pop_sel.sheet.vf_2_cs(pop_sel.sheet.pop.positions[0], pop_sel.sheet.pop.positions[1])
        picked_region = set(np.arange(0,len(xx))[np.logical_and(abs(np.array(xx - pop_sel.parameters.offset_x))\
            < pop_sel.parameters.size/2.0, abs(np.array(yy - pop_sel.parameters.offset_y)) < pop_sel.parameters.size/2.0)])
        picked = sorted(list(picked_or & picked_region))

        selected_pop = pop_sel.generate_idd_list_of_neurons()
        
        if sel_len > 0:
            assert len(selected_pop) == min(sel_len, len(picked))
            if len(selected_pop) > 0 and len(pop_sel.z) > 1: # if able to shuffle
                assert (selected_pop != pop_sel.z[picked[:sel_len]]).any() # if shuffled # what if identity
        else:
            if len(picked) > 0: # if n < 0 and there are neurons to select
                assert len(selected_pop) <= max(0, len(picked) + sel_len) # check upper bound
            else:
                assert np.array_equal(selected_pop, np.array([]).astype(int))
