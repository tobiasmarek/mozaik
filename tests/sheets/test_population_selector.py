import mozaik
import numpy as np
from mozaik.core import ParametrizedObject
from parameters import ParameterSet
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet

import pytest
from unittest.mock import MagicMock
from unittest.mock import patch
from unittest.mock import call



@pytest.fixture(scope="module")
def mock_sheet(): # or using the real Sheet class instead?
    """
    Mocking the Sheet class to separate the testing of PopulationSelector and the Sheet itself.
    Scope being equal to module means creating the mocked object only once per module.
    """
    sheet = MagicMock()
    sheet.pop.all_cells = np.array([[0,1.5,3], [4,5.5,7]]) # not how all_cells look at all
    sheet.pop.positions = MagicMock() # need to be real positions, not mock
    
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

        selected_pop = pop_sel.generate_idd_list_of_neurons()

        assert len(pop_sel.sheet.pop.all_cells) == len(set(selected_pop))




from mozaik.sheets.population_selector import RCRandomN

class TestRCRandomN:

    @pytest.fixture(scope="class", params=["rc_random_n_pop_sel"])
    def params(self, request):
        """A fixture for getting the current population selector's parameters"""
        yield MozaikExtendedParameterSet(f"tests/sheets/PopulationSelectorTests/param/{request.param}")
    
    
    @pytest.fixture(scope="class")
    def init_pop_selector(self, mock_sheet, params):
        """A fixture for the initialization of the PopulationSelector object, created once"""
        yield RCRandomN(mock_sheet, params), params


    # __INIT__

    def test_init_assertions(self, init_pop_selector):
        pop_sel, params = init_pop_selector
        
        assert params.num_of_cells == pop_sel.parameters.num_of_cells # int


    # GENERATE_IDD_LIST_OF_NEURONS
    # parametrize smh
    def test_generate_idd_list_of_neurons(self, init_pop_selector): # what if n > than real number of cells || <= 0
        pop_sel, _ = init_pop_selector

        selected_pop = pop_sel.generate_idd_list_of_neurons() # mozaik.rng returns None

        assert len(set(selected_pop)) == pop_sel.parameters.num_of_cells
        assert selected_pop != pop_sel.sheet.pop.all_cells.astype(int)[:pop_sel.parameters.num_of_cells] # if shuffled




from mozaik.sheets.population_selector import RCRandomPercentage

class TestRCRandomPercentage:

    @pytest.fixture(scope="class", params=["rc_random_percentage_pop_sel"])
    def params(self, request):
        yield MozaikExtendedParameterSet(f"tests/sheets/PopulationSelectorTests/param/{request.param}")
    
    
    @pytest.fixture(scope="class")
    def init_pop_selector(self, mock_sheet, params):
        yield RCRandomPercentage(mock_sheet, params), params


    # __INIT__

    def test_init_assertions(self, init_pop_selector):
        pop_sel, params = init_pop_selector
        
        assert params.percentage == pop_sel.parameters.percentage # float


    # GENERATE_IDD_LIST_OF_NEURONS

    def test_generate_idd_list_of_neurons(self, init_pop_selector): # what if percentage > 100 || <= 0
        pop_sel, _ = init_pop_selector
        true_len = int(len(pop_sel.sheet.pop.all_cells.astype(int)) * pop_sel.parameters.percentage/100)

        selected_pop = pop_sel.generate_idd_list_of_neurons() # mozaik.rng returns None

        assert len(set(selected_pop)) == true_len
        assert selected_pop != pop_sel.sheet.pop.all_cells.astype(int)[:true_len] # if shuffled




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

    def test_generate_idd_list_of_neurons(self, init_pop_selector): # unfinished
        pop_sel, _ = init_pop_selector

        selected_pop = pop_sel.generate_idd_list_of_neurons()
        
        ... 



    def test_generate_idd_size_not_multiple_of_spacing(self, init_pop_selector): # unfinished
        pop_sel, _ = init_pop_selector # size has to be a multiple of spacing, new param needed
        
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

    def test_generate_idd_list_of_neurons(self, init_pop_selector): # unfinished
        pop_sel, _ = init_pop_selector

        selected_pop = pop_sel.generate_idd_list_of_neurons()
        
        ...

    
    def test_generate_idd_size_not_multiple_of_spacing(self, init_pop_selector): # unfinished
        pop_sel, _ = init_pop_selector # size has to be a multiple of spacing, new param needed
        
        with pytest.raises(AssertionError):
            pop_sel.generate_idd_list_of_neurons()




from mozaik.sheets.population_selector import SimilarAnnotationSelector

class TestSimilarAnnotationSelector:

    @pytest.fixture(scope="class", params=["similar_annotation_selector"])
    def params(self, request):
        yield MozaikExtendedParameterSet(f"tests/sheets/PopulationSelectorTests/param/{request.param}")
    
    
    @pytest.fixture(scope="class")
    def init_pop_selector(self, mock_sheet, params):
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
        ...


    def test_pick_close_to_annotation_non_zero_period(self, init_pop_selector):
        ...


    # GENERATE_IDD_LIST_OF_NEURONS

    def test_generate_idd_list_of_neurons(self, init_pop_selector): # unfinished
        pop_sel, _ = init_pop_selector

        selected_pop = pop_sel.generate_idd_list_of_neurons()
        
        ... # check if it is shuffled smh




from mozaik.sheets.population_selector import SimilarAnnotationSelectorRegion

class TestSimilarAnnotationSelectorRegion:

    @pytest.fixture(scope="class", params=["similar_annotation_selector_region"])
    def params(self, request):
        yield MozaikExtendedParameterSet(f"tests/sheets/PopulationSelectorTests/param/{request.param}")
    
    
    @pytest.fixture(scope="class")
    def init_pop_selector(self, mock_sheet, params):
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

    def test_generate_idd_list_of_neurons(self, init_pop_selector): # unfinished
        pop_sel, _ = init_pop_selector

        selected_pop = pop_sel.generate_idd_list_of_neurons()
        
        ... # check if it is shuffled smh and other things
