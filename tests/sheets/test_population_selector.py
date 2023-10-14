import mozaik
from mozaik.core import ParametrizedObject
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet

import pytest
from unittest.mock import MagicMock
from unittest.mock import patch
from unittest.mock import call



@pytest.fixture(scope="module")
def mock_sheet():
    """
    Mocking the Sheet class to separate the testing of PopulationSelector and the Sheet itself.
    Scope being equal to module means creating the mocked object only once per module.
    """
    sheet = MagicMock()
    
    yield sheet




from mozaik.sheets.population_selector import PopulationSelector

class TestPopulationSelector:

    @pytest.fixture(scope="class", params=["retinal_sheet"]) # muzu z SheetsTests asi brát
    def params(self, request):
        """A fixture for getting the current population selector's parameters"""
        yield MozaikExtendedParameterSet(f"tests/sheets/SheetsTests/param/{request.param}")


    @pytest.fixture(scope="class")
    def init_pop_selector(self, mock_sheet, params):
        """A fixture for the initialization of the PopulationSelector object, created once"""
        yield PopulationSelector(mock_sheet, params), params


    # __INIT__

    def test_init_assertions(self, init_pop_selector):
        pop_sel, params = init_pop_selector

        # assert pop_sel.sheet == params.sheet # nebo teda mock_sheet


    def test_init_parametrized_object_call(self, mock_sheet, params):
        with patch.object(ParametrizedObject, '__init__') as mock_parent_call:
            pop_sel = PopulationSelector(mock_sheet, params)

            mock_parent_call.assert_called_once_with(pop_sel, params)


    # GENERATE_IDD_LIST_OF_NEURONS

    def test_generate_idd_list_of_neurons(self, init_pop_selector):
        pop_sel, _ = init_pop_selector
        
        with pytest.raises(NotImplementedError):
            pop_sel.generate_idd_list_of_neurons()





class TestRCAll:

    # GENERATE_IDD_LIST_OF_NEURONS

    def test_generate_idd_list_of_neurons(self):#, init_pop_selector):
        # pop_sel, _ = init_pop_selector
        ...
        # selected_pop = pop_sel.generate_idd_list_of_neurons

        # assert len(pop_sel.sheet.pop.all_cells) == len(selected_pop) # líp brát len(parametrů)
        # assert pop_sel.sheet.pop.all_cells == selected_pop # jinak




class TestRCRandomN:

    # __INIT__

    def test_init_assertions(self):#, init_pop_selector):
        # pop_sel, params = init_pop_selector
        ...
        # assert params... == pop_sel.num_of_cells # int


    # GENERATE_IDD_LIST_OF_NEURONS

    def test_generate_idd_list_of_neurons(self):
        ...




class TestRCRandomPercentage:

    # __INIT__

    def test_init_assertions(self):#, init_pop_selector):
        # pop_sel, params = init_pop_selector
        ...
        # assert params... == pop_sel.percentage # float


    # GENERATE_IDD_LIST_OF_NEURONS

    def test_generate_idd_list_of_neurons(self):
        ...




class TestRCGrid:

    # __INIT__

    def test_init_assertions(self):#, init_pop_selector):
        # pop_sel, params = init_pop_selector
        ...
        # assert params... == pop_sel.size # float
        # assert params... == pop_sel.spacing  # float
        # assert params... == pop_sel.offset_x  # float
        # assert params... == pop_sel.offset_y # float


    # GENERATE_IDD_LIST_OF_NEURONS

    def test_generate_idd_list_of_neurons(self):
        ...




class TestRCGridDegree:

    # __INIT__

    def test_init_assertions(self):#, init_pop_selector):
        # pop_sel, params = init_pop_selector
        ...
        # assert params... == pop_sel.size  # float
        # assert params... == pop_sel.spacing  # float
        # assert params... == pop_sel.offset_x  # float
        # assert params... == pop_sel.offset_y  # float


    # GENERATE_IDD_LIST_OF_NEURONS

    def test_generate_idd_list_of_neurons(self):
        ...




class TestSimilarAnnotationSelector:

    # __INIT__

    def test_init_assertions(self):#, init_pop_selector):
        # pop_sel, params = init_pop_selector
        ...
        # assert params... == pop_sel.annotation  # str
        # assert params... == pop_sel.distance  # float
        # assert params... == pop_sel.value  # float
        # assert params... == pop_sel.num_of_cells  # int
        # assert params... == pop_sel.period  # float


    # PICK_CLOSE_TO_ANNOTATION

    def test_pick_close_to_annotation(self):
        ...


    # GENERATE_IDD_LIST_OF_NEURONS

    def test_generate_idd_list_of_neurons(self):
        ...




class TestSimilarAnnotationSelectorRegion:

    # __INIT__

    def test_init_assertions(self):#, init_pop_selector):
        # pop_sel, params = init_pop_selector
        ...
        # assert params... == pop_sel.size  # float
        # assert params... == pop_sel.offset_x  # float
        # assert params... == pop_sel.offset_y  # float


    # GENERATE_IDD_LIST_OF_NEURONS

    def test_generate_idd_list_of_neurons(self):
        ...
