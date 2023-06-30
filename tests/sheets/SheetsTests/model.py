from parameters import ParameterSet
from mozaik.models import Model
from mozaik import load_component
import mozaik


class ModelForSheets(Model):
    required_parameters = ParameterSet(
        {
            "sheets": ParameterSet(
                {
                    "sheet": ParameterSet,
                }
            )
        }
    )

    def __init__(self, sim, num_threads, parameters):
        Model.__init__(self, sim, num_threads, parameters)

        Sheet = load_component(self.parameters.sheets.sheet.component)
        
        sheet = Sheet(self, self.parameters.sheets.sheet.params) # nemelo by být self.sheet?
