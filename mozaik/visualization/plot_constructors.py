"""
This plot contains classes that assist with construction of complicated arrangements of plots.
The typical example is class that help with creating a line of plots with common y axis.
"""
import param
from param.parameterized import Parameterized
from mozaik.storage.queries import *
import matplotlib.gridspec as gridspec



class LinePlot(Parameterized):          
        """
        Plot multiple plots with common x or y axis in a row or column.
        """ 
        horizontal = param.Boolean(default=True,instantiate=True,doc="Should the line of plots be horizontal or vertical")
        shared_axis = param.Boolean(default=True,instantiate=True,doc="should the x axis or y axis (depending on the horizontal flag) considered shared") 
        shared_lim = param.Boolean(default=False,instantiate=True,doc="should the limits of the x axis or y axis (depending on the horizontal flag) be considered shared") 
        length = param.Integer(default=0,instantiate=True,doc="how many plots will there be")
        function = param.Callable(doc="The function that should be called to plot individual plots. It should accept three parameters: self,index in the line, gridspec object into which to plot the plot, the simple_plot parameters")
        max_length = param.Integer(default=4,instantiate=True,doc="The maximum # plots actually displayed")
        
        def make_line_plot(self,subplotspec,params):
            """
            Call to execute the line plot.
            
            funtion - is the function that plots the individual plots. Function has to accept idx
            
            subplotspec - is the subplotspec into which the whole lineplot is to be plotted
            params - are the simple plot parameters that will be modified and passed to each of the subpots
            """
            if not self.length:
               print 'Length not specified'
               return
               
            l = numpy.min([self.max_length,self.length])   
            
            if self.horizontal:
                gs = gridspec.GridSpecFromSubplotSpec(1,l, subplot_spec=subplotspec)
            else:
                gs = gridspec.GridSpecFromSubplotSpec(l,1, subplot_spec=subplotspec)
            
            for idx in xrange(0,l):
              p = params.copy()
              if idx > 0 and self.shared_axis and self.horizontal:
                  p.setdefault("y_label",None)
                  if self.shared_lim:
                     p.setdefault("y_axis",False)  
                  
              if (idx < l-1) and self.shared_axis and not self.horizontal:
                  p.setdefault("x_label",None)
                  if self.shared_lim:
                     p.setdefault("x_axis",False)  
                          
              if self.horizontal:
                  self._single_plot(idx,gs[0,idx],p)
              else:
                  self._single_plot(idx,gs[idx,0],p)
        
        def _single_plot(self,idx,gs,p):
            self.function(idx,gs,p)


class PerDSVPlot(LinePlot):          
      """
      This is a LinePlot that automatically partitions the datastore view based on some rules. And than
      executes the individual plots on the line over the individual DSVs.
      
      Unlike LinePlot the function parameter should accept as the first arguemnt DSV not the index of the plot on the line.
      
      The partition dsvs function should perform the partitioning.
      """
      
      function = param.Callable(instantiate=True,doc="The function that should be called to plot individual plots. It should accept three parameters: self,the DSV from which to generate the plot, gridspec object into which to plot the plot, the simple_plot parameters")

      def  __init__(self,datastore,**params):
           LinePlot.__init__(self,**params)
           self.datastore = datastore
           self.dsvs = self.partiotion_dsvs()
           self.length = len(self.dsvs)
      
      def partiotion_dsvs(self):
          raise NotImplementedError
          pass
        
      def _single_plot(self,idx,gs,p):
          f = self.function
          self.function(self.dsvs[idx],gs,p)

class PerStimulusPlot(PerDSVPlot):
    """
    Line plot where each plot corresponds to stimulus with the same parameter except trials.
    
    The self.dsvs will contain the datastores you want to plot in each of the subplots - i.e. all recordings
    in the given datastore come from the same stimulus of the same parameters except for the trial parameter.
    
    PerStimulusPlot provides several automatic titling of plots based on the stimulus name and parameters. The
    currently supported styles are:
    
    
    "None" - No title
    
    "Standard" - Simple style where the Stimulus name is plotted on one line and the parameter values on the second line
    
    "Clever" - This style is valid only for cases where only stimuli of the same type are present in the supplied DSV.
               If the style is set to Clever but the conditions doesn't hold it falls back to Standard and emits a warning.
               In this case the name of the stimulus and all parameters which are the same for all stimuli in DSV are
               not displayed. The remaining parameters are shown line after line in the format 'stimulus : value'.
    """
    title_style = param.String(default="None",instantiate=True,doc="The style of the title")
    
    def partiotion_dsvs(self):       
        return partition_by_stimulus_paramter_query(self.datastore,8)

    def _single_plot(self,idx,gs,p):
            title = self.title(idx)
            if title != None:
                p.setdefault("title",title)
            PerDSVPlot._single_plot(self,idx,gs,p)
            
    def title(self,idx):
        return None
        stimulus = self.dsvs[idx].get_stimuli()[0]
        
        if self.title_style == "None":
           return None 
        
        if self.title_style == "Standard":
           return None
        
            
            