# -*- coding: utf-8 -*-
"""
Module containing texture specific analysis.
"""
import mozaik
import numpy
import quantities as qt
from analysis import Analysis
from mozaik.tools.mozaik_parametrized import colapse, colapse_to_dictionary, MozaikParametrized
from mozaik.analysis.data_structures import SingleValue
from mozaik.analysis.data_structures import AnalogSignal
from mozaik.analysis.data_structures import AnalogSignalList
from mozaik.analysis.data_structures import PerNeuronValue

from mozaik.analysis.helper_functions import psth
from parameters import ParameterSet
from mozaik.storage import queries
from mozaik.tools.circ_stat import circ_mean, circular_dist
from mozaik.tools.neo_object_operations import neo_mean, neo_sum
from neo.core.analogsignal import AnalogSignal as NeoAnalogSignal


class TextureModulation(Analysis):
    """
    Calculates the modulation of the response to texture stimuli compared to the response to spectrally-matched noise
    """
    required_parameters = ParameterSet({
        'sheet_list' : list,
    })

    def perform_analysis(self):
        for sheet in self.parameters.sheet_list:
            textures = list(set([MozaikParametrized.idd(ads.stimulus_id).texture for ads in self.datastore.get_analysis_result()]))
            samples = list(set([MozaikParametrized.idd(ads.stimulus_id).sample for ads in self.datastore.get_analysis_result()]))
            for texture in textures:
                for sample in samples:
                    pnv_noise = queries.param_filter_query(self.datastore,identifier='PerNeuronValue',sheet_name=sheet, st_sample = sample, st_texture=texture, st_stats_type = 2).get_analysis_result()[0]
                    pnv_texture = queries.param_filter_query(self.datastore,identifier='PerNeuronValue',sheet_name=sheet, st_sample = sample, st_texture=texture, st_stats_type = 1).get_analysis_result()[0]
                    modulation=[]
                    for texture_firing_rate,noise_firing_rate in zip(pnv_texture.get_value_by_id(pnv_texture.ids),pnv_noise.get_value_by_id(pnv_noise.ids)):
                        if texture_firing_rate + noise_firing_rate > 0:
                            modulation.append((texture_firing_rate - noise_firing_rate)/(texture_firing_rate + noise_firing_rate))
                        else:
                            modulation.append(0)
                    st = MozaikParametrized.idd(pnv_texture.stimulus_id)
                    setattr(st,'stats_type',None)
                    self.datastore.full_datastore.add_analysis_result(PerNeuronValue(modulation,pnv_texture.ids,None,value_name = "Sample Modulation of " + pnv_texture.value_name, sheet_name=sheet,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(st)))

                pnvs_noise = queries.param_filter_query(self.datastore,identifier='PerNeuronValue',sheet_name=sheet, st_texture = texture, st_stats_type = 2).get_analysis_result()
                pnvs_texture = queries.param_filter_query(self.datastore,identifier='PerNeuronValue',sheet_name=sheet, st_texture = texture, st_stats_type = 1).get_analysis_result()
                mean_rates_noise = [pnv.get_value_by_id(pnvs_noise[0].ids) for pnv in pnvs_noise]
                mean_rates_texture = [pnv.get_value_by_id(pnvs_noise[0].ids) for pnv in pnvs_texture]
                _mean_rates_noise = numpy.mean(mean_rates_noise,axis=0)
                _mean_rates_texture = numpy.mean(mean_rates_texture,axis=0)
                modulation = numpy.nan_to_num((_mean_rates_texture - _mean_rates_noise)/(_mean_rates_texture + _mean_rates_noise))
                st = MozaikParametrized.idd(pnvs_texture[0].stimulus_id)

                setattr(st,'stats_type',None)
                setattr(st,'sample',None)
                self.datastore.full_datastore.add_analysis_result(PerNeuronValue(modulation,pnv_texture.ids,None,value_name = "Texture Modulation of " + pnv_texture.value_name ,sheet_name=sheet,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(st)))

            pnvs_noise = queries.param_filter_query(self.datastore,identifier='PerNeuronValue',sheet_name=sheet, st_stats_type = 2).get_analysis_result()
            pnvs_texture = queries.param_filter_query(self.datastore,identifier='PerNeuronValue',sheet_name=sheet, st_stats_type = 1).get_analysis_result()
            mean_rates_noise = [pnv.get_value_by_id(pnvs_noise[0].ids) for pnv in pnvs_noise]
            mean_rates_texture = [pnv.get_value_by_id(pnvs_noise[0].ids) for pnv in pnvs_texture]
            _mean_rates_noise = numpy.mean(mean_rates_noise,axis=0)
            _mean_rates_texture = numpy.mean(mean_rates_texture,axis=0)
            modulation = numpy.nan_to_num((_mean_rates_texture - _mean_rates_noise)/(_mean_rates_texture + _mean_rates_noise))
            st = MozaikParametrized.idd(pnvs_texture[0].stimulus_id)

            setattr(st,'stats_type',None)
            setattr(st,'sample',None)
            setattr(st,'texture',None)
            self.datastore.full_datastore.add_analysis_result(PerNeuronValue(modulation,pnv_texture.ids,None,value_name = "Global Modulation of " + pnv_texture.value_name ,sheet_name=sheet,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(st)))



class TextureVarianceRatio(Analysis):
    """
    Calculates the ratio of the variance inter-texture families on the variance intra-texture families of the response to texture stimuli compared to the response to spectrally-matched noise
    """
    required_parameters = ParameterSet({
        'sheet_list' : list,
    })

    def perform_analysis(self):
        for sheet in self.parameters.sheet_list:
            textures = list(set([MozaikParametrized.idd(ads.stimulus_id).texture for ads in self.datastore.get_analysis_result()]))
            samples = list(set([MozaikParametrized.idd(ads.stimulus_id).sample for ads in self.datastore.get_analysis_result()]))
            trials = list(set([MozaikParametrized.idd(ads.stimulus_id).trial for ads in self.datastore.get_analysis_result()]))
            mean_rates = []
            for texture in textures:
                mean_rates_texture = []
                for sample in samples:
                    mean_rates_sample = []
                    for trial in trials:

                        pnv = queries.param_filter_query(self.datastore,identifier='PerNeuronValue',sheet_name=sheet, st_sample = sample, st_texture=texture, st_trial = trial, st_stats_type = 1).get_analysis_result()[0]
                        mean_rates_sample.append(pnv.values)
                    mean_rates_texture.append(mean_rates_sample)
                mean_rates.append(mean_rates_texture)

            global_averaged_rates = numpy.mean(mean_rates, axis = (0,1,2))
            textures_averaged_rates = numpy.mean(mean_rates, axis = (1,2))
            samples_averaged_rates = numpy.mean(mean_rates, axis = 2)
            SStextures = len(trials) * len(samples) * numpy.sum((textures_averaged_rates - global_averaged_rates)**2, axis=0)
            SSsamples = len(trials) * numpy.sum((numpy.transpose(samples_averaged_rates,(1,0,2)) - textures_averaged_rates)**2, axis=(0,1))
            SStrials = numpy.sum((numpy.transpose(mean_rates,(2,0,1,3)) - samples_averaged_rates)**2, axis=(0,1,2))
            MStextures = SStextures/(len(textures)-1)
            MSsamples = SSsamples/(len(textures) * (len(samples) - 1))
            MStrials = SStrials/(len(textures) * len(samples) * (len(trials) - 1))
            SStotal = SStextures + SSsamples + SStrials
            RsquaredTextures = SStextures/SStotal
            RsquaredSamples = SSsamples/SStotal
            RsquaredTrials = SStrials/SStotal

            varianceRatio = MStextures/MSsamples
            st = MozaikParametrized.idd(pnv.stimulus_id)
            arg = numpy.argmax(varianceRatio)
            #varianceRatio[numpy.isnan(varianceRatio)] = 1
            #varianceRatio[numpy.isinf(varianceRatio)] = numpy.max(numpy.ma.masked_invalid(varianceRatio))
            setattr(st,'stats_type',None)
            setattr(st,'trial',None)
            setattr(st,'sample',None)
            setattr(st,'texture',None)



            self.datastore.full_datastore.add_analysis_result(PerNeuronValue(varianceRatio,pnv.ids,None,value_name = "Texture variance ratio",sheet_name=sheet,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(st)))
            self.datastore.full_datastore.add_analysis_result(PerNeuronValue(RsquaredTextures * 100,pnv.ids,value_units=qt.percent,value_name = "Texture r-squared",sheet_name=sheet,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(st)))
            self.datastore.full_datastore.add_analysis_result(PerNeuronValue(RsquaredSamples * 100,pnv.ids,value_units=qt.percent,value_name = "Sample r-squared",sheet_name=sheet,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(st)))
            self.datastore.full_datastore.add_analysis_result(PerNeuronValue(RsquaredTrials * 100,pnv.ids,value_units=qt.percent,value_name = "Trial r-squared",sheet_name=sheet,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(st)))


class PercentageNeuronsModulated(Analysis):
    """
    Calculates the modulation of the response to texture stimuli compared to the response to spectrally-matched noise
    """
    required_parameters = ParameterSet({
        'sheet_list' : list,
        'texture_list' : list,
    })

    def randomization_test(self, response_noise, response_texture, modulation):
        count_more_modulated = 0
        for _ in range(10):
            _tmp = numpy.concatenate((response_noise,  response_texture))
            numpy.random.shuffle(_tmp)
            tmp = numpy.split(_tmp,2)
            mean1 = numpy.mean(tmp[1])
            mean0 = numpy.mean(tmp[0])
            if abs(modulation) < abs((mean1 - mean0)/(mean1 + mean0)):
                count_more_modulated += 1
        if count_more_modulated >= 500:
            modulated = 0
        else:
            modulated = 1
        return modulated

    def perform_analysis(self):
        for sheet in self.parameters.sheet_list:
            dsv_noise = queries.param_filter_query(self.datastore,identifier='PerNeuronValue',sheet_name=sheet, st_texture = self.parameters.texture_list, value_name = "Firing rate", st_stats_type = 2)
            dsv_texture = queries.param_filter_query(self.datastore,identifier='PerNeuronValue',sheet_name=sheet, st_texture = self.parameters.texture_list, value_name = "Firing rate", st_stats_type = 1)
            pnvs_noise = dsv_noise.get_analysis_result()
            pnvs_texture = dsv_texture.get_analysis_result()
            firing_rates_noise = numpy.array([pnv.get_value_by_id(pnvs_noise[0].ids) for pnv in pnvs_noise])
            firing_rates_texture = numpy.array([pnv.get_value_by_id(pnvs_texture[0].ids) for pnv in pnvs_texture])

            assert firing_rates_noise.shape == firing_rates_texture.shape

            count_positively_modulated = 0
            count_negatively_modulated = 0

            for i in range (firing_rates_noise.shape[1]):
                mean_response_texture = numpy.mean(firing_rates_texture[:,i])
                mean_response_noise = numpy.mean(firing_rates_noise[:,i])
                modulation = (mean_response_texture - mean_response_noise)/(mean_response_texture + mean_response_noise)
                neuron_modulated = self.randomization_test(firing_rates_noise[:,i],firing_rates_texture[:,i], modulation)
                if modulation > 0:
                    count_positively_modulated += neuron_modulated
                elif modulation < 0:
                    count_negatively_modulated += neuron_modulated
            st = MozaikParametrized.idd(pnvs_texture[0].stimulus_id)

            setattr(st,'stats_type',None)
            setattr(st,'sample',None)
            setattr(st,'texture',None)

            self.datastore.full_datastore.add_analysis_result(SingleValue(value = float(count_positively_modulated)/firing_rates_noise.shape[1] * 100, value_name = "Percentage of neurons significantly positively modulated", sheet_name=sheet, tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(st)))
            self.datastore.full_datastore.add_analysis_result(SingleValue(value = float(count_negatively_modulated)/firing_rates_noise.shape[1] * 100, value_name = "Percentage of neurons significantly negatively modulated", sheet_name=sheet, tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(st)))
            self.datastore.full_datastore.add_analysis_result(SingleValue(value = float(firing_rates_noise.shape[1] - count_positively_modulated - count_negatively_modulated)/firing_rates_noise.shape[1] * 100, value_name = "Percentage of neurons not significantly modulated", sheet_name=sheet, tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(st)))

class TextureModulationFromPSTH(Analysis):
    """
    Calculates the modulation of the response to texture stimuli compared to the response to spectrally-matched noise
    """
    required_parameters = ParameterSet({
        'sheet_list' : list,
        'texture_list' : list,
        'analyze_blank_stimuli' : bool, #If true, perform the analysis also on the psth computed for the blank stimuli
    })

    def perform_analysis(self):
        samples = list(set([MozaikParametrized.idd(ads.stimulus_id).sample for ads in self.datastore.get_analysis_result()]))

        for sheet in self.parameters.sheet_list:
            averaged_noise_psths = []
            averaged_texture_psths = []
            modulation_list = []

            if self.parameters.analyze_blank_stimuli:
                averaged_noise_psths_null = []
                averaged_texture_psths_null = []
                modulation_list_null = []

            for texture in self.parameters.texture_list:
                psths_noise = queries.param_filter_query(self.datastore,identifier='AnalogSignalList',sheet_name=sheet, analysis_algorithm = "PSTH", st_stats_type = 2, st_texture = texture).get_analysis_result()
                psths_texture = queries.param_filter_query(self.datastore,identifier='AnalogSignalList',sheet_name=sheet, analysis_algorithm = "PSTH", st_stats_type = 1, st_texture = texture).get_analysis_result()
                ids = psths_noise[0].ids
                t_start = psths_noise[0].asl[0].t_start
                sampling_period = psths_noise[0].asl[0].sampling_period
                units = psths_noise[0].asl[0].units
                assert len(psths_noise) == len(psths_texture)
                asls_noise = [psth.get_asl_by_id(ids) for psth in psths_noise]
                asls_texture = [psth.get_asl_by_id(ids) for psth in psths_texture]

                noise_psth = numpy.mean(asls_noise, axis = 0)
                texture_psth = numpy.mean(asls_texture, axis = 0)
                modulation = numpy.nan_to_num((texture_psth - noise_psth)/(texture_psth + noise_psth))
                averaged_noise_psths.append(noise_psth)
                averaged_texture_psths.append(texture_psth)
                modulation_list.append(modulation)

                averaged_noise_asls = [NeoAnalogSignal(asl, t_start=t_start, sampling_period=sampling_period,units = units) for asl in noise_psth]
                averaged_texture_asls = [NeoAnalogSignal(asl, t_start=t_start, sampling_period=sampling_period,units = units) for asl in texture_psth]
                modulation_asls = [NeoAnalogSignal(asl, t_start=t_start, sampling_period=sampling_period,units = qt.dimensionless) for asl in modulation]

                st_noise = MozaikParametrized.idd(psths_noise[0].stimulus_id)
                setattr(st_noise,'sample',None)
                st_texture = MozaikParametrized.idd(psths_texture[0].stimulus_id)
                setattr(st_texture,'sample',None)

                st_modulation = MozaikParametrized.idd(psths_noise[0].stimulus_id)
                setattr(st_modulation,'sample',None)
                setattr(st_modulation,'stats_type',None)

                self.datastore.full_datastore.add_analysis_result(
                    AnalogSignalList(averaged_noise_asls,
                                         ids,
                                         psths_noise[0].y_axis_units,
                                         x_axis_name='time',
                                         y_axis_name='Noise ' + psths_noise[0].y_axis_name + ' samples averaged',
                                         sheet_name=sheet,
                                         tags=self.tags,
                                         analysis_algorithm=self.__class__.__name__,
                                         stimulus_id=str(st_noise)))
                self.datastore.full_datastore.add_analysis_result(
                    AnalogSignalList(averaged_texture_asls,
                                         ids,
                                         psths_noise[0].y_axis_units,
                                         x_axis_name='time',
                                         y_axis_name='Texture ' + psths_noise[0].y_axis_name + ' samples averaged',
                                         sheet_name=sheet,
                                         tags=self.tags,
                                         analysis_algorithm=self.__class__.__name__,
                                         stimulus_id=str(st_texture)))
                self.datastore.full_datastore.add_analysis_result(
                    AnalogSignalList(modulation_asls,
                                         ids,
                                         qt.dimensionless,
                                         x_axis_name='time',
                                         y_axis_name='Modulation ' + psths_noise[0].y_axis_name + ' samples averaged',
                                         sheet_name=sheet,
                                         tags=self.tags,
                                         analysis_algorithm=self.__class__.__name__,
                                         stimulus_id=str(st_modulation)))

                if self.parameters.analyze_blank_stimuli:
                    
                    psths_noise_null = queries.param_filter_query(self.datastore,identifier='AnalogSignalList',sheet_name=sheet, analysis_algorithm = "PSTH_null", st_stats_type = 2, st_texture = texture).get_analysis_result()
                    psths_texture_null = queries.param_filter_query(self.datastore,identifier='AnalogSignalList',sheet_name=sheet, analysis_algorithm = "PSTH_null", st_stats_type = 1, st_texture = texture).get_analysis_result()
                    ids = psths_noise_null[0].ids
                    t_start_null = psths_noise_null[0].asl[0].t_start
                    sampling_period_null = psths_noise_null[0].asl[0].sampling_period
                    assert len(psths_noise_null) == len(psths_texture_null)
                    asls_noise = [psth.get_asl_by_id(ids) for psth in psths_noise_null]
                    asls_texture = [psth.get_asl_by_id(ids) for psth in psths_texture_null]

                    noise_psth = numpy.mean(asls_noise, axis = 0)
                    texture_psth = numpy.mean(asls_texture, axis = 0)
                    modulation = numpy.nan_to_num((texture_psth - noise_psth)/(texture_psth + noise_psth))
                    averaged_noise_psths_null.append(noise_psth)
                    averaged_texture_psths_null.append(texture_psth)
                    modulation_list_null.append(modulation)

                    averaged_noise_asls = [NeoAnalogSignal(asl, t_start=t_start_null, sampling_period=sampling_period_null,units = units) for asl in noise_psth]
                    averaged_texture_asls = [NeoAnalogSignal(asl, t_start=t_start_null, sampling_period=sampling_period_null,units = units) for asl in texture_psth]
                    modulation_asls = [NeoAnalogSignal(asl, t_start=t_start_null, sampling_period=sampling_period_null,units = qt.dimensionless) for asl in modulation]

                    self.datastore.full_datastore.add_analysis_result(
                        AnalogSignalList(averaged_noise_asls,
                                             ids,
                                             psths_noise_null[0].y_axis_units,
                                             x_axis_name='time',
                                             y_axis_name='Noise null ' + psths_noise[0].y_axis_name + ' samples averaged',
                                             sheet_name=sheet,
                                             tags=self.tags,
                                             analysis_algorithm=self.__class__.__name__,
                                             stimulus_id=str(st_noise)))
                    self.datastore.full_datastore.add_analysis_result(
                        AnalogSignalList(averaged_texture_asls,
                                             ids,
                                             psths_noise_null[0].y_axis_units,
                                             x_axis_name='time',
                                             y_axis_name='Texture null ' + psths_noise[0].y_axis_name + ' samples averaged',
                                             sheet_name=sheet,
                                             tags=self.tags,
                                             analysis_algorithm=self.__class__.__name__,
                                             stimulus_id=str(st_texture)))
                    self.datastore.full_datastore.add_analysis_result(
                        AnalogSignalList(modulation_asls,
                                             ids,
                                             qt.dimensionless,
                                             x_axis_name='time',
                                             y_axis_name='Modulation null ' + psths_noise[0].y_axis_name + ' samples averaged',
                                             sheet_name=sheet,
                                             tags=self.tags,
                                             analysis_algorithm=self.__class__.__name__,
                                             stimulus_id=str(st_modulation)))


            noise_psth = numpy.mean(averaged_noise_psths, axis = 0)
            texture_psth = numpy.mean(averaged_texture_psths, axis = 0)
            modulation = numpy.mean(modulation_list, axis = 0)

            var_noise_psth = numpy.var(averaged_noise_psths, axis = 0)
            var_texture_psth = numpy.var(averaged_texture_psths, axis = 0)

            averaged_noise_asls = [NeoAnalogSignal(asl, t_start=t_start, sampling_period=sampling_period,units = units) for asl in noise_psth]
            averaged_texture_asls = [NeoAnalogSignal(asl, t_start=t_start, sampling_period=sampling_period,units = units) for asl in texture_psth]
            modulation_asls = [NeoAnalogSignal(asl, t_start=t_start, sampling_period=sampling_period,units = qt.dimensionless) for asl in modulation]

            var_noise_asls = [NeoAnalogSignal(asl, t_start=t_start, sampling_period=sampling_period,units = units) for asl in var_noise_psth]
            var_texture_asls = [NeoAnalogSignal(asl, t_start=t_start, sampling_period=sampling_period,units = units) for asl in var_texture_psth]

            setattr(st_noise,'texture',None)
            setattr(st_texture,'texture',None)
            setattr(st_modulation,'texture',None)

            self.datastore.full_datastore.add_analysis_result(
                    AnalogSignalList(averaged_noise_asls,
                                         ids,
                                         psths_noise[0].y_axis_units,
                                         x_axis_name='time',
                                         y_axis_name='Noise ' +  psths_noise[0].y_axis_name + ' textures averaged',
                                         sheet_name=sheet,
                                         tags=self.tags,
                                         analysis_algorithm=self.__class__.__name__,
                                         stimulus_id=str(st_noise)))
            self.datastore.full_datastore.add_analysis_result(
                    AnalogSignalList(averaged_texture_asls,
                                         ids,
                                         psths_noise[0].y_axis_units,
                                         x_axis_name='time',
                                         y_axis_name='Texture ' + psths_noise[0].y_axis_name + ' textures averaged',
                                         sheet_name=sheet,
                                         tags=self.tags,
                                         analysis_algorithm=self.__class__.__name__,
                                         stimulus_id=str(st_texture)))
            self.datastore.full_datastore.add_analysis_result(
                    AnalogSignalList(var_noise_asls,
                                         ids,
                                         psths_noise[0].y_axis_units,
                                         x_axis_name='time',
                                         y_axis_name='Noise ' + psths_noise[0].y_axis_name + ' textures var',
                                         sheet_name=sheet,
                                         tags=self.tags,
                                         analysis_algorithm=self.__class__.__name__,
                                         stimulus_id=str(st_noise)))
            self.datastore.full_datastore.add_analysis_result(
                    AnalogSignalList(var_texture_asls,
                                         ids,
                                         psths_noise[0].y_axis_units,
                                         x_axis_name='time',
                                         y_axis_name='Texture ' + psths_noise[0].y_axis_name + ' textures var',
                                         sheet_name=sheet,
                                         tags=self.tags,
                                         analysis_algorithm=self.__class__.__name__,
                                         stimulus_id=str(st_texture)))
            self.datastore.full_datastore.add_analysis_result(
                    AnalogSignalList(modulation_asls,
                                         ids,
                                         qt.dimensionless,
                                         x_axis_name='time',
                                         y_axis_name='Modulation ' + psths_noise[0].y_axis_name + ' textures averaged',
                                         sheet_name=sheet,
                                         tags=self.tags,
                                         analysis_algorithm=self.__class__.__name__,
                                         stimulus_id=str(st_modulation)))

            if self.parameters.analyze_blank_stimuli:

                noise_psth = numpy.mean(averaged_noise_psths_null, axis = 0)
                texture_psth = numpy.mean(averaged_texture_psths_null, axis = 0)
                modulation = numpy.mean(modulation_list_null, axis = 0)

                var_noise_psth = numpy.var(averaged_noise_psths_null, axis = 0)
                var_texture_psth = numpy.var(averaged_texture_psths_null, axis = 0)

                averaged_noise_asls = [NeoAnalogSignal(asl, t_start=t_start_null, sampling_period=sampling_period_null,units = units) for asl in noise_psth]
                averaged_texture_asls = [NeoAnalogSignal(asl, t_start=t_start_null, sampling_period=sampling_period_null,units = units) for asl in texture_psth]
                modulation_asls = [NeoAnalogSignal(asl, t_start=t_start_null, sampling_period=sampling_period_null,units = qt.dimensionless) for asl in modulation]

                var_noise_asls = [NeoAnalogSignal(asl, t_start=t_start_null, sampling_period=sampling_period_null,units = units) for asl in var_noise_psth]
                var_texture_asls = [NeoAnalogSignal(asl, t_start=t_start_null, sampling_period=sampling_period_null,units = units) for asl in var_texture_psth]

                setattr(st_noise,'texture',None)
                setattr(st_texture,'texture',None)
                setattr(st_modulation,'texture',None)

                self.datastore.full_datastore.add_analysis_result(
                        AnalogSignalList(averaged_noise_asls,
                                             ids,
                                             psths_noise_null[0].y_axis_units,
                                             x_axis_name='time',
                                             y_axis_name='Noise null ' + psths_noise[0].y_axis_name + ' textures averaged',
                                             sheet_name=sheet,
                                             tags=self.tags,
                                             analysis_algorithm=self.__class__.__name__,
                                             stimulus_id=str(st_noise)))
                self.datastore.full_datastore.add_analysis_result(
                        AnalogSignalList(averaged_texture_asls,
                                             ids,
                                             psths_noise_null[0].y_axis_units,
                                             x_axis_name='time',
                                             y_axis_name='Texture null ' + psths_noise[0].y_axis_name + ' textures averaged',
                                             sheet_name=sheet,
                                             tags=self.tags,
                                             analysis_algorithm=self.__class__.__name__,
                                             stimulus_id=str(st_texture)))
                self.datastore.full_datastore.add_analysis_result(
                        AnalogSignalList(var_noise_asls,
                                             ids,
                                             psths_noise_null[0].y_axis_units,
                                             x_axis_name='time',
                                             y_axis_name='Noise null ' + psths_noise[0].y_axis_name + ' textures var',
                                             sheet_name=sheet,
                                             tags=self.tags,
                                             analysis_algorithm=self.__class__.__name__,
                                             stimulus_id=str(st_noise)))
                self.datastore.full_datastore.add_analysis_result(
                        AnalogSignalList(var_texture_asls,
                                             ids,
                                             psths_noise_null[0].y_axis_units,
                                             x_axis_name='time',
                                             y_axis_name='Texture null ' + psths_noise[0].y_axis_name + ' textures var',
                                             sheet_name=sheet,
                                             tags=self.tags,
                                             analysis_algorithm=self.__class__.__name__,
                                             stimulus_id=str(st_texture)))
                self.datastore.full_datastore.add_analysis_result(
                        AnalogSignalList(modulation_asls,
                                             ids,
                                             qt.dimensionless,
                                             x_axis_name='time',
                                             y_axis_name='Modulation null ' + psths_noise[0].y_axis_name + ' textures averaged',
                                             sheet_name=sheet,
                                             tags=self.tags,
                                             analysis_algorithm=self.__class__.__name__,
                                             stimulus_id=str(st_modulation)))

