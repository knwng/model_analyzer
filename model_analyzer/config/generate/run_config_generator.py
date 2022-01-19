# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .config_generator_interface import ConfigGeneratorInterface
from .model_config_generator import ModelConfigGenerator
from .perf_analyzer_config_generator import PerfAnalyzerConfigGenerator

from model_analyzer.config.run.run_config import RunConfig


class RunConfigGenerator(ConfigGeneratorInterface):
    """
    Generates all RunConfigs to execute for the given model
    """

    def __init__(self, config, models, client):
        """
        Parameters
        ----------
        config: ModelAnalyzerConfig
        
        models: List of ConfigModelProfileSpec
            The list of models to generate RunConfigs for
            
        client: TritonClient
        """
        self._config = config
        self._models = models
        self._client = client

        # MM-PHASE 0: Assuming that all models are identical, so using first model's name/flag/parameters
        self._model_name = models[0].model_name()
        self._model_pa_flags = models[0].perf_analyzer_flags()
        self._model_parameters = models[0].parameters()

        self._is_done = False

    def is_done(self):
        """ Returns true if this generator is done generating configs """
        return self._is_done

    def next_config(self):
        """
        Returns
        -------
        RunConfig
            The next RunConfig generated by this class
        """

        pacg = PerfAnalyzerConfigGenerator(self._config, self._model_name,
                                           self._model_pa_flags,
                                           self._model_parameters)

        while not pacg.is_done():
            perf_analyzer_config = pacg.next_config()

            model_configs = self._generate_all_model_config_permuations(
                self._models)

            for model_config in model_configs:
                run_config = self._generate_run_config(model_config,
                                                       perf_analyzer_config)

                if pacg.is_done() and model_config == model_configs[-1]:
                    self._is_done = True

                yield run_config

    def set_last_results(self, measurements):
        """ 
        Given the results from the last RunConfig, make decisions 
        about future configurations to generate

        Parameters
        ----------
        measurements: List of Measurements from the last run(s)
        """
        self._curr_pacg.set_last_results(measurements)

    def _generate_run_config(self, model_configs, perf_analyzer_config):
        # MM-PHASE 0: Assuming that all models are identical, so using first model's server env
        run_config = RunConfig(self._model_name, model_configs,
                               perf_analyzer_config,
                               self._models[0].triton_server_environment())

        return run_config

    def _generate_all_model_config_permuations(self, models):
        child_model_configs = []
        if (len(models) > 1):
            child_model_configs.extend(
                self._generate_all_model_config_permuations(models[1:]))

        parent_model_configs = self._generate_parent_model_configs(models[0])

        return self._combine_model_config_permuations(parent_model_configs,
                                                      child_model_configs)

    def _generate_parent_model_configs(self, model):
        mcg = ModelConfigGenerator(self._config, model, self._client)

        model_configs = []
        while not mcg.is_done():
            model_configs.append(mcg.next_config())

        return model_configs

    def _combine_model_config_permuations(self, parent, child):
        model_configs = []

        if len(child) > 0:
            for p in parent:
                for c in child:
                    combined_list = [p]
                    # Children at the lowest level of recursion will not be lists,
                    # so we need to check and handle this case correctly
                    combined_list.extend([c] if not isinstance(c, list) else c)

                    model_configs.append(combined_list)
        else:
            model_configs.extend(p for p in parent)

        return model_configs
