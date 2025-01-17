# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.device.gpu_device import GPUDevice
from model_analyzer.triton.client.client import TritonClient
from model_analyzer.result.run_config_measurement import RunConfigMeasurement
from model_analyzer.config.generate.model_variant_name_manager import ModelVariantNameManager
from .config_generator_interface import ConfigGeneratorInterface
from typing import List, Optional, Generator, Dict, Any
from model_analyzer.constants import LOGGER_NAME
from model_analyzer.triton.model.model_config import ModelConfig
from .model_profile_spec import ModelProfileSpec
import abc
import logging

logger = logging.getLogger(LOGGER_NAME)


class BaseModelConfigGenerator(ConfigGeneratorInterface):
    """ Base class for generating model configs """

    def __init__(self, config: ConfigCommandProfile, gpus: List[GPUDevice],
                 model: ModelProfileSpec, client: TritonClient,
                 model_variant_name_manager: ModelVariantNameManager,
                 default_only: bool, early_exit_enable: bool) -> None:
        """
        Parameters
        ----------
        config: ConfigCommandProfile
        gpus: List of GPUDevices
        model: ModelProfileSpec
            The model to generate ModelConfigs for
        client: TritonClient
        model_variant_name_manager: ModelVariantNameManager
        default_only: Bool
            If true, only the default config will be generated
            If false, the default config will NOT be generated
        early_exit_enable: Bool
            If true, the generator can early exit if throughput plateaus
        """
        self._config = config
        self._client = client
        self._model_variant_name_manager = model_variant_name_manager
        self._base_model = model
        self._base_model_name = model.model_name()
        self._remote_mode = config.triton_launch_mode == 'remote'
        self._cpu_only = model.cpu_only()
        self._default_only = default_only
        self._early_exit_enable = early_exit_enable
        self._model_name_index = 0
        self._generator_started = False
        self._max_batch_size_warning_printed = False
        self._last_results: List[Optional[RunConfigMeasurement]] = []
        # Contains the max throughput from each provided list of measurements
        # since the last time we stepped max_batch_size
        #
        self._curr_max_batch_size_throughputs: List[float] = []

    def _is_done(self) -> bool:
        """ Returns true if this generator is done generating configs """
        return self._generator_started and (self._default_only or
                                            self._done_walking())

    def get_configs(self) -> Generator[ModelConfig, None, None]:
        """
        Returns
        -------
        ModelConfig
            The next ModelConfig generated by this class
        """
        while True:
            if self._is_done():
                break

            self._generator_started = True
            config = self._get_next_model_config()
            yield (config)
            self._step()

    def set_last_results(
            self, measurements: List[Optional[RunConfigMeasurement]]) -> None:
        """
        Given the results from the last ModelConfig, make decisions
        about future configurations to generate

        Parameters
        ----------
        measurements: List of Measurements from the last run(s)
        """
        self._last_results = measurements

    @abc.abstractmethod
    def _done_walking(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def _step(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_next_model_config(self) -> ModelConfig:
        raise NotImplementedError

    def _last_results_erroneous(self) -> bool:
        last_max_throughput = self._get_last_results_max_throughput()
        return last_max_throughput is None

    def _last_results_increased_throughput(self) -> bool:
        if len(self._curr_max_batch_size_throughputs) < 2:
            return True

        lastest_throughput = self._curr_max_batch_size_throughputs[-1]
        return all(
            lastest_throughput > prev_throughput
            for prev_throughput in self._curr_max_batch_size_throughputs[:-1])

    def _get_last_results_max_throughput(self) -> Optional[float]:
        throughputs = [
            m.get_non_gpu_metric_value('perf_throughput')
            for m in self._last_results
            if m is not None
        ]
        if not throughputs:
            return None
        else:
            return max(throughputs)

    def _make_remote_model_config(self) -> ModelConfig:
        if not self._config.reload_model_disable:
            self._client.load_model(self._base_model_name)
        model_config = ModelConfig.create_from_triton_api(
            self._client, self._base_model_name,
            self._config.client_max_retries)
        model_config.set_cpu_only(self._cpu_only)
        if not self._config.reload_model_disable:
            self._client.unload_model(self._base_model_name)

        return model_config

    def _make_direct_mode_model_config(self, param_combo: Dict) -> ModelConfig:
        return BaseModelConfigGenerator.make_model_config(
            param_combo=param_combo,
            model=self._base_model,
            model_variant_name_manager=self._model_variant_name_manager)

    @staticmethod
    def make_model_config(
            param_combo: dict, model: ModelProfileSpec,
            model_variant_name_manager: ModelVariantNameManager) -> ModelConfig:
        """
        Loads the base model config from the model repository, and then applies the
        parameters in the param_combo on top to create and return a new model config
        
        Parameters:
        -----------
        param_combo: dict
            dict of key:value pairs to apply to the model config
        model: ModelProfileSpec
        model_variant_name_manager: ModelVariantNameManager
        """
        logger_str: List[str] = []
        model_name = model.model_name()
        model_config_dict = BaseModelConfigGenerator._apply_param_combo_to_model(
            model, param_combo, logger_str)

        (variant_found,
         variant_name) = model_variant_name_manager.get_model_variant_name(
             model_name, model_config_dict, param_combo)

        model_config_dict['name'] = variant_name
        logger.info("")
        if variant_found:
            logger.info(
                f"Found existing model config: {model_config_dict['name']}")
        else:
            logger.info(f"Creating model config: {model_config_dict['name']}")
        for str in logger_str:
            logger.info(str)
        logger.info("")

        model_config = ModelConfig.create_from_dictionary(model_config_dict)
        model_config.set_cpu_only(model.cpu_only())

        return model_config

    @staticmethod
    def make_ensemble_model_config(
            model: ModelProfileSpec,
            ensemble_submodel_configs: List[ModelConfig],
            model_variant_name_manager: ModelVariantNameManager,
            param_combo: Dict = {}) -> ModelConfig:
        """
        Loads the ensemble model spec from the model repository, and then mutates
        the names to match the ensemble submodels
        
        Parameters
        ----------
        model: ModelProfileSpec
            The top-level ensemble model spec
        ensemble_submodel_configs: List of ModelConfigs
            The list of submodel ModelConfigs 
        model_variant_name_manager: ModelVariantNameManager

        """
        logger_str: List[str] = []
        model_name = model.model_name()
        model_config_dict = BaseModelConfigGenerator._apply_param_combo_to_model(
            model, param_combo, logger_str)

        ensemble_config_dicts = [
            submodel_config.to_dict()
            for submodel_config in ensemble_submodel_configs
        ]
        ensemble_key = ModelVariantNameManager.make_ensemble_submodel_key(
            ensemble_config_dicts)

        (variant_found, variant_name
        ) = model_variant_name_manager.get_ensemble_model_variant_name(
            model_name, ensemble_key)

        model_config_dict['name'] = variant_name
        model_config = ModelConfig.create_from_dictionary(model_config_dict)

        for submodel_config in ensemble_submodel_configs:
            variant_name = submodel_config.get_field("name")
            submodel_name = BaseModelConfigGenerator.extract_model_name_from_variant_name(
                variant_name)

            model_config.set_submodel_variant_name(submodel_name=submodel_name,
                                                   variant_name=variant_name)

        return model_config

    @staticmethod
    def _apply_param_combo_to_model(model: ModelProfileSpec, param_combo: dict,
                                    logger_str: List[str]) -> dict:
        """
        Given a model, apply any parameters and return a model config dictionary
        """
        model_config_dict = model.get_default_config()
        if param_combo is not None:
            for key, value in param_combo.items():
                if value is not None:
                    BaseModelConfigGenerator._apply_value_to_dict(
                        key, value, model_config_dict)

                    if value == {}:
                        logger_str.append(f"  Enabling {key}")
                    else:
                        logger_str.append(f"  Setting {key} to {value}")

        return model_config_dict

    def _reset_max_batch_size(self) -> None:
        self._max_batch_size_warning_printed = False
        self._curr_max_batch_size_throughputs = []

    def _print_max_batch_size_plateau_warning(self) -> None:
        if not self._max_batch_size_warning_printed:
            logger.info(
                "No longer increasing max_batch_size because throughput has plateaued"
            )
            self._max_batch_size_warning_printed = True

    @staticmethod
    def extract_model_name_from_variant_name(variant_name: str) -> str:
        """
        Removes '_config_#/default' from the variant name and returns
        the model name, eg. model_name_config_10 -> model_name
        """
        return variant_name[:variant_name.find("_config_")]

    @staticmethod
    def _apply_value_to_dict(key: Any, value: Any, dict_in: Dict) -> None:
        """
        Apply the supplied value at the given key into the provided dict.

        If the key already exists in the dict and both the existing value as well
        as the new input value are dicts, only overwrite the subkeys (recursively)
        provided in the value
        """

        if type(dict_in.get(key, None)) is dict and type(value) is dict:
            for subkey, subvalue in value.items():
                BaseModelConfigGenerator._apply_value_to_dict(
                    subkey, subvalue, dict_in.get(key, None))
        else:
            dict_in[key] = value
