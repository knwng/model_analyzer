# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from model_analyzer.config.input.config_field import ConfigField
from model_analyzer.config.input.config_primitive import ConfigPrimitive


class ConfigCommandExperiment(ConfigCommandProfile):
    """ 
    Extended ConfigCommandProfile with extra options for experiment algorithm configuration
    """

    def _fill_config(self):
        super()._fill_config()
        self._add_config(
            ConfigField('radius',
                        field_type=ConfigPrimitive(int),
                        flags=['--radius'],
                        default_value=2,
                        description='The size of the neighborhood radius'))
        self._add_config(
            ConfigField('magnitude',
                        field_type=ConfigPrimitive(int),
                        flags=['--magnitude'],
                        default_value=2,
                        description='The size of each step'))
        self._add_config(
            ConfigField(
                'min_initialized',
                field_type=ConfigPrimitive(int),
                flags=['--min-initialized'],
                default_value=2,
                description=
                'The minimum number of datapoints needed in a neighborhood before stepping'
            ))
        self._add_config(
            ConfigField('step_mode',
                        field_type=ConfigPrimitive(str),
                        flags=['--step-mode'],
                        choices=['Unit', 'Variable'],
                        default_value="Unit",
                        description='Unit vs Variable'))
