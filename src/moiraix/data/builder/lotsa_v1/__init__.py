#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from ._base import LOTSADatasetBuilder
from .buildings_bench import Buildings900KDatasetBuilder, BuildingsBenchDatasetBuilder
from .chronos_synth import ChronosSyntheticDatasetBuilder
from .cloudops_tsf import CloudOpsTSFDatasetBuilder
from .cmip6 import CMIP6DatasetBuilder
from .era5 import ERA5DatasetBuilder
from .gift_eval_test import GiftEvalTestDatasetBuilder
from .gift_eval_train import (
    GiftEvalTrainSplitDatasetBuilder,
    GiftEvalTrainSplitEW0DatasetBuilder,
)
from .gluonts import GluonTSDatasetBuilder
from .gluonts_elec import GluonTSElecDatasetBuilder
from .gluonts_sub import GluonTSSubDatasetBuilder
from .gluonts_wiki import GluonTSWikiDatasetBuilder
from .largest import LargeSTDatasetBuilder
from .lib_city import LibCityDatasetBuilder
from .others import OthersLOTSADatasetBuilder
from .proenfo import ProEnFoDatasetBuilder
from .rmisc import RMISCDatasetBuilder
from .subseasonal import SubseasonalDatasetBuilder
from .synth_scale import (
    SynthBackboneV1DatasetBuilder,
    SynthMixedV1DatasetBuilder,
    SynthSarima2V1DatasetBuilder,
    SynthStressV1DatasetBuilder,
)
from .synth_stress import SynthStressDatasetBuilder
from .synth_trackb import (
    SynthBackboneDatasetBuilder,
    SynthSarima2DatasetBuilder,
    SynthSDEDatasetBuilder,
)

__all__ = [
    "LOTSADatasetBuilder",
    "Buildings900KDatasetBuilder",
    "BuildingsBenchDatasetBuilder",
    "CloudOpsTSFDatasetBuilder",
    "CMIP6DatasetBuilder",
    "ERA5DatasetBuilder",
    "GluonTSDatasetBuilder",
    "LargeSTDatasetBuilder",
    "LibCityDatasetBuilder",
    "OthersLOTSADatasetBuilder",
    "ProEnFoDatasetBuilder",
    "RMISCDatasetBuilder",
    "SubseasonalDatasetBuilder",
    "ChronosSyntheticDatasetBuilder",
    "GiftEvalTrainSplitDatasetBuilder",
    "GiftEvalTrainSplitEW0DatasetBuilder",
    "SynthStressDatasetBuilder",
    "SynthBackboneDatasetBuilder",
    "SynthSDEDatasetBuilder",
    "SynthSarima2DatasetBuilder",
    "SynthBackboneV1DatasetBuilder",
    "SynthStressV1DatasetBuilder",
    "SynthSarima2V1DatasetBuilder",
    "SynthMixedV1DatasetBuilder",
    "GluonTSSubDatasetBuilder",
    "GluonTSElecDatasetBuilder",
    "GluonTSWikiDatasetBuilder",
    "GiftEvalTestDatasetBuilder",
]
