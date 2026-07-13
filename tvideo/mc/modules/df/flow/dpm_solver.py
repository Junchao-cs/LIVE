# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from tqdm import tqdm
import numpy as np

from . import gaussian_diffusion as gd

class NoiseScheduleFlow:
    def __init__(
        self,
        schedule="discrete_flow",
    ):
        """Create a wrapper class for the forward SDE (EDM type)."""
        self.T = 1
        self.t0 = 0.001
        self.schedule = schedule  # ['continuous', 'discrete_flow']
        self.total_N = 1000

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        return torch.log(self.marginal_alpha(t))

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return 1 - t

    @staticmethod
    def marginal_std(t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return t

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = torch.log(self.marginal_std(t))
        return log_mean_coeff - log_std

    @staticmethod
    def inverse_lambda(lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        return torch.exp(-lamb)

    def edm_sigma(self, t):
        return self.marginal_std(t) / self.marginal_alpha(t)

    def edm_inverse_sigma(self, edmsigma):
        sigma = edmsigma
        lambda_t = torch.log(1 / sigma)
        t = self.inverse_lambda(lambda_t)
        return t
