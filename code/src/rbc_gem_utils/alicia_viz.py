import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec


class FluxOptimizationViz:
    def __init__(self, model, df_pcfva_alleles):
        """
        Instantiate an object to create flux optimization visualizations.

        Parameters
        ----------
        model : cobra.Model
            An instance of the non-protein-constrained model from which
            to obtain data about reactions.

        df_pcfva_alleles : pd.DataFrame
            DataFrame with the following columns: day, reactions, optimum,
            G6PD_alleles, minimum, maximum, range, and sample_id
        """
        self.model = model
        self.df_pcfva_alleles = df_pcfva_alleles
