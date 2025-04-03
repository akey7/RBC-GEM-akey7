import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from rbc_gem_utils import read_cobra_model


class FluxOptimizationViz:
    def __init__(self, model_filename, df_pcfva_alleles_filename):
        """
        Instantiate an object to create flux optimization visualizations.

        Parameters
        ----------
        model_filename : str
            Path to the non-protein-constraied COBRA model from which to
            obtain data about reactions.

        df_pcfva_alleles_filename : str
            Path to a csv DataFrame with the following columns: day,
            reactions, optimum, G6PD_alleles, minimum, maximum, range,
            and sample_id.
        """
        self.model = read_cobra_model(model_filename)
        df_pcfva_alleles = pd.read_csv(df_pcfva_alleles_filename)
        df_pcfva_alleles.drop("sample_id", axis=1, inplace=True)
        df_pcfva_alleles.set_index(
            ["G6PD_alleles", "day", "reactions", "optimum"], inplace=True
        )
        df_pcfva_alleles.sort_index(inplace=True)
        self.df_pcfva_alleles = df_pcfva_alleles
