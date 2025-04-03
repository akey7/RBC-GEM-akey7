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

    def get_subsystem_reactions_dict(self):
        """
        Return a dictionary with keys of subsystems and values as
        lists of reaction ids within each subsystem.

        Parameters
        ----------
        No parameters.

        Returns
        -------
        dict
            A dictionary with the keys and values specified above.
        """
        subsystem_reaction = {}
        for reaction in self.model.reactions:
            subsystem = reaction.subsystem
            if subsystem in subsystem_reaction:
                subsystem_reaction[subsystem].append(reaction.id)
            else:
                subsystem_reaction[subsystem] = [reaction.id]
        return subsystem_reaction

    def make_optimum_min_max_plot(
        self,
        day,
        reaction,
        optima=None,
        optimum_colors=None,
        save_filename=None,
        **kwargs,
    ):
        """
        Generate a plot with flux on the y axis and rank of flux range on the
        x axis, with subplots seprated on the x axis by copy number of the allele.
        Each plot is bands of flux ranges of increasingly stringent optimizations.

        If this method is called in a loop, many plots will be opened which
        may generate a warning. In this case, the caller should call plt.close()
        after using the resulting plot.

        Parameters
        ----------
        day : int
            Data from this day will be used to make the plot.

        reaction : str
            The reaction id of the underlying flux to plot.

        optima : List[float], optional
            A list of optima from the FVA to plot. If left as the default
            value of None, the list of `[0.0, 0.5, 0.9, 0.99]` will be used.

        optimum_color : List[str]
            Color specifications for each of the optima. If left as the default
            value of None, will use a spectrum of blues: 
            `["#87CEEB", "#3399CC", "#004C99", "#000080"]`

        save_filename : str
            Path to save an svg format file to. If left at the default value of
            None, no file is saved.

        Returns
        -------
        Figure
            Returns a figure that can be displayed.
        """
        optima = [0.0, 0.5, 0.9, 0.99] if not optima else optima
        optimum_colors = (
            ["#87CEEB", "#3399CC", "#004C99", "#000080"]
            if not optimum_colors
            else optimum_colors
        )
        fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, **kwargs)
        for allele_count, (ax_idx, ax) in zip(range(3), enumerate(axs)):
            for optimum, optimum_color in zip(optima, optimum_colors):
                multi = (allele_count, day, reaction, optimum)
                df_for_plot = (
                    self.df_pcfva_alleles.loc[multi, :].sort_values(by="range").copy()
                )
                y_mins = df_for_plot["minimum"]
                y_maxs = df_for_plot["maximum"]
                xs = np.arange(1, len(y_maxs) + 1)
                ax.fill_between(
                    xs,
                    y_mins,
                    y_maxs,
                    color=optimum_color,
                    label=f"{optimum*100:.0f}% Max NaKt",
                )
                ax.set_xlabel(allele_count, fontsize=14)
                ax.set_xticks([])
                if ax_idx == 0:
                    ax.set_ylabel("Flux (mmol/gDW/hr)", fontsize=14)
                if ax_idx == 2:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.suptitle(f"{reaction}, Day {day}", fontsize=18)
        if save_filename:
            plt.savefig(
                save_filename,
                dpi=300,
                transparent=False,
                bbox_inches="tight",
                pad_inches=0.5,
                format="svg",
            )
        return fig
