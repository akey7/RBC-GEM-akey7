from itertools import chain
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

    def iterate_make_optimum_min_max_plots(self, fluxes_to_plot, flux_plots_path):
        """
        Iterate over nested lists of reaction names and plot the associated
        flux ranges with make_optimum_min_max_plot().

        Parameters
        ----------
        fluxes_to_plot : List[List[str]]
            The reactions to plot.

        flux_plots_path : Path
            Path to the folder to hold the resulting plots.

        Returns
        -------
        None
        """
        flat_list = list(chain(*fluxes_to_plot))
        for flux in flat_list:
            if flux:
                for day in [10, 23, 42]:
                    save_filename = flux_plots_path / f"{flux}_day_{day}.svg"
                    self.make_optimum_min_max_plot(
                        day, flux, figsize=(5, 2.5), save_filename=save_filename
                    )
                    plt.close()
                    print(save_filename, "saved")

    def make_flux_alleles_positions_for_day(self, day, fluxes_to_plot):
        """
        Used by alleles_day_flux_small_multiples() and shouldn't need to
        be called outside of instances of this class.

        Given a day and nested lists of fluxes to plot, generate the
        allele copy number, day, and reaction elements that will
        eventually be used to select from the self.df_pcfva_alleles
        DataFrame to generate small multiples of groups of reactions
        and their corresponding flux ranges.

        Parameters
        ----------
        day : int
            The day to create the indices for.

        fluxes_to_plot : List[List[str]]
            The nested lists of reaction ids that will be in the generated
            indices.

        Returns
        -------
        List[List[List[int, int, str]]]
            Lists of 3 of the 4 elements of the multi index to plot in each
            position.
        """
        flux_alleles_positions = []
        for row_idx, row in enumerate(fluxes_to_plot):
            flux_alleles_positions.append([])
            for reaction in row:
                if not reaction:
                    alleles_day_reaction = [[], [], []]
                else:
                    alleles_day_reaction = [
                        [0, day, reaction],
                        [1, day, reaction],
                        [2, day, reaction],
                    ]
                flux_alleles_positions[row_idx].append(alleles_day_reaction)
        return flux_alleles_positions

    def min_max_y_for_alleles_day_reaction(self, alleles_days_reactions):
        """
        Used by alleles_day_flux_small_multiples(). Should not need to be
        called outside of instances of this class.

        Given lists of alleles, day, and flux names, find the minimum and
        maximum extents for y limits on plots. Used to make small multiples
        of flux ranges.

        Parameters
        ----------
        alleles_days_reactions : List[List[int, int, str]]
            List of lists of an allele count, day number, and flux name.

        Returns
        -------
        Tuple[float, float]
            Minimum and maximum extent of the y values, respectively.
        """
        min_y = 0.0
        max_y = 0.0
        optima = [0.0, 0.5, 0.9, 0.99]
        for alleles_day_reaction in alleles_days_reactions:
            if alleles_day_reaction:
                alleles, day, flux = alleles_day_reaction
                for optimum in optima:
                    multi_index = (alleles, day, flux, optimum)
                    multi_index = (alleles, day, flux, optimum)
                    df = (
                        self.df_pcfva_alleles.loc[multi_index]
                        .sort_values(by="range")
                        .copy()
                    )
                    y_mins = df["minimum"]
                    y_maxs = df["maximum"]
                    min_y = min_y if min(y_mins) > min_y else min(y_mins)
                    max_y = max_y if max(y_maxs) < max_y else max(y_maxs)
        return min_y, max_y

    def alleles_day_flux_small_multiples(
        self, day, flux_group, title, flux_plots_path=None, figsize=(15, 8)
    ):
        """
        Create a grid of plots of flux ranges related in some way and
        specified as a nested lists of lists as a flux group. The rows
        and columns of the list of lists specify the fluxes to plot
        in those positions.

        If this method is called in a Jupyter notebook, the output should
        render after its invocation, and might be repeated twice. The second
        repeat can be supressed with a `;` after the call to this method.

        Parameters
        ----------
        day : int
            A day to plot the fluxes for.

        flux_group : List[List[str]]
            Names and positions of the fluxes to plot.

        title : str
            Title to place over all the plots.

        flux_plots_path : Path, optional
            Path to save the plot to. If left as the default of None, 
            will not save the resulting plot.

        figsize : Tuple[float, float]
            Figsize of the entire plot. Defaults to (15, 8).
        """
        flux_alleles_positions = self.make_flux_alleles_positions_for_day(
            day, flux_group
        )
        fig = plt.figure(figsize=figsize)
        zero_one_two_padding = 4
        rows = len(flux_alleles_positions)
        cols = len(flux_alleles_positions[0]) * zero_one_two_padding
        optima = [0.0, 0.5, 0.9, 0.99]
        optima_colors = ["#87CEEB", "#3399CC", "#004C99", "#000080"]
        gs = GridSpec(rows, cols, wspace=0)
        for gs_row, day_flux_alleles_positions_row in zip(
            range(rows), flux_alleles_positions
        ):
            for gs_col_left, alleles_day_flux_group in zip(
                range(0, cols, zero_one_two_padding), day_flux_alleles_positions_row
            ):
                ylim_min, ylim_max = self.min_max_y_for_alleles_day_reaction(
                    alleles_day_flux_group
                )
                for group_idx in range(zero_one_two_padding):
                    if group_idx == zero_one_two_padding - 1:
                        ax = fig.add_subplot(gs[gs_row, gs_col_left + group_idx])
                        ax.axis("off")
                    elif alleles_day_flux_group[group_idx]:
                        alleles, day, flux = alleles_day_flux_group[group_idx]
                        ax = fig.add_subplot(gs[gs_row, gs_col_left + group_idx])
                        ax.set_ylim(ylim_min, ylim_max)
                        if group_idx == 0:
                            ax.tick_params(right=False)
                            ax.spines["right"].set_visible(False)
                            ax.set_title(flux)
                        elif group_idx == 1:
                            ax.tick_params(left=False, right=False)
                            ax.spines["left"].set_visible(False)
                            ax.spines["right"].set_visible(False)
                            ax.set_yticklabels([])
                        elif group_idx == 2:
                            ax.spines["left"].set_visible(False)
                            ax.tick_params(left=False)
                            ax.set_yticklabels([])
                        for optimum, optimum_color in zip(optima, optima_colors):
                            multi_index = (alleles, day, flux, optimum)
                            df = (
                                self.df_pcfva_alleles.loc[multi_index]
                                .sort_values(by="range")
                                .copy()
                            )
                            y_mins = df["minimum"]
                            y_maxs = df["maximum"]
                            xs = np.arange(1, len(y_maxs) + 1)
                            ax.fill_between(
                                xs,
                                y_mins,
                                y_maxs,
                                color=optimum_color,
                                label=f"{optimum*100:.0f}% Max NaKt",
                            )
                            ax.set_xlabel(alleles, fontsize=14)
                            ax.set_xticks([])
        fig.suptitle(f"{title} Storage Day {day}", fontsize=24)
        fig.tight_layout()
        clean_title = title.lower().strip().replace(" ", "_")
        if flux_plots_path:
            flux_small_mutiples_plot_filename = (
                flux_plots_path / f"{clean_title}_small_mutiples_day_{day}.svg"
            )
            plt.savefig(
                flux_small_mutiples_plot_filename,
                dpi=300,
                transparent=False,
                bbox_inches="tight",
                pad_inches=0.5,
                format="svg",
            )
