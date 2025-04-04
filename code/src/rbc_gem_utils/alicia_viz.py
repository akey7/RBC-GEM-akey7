from itertools import chain
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rbc_gem_utils import read_cobra_model


class CorrelationsViz:
    def __init__(self, df_flux_abundance_correlation_filename):
        """
        Instantiate a class to make plots correlations

        Parameters
        ----------
        df_flux_abundance_correlation_filename
            Full path to the csv that holds the flux-anundance
            correlation data.
        """
        self.df_flux_abundance_correlation = pd.read_csv(
            df_flux_abundance_correlation_filename
        )

    def plot_correlations(
        self,
        df,
        vertical_lines,
        ax=None,
        histx=True,
        histy=True,
        colorbar=True,
        **kwargs,
    ):
        """
        Create a u-plot with horizontal and vertical historgrams above and
        to the side.

        This method is meant to be called by other methods in this class,
        and not by methods outside this class. See
        plot_flux_abundance_correlations() as an example.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with `rho` and `neg_log10_adj_p_value` columns to plot.

        vertical_lines
            Vertical lines to place on the u plot as provided by the caller.

        ax : Axis, optional
            Axis on which to draw the plot.

        histx : bool, optional
            Defaults to True, which draws a histogram on the x-axis above
            the plot.

        histy : bool, optional
            Defaults to True, which draws a histogram on the y-axis
            to the right of the plot.

        colorbar : bool, optional
            Defaults to true which plots a colorbar legend on the plot.

        **kwargs
            Various plotting options accessed by this method. If left unspecified
            this method will use default values.
        """
        # Define figure if no axes provided.
        scatter_inch = kwargs.get("scatter_inch", 5.0)
        hist_inch = kwargs.get("hist_inch", 1.0)
        hist_pad = kwargs.get("hist_pad", 0.25)
        if ax is None:
            _, ax = plt.subplots(
                nrows=1,
                ncols=1,
                figsize=(
                    scatter_inch + (hist_inch + hist_pad if histy else 0),
                    scatter_inch + (hist_inch + hist_pad if histx else 0),
                ),
            )
        xy = {"x": "rho", "y": "neg_log10_adj_p_value"}
        limits = {
            "x": (kwargs.get("xmin", -1.0), kwargs.get("xmax", 1.0)),
            "y": (kwargs.get("ymin", 0.0), kwargs.get("ymax", df[xy["y"]].max())),
        }
        pads = {
            axis: kwargs.get(f"{axis}pad", (limits[axis][1] - limits[axis][0]) / 2 / 20)
            for axis in list(xy)
        }
        cmap = kwargs.get("cmap", "viridis")
        zorder = kwargs.get("zorder", 2)
        edgecolor = kwargs.get("edgecolor", "black")
        edgewidth = kwargs.get("edgewidth", 0.5)
        scatter = ax.scatter(
            xy["x"],
            xy["y"],
            data=df,
            c=kwargs.get("c", xy["y"]),
            s=kwargs.get("s", 40),
            alpha=0.5,
            zorder=zorder,
            edgecolor=edgecolor,
            linewidth=edgewidth,
            cmap=mpl.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap,
            norm=mpl.colors.Normalize(
                vmin=limits["y"][0] - pads["y"], vmax=limits["y"][1] + pads["y"]
            ),
        )
        ax.set_xlabel(r"Spearman Correlation $(\rho)$", fontdict={"size": "xx-large"})
        ax.set_ylabel("-log$_{10}$(adj p-value)", fontdict={"size": "xx-large"})
        ax.set_xlim((limits["x"][0] - pads["x"], limits["x"][1] + pads["x"]))
        ax.set_ylim((limits["y"][0] - pads["y"], limits["y"][1] + pads["y"]))

        major_ticks = {axis: kwargs.get(f"{axis}tick_major") for axis in list(xy)}
        minor_ticks = {
            axis: kwargs.get(
                f"{axis}tick_minor",
                major_ticks[axis] / 2 if major_ticks[axis] is not None else None,
            )
            for axis in list(xy)
        }
        for axis in list(xy):
            if major_ticks[axis] is not None:
                getattr(ax, f"{axis}axis").set_major_locator(
                    mpl.ticker.MultipleLocator(major_ticks[axis])
                )
            if minor_ticks[axis] is not None:
                getattr(ax, f"{axis}axis").set_minor_locator(
                    mpl.ticker.MultipleLocator(minor_ticks[axis])
                )
            ax.tick_params(axis=axis, labelsize="large")

        if vertical_lines:
            for lineval, (lineprops, textprops) in vertical_lines.items():
                if lineprops:
                    ax.vlines(
                        x=lineval,
                        ymin=limits["y"][0] - pads["y"],
                        ymax=limits["y"][1] + pads["y"],
                        **lineprops,
                    )
                if textprops:
                    ax.text(
                        x=lineval + pads["x"] / 2, transform=ax.transData, **textprops
                    )

        if kwargs.get("grid", False):
            ax.grid(True, **dict(which="both", alpha=0.75))

        if colorbar:
            cax = ax.inset_axes(
                [
                    limits["x"][0] - pads["x"],  # lower left corner xpos
                    limits["y"][0] - pads["y"],  # lower left corner ypos
                    pads["x"],  # width of colorbar
                    limits["y"][1]
                    + pads["y"]
                    + pads[
                        "y"
                    ],  # height of colorbar, need extra ypad to make up for lowering ypos
                ],
                transform=ax.transData,
            )
            cbar = ax.get_figure().colorbar(scatter, cax=cax)
            cax.set_ylim((limits["y"][0] - pads["y"], limits["y"][1] + pads["y"]))
            cax.set_xticks([])
            cax.set_yticks([])

        ax_histx = None
        ax_histy = None
        if histx or histy:
            divider = make_axes_locatable(ax)
            # Histogram axes
            ax_histx = (
                divider.append_axes("top", hist_inch, pad=hist_pad, sharex=ax)
                if histx
                else None
            )
            ax_histy = (
                divider.append_axes("right", hist_inch, pad=hist_pad, sharey=ax)
                if histy
                else None
            )

            for axis, ax_hist in zip(list(xy), [ax_histx, ax_histy]):
                if ax_hist is None:
                    continue
                binwidth = kwargs.get(
                    f"{axis}binwidth",
                    (
                        minor_ticks[axis]
                        if minor_ticks[axis] is not None
                        else major_ticks[axis]
                    ),
                )
                counts, bins, patches = ax_hist.hist(
                    df[xy[axis]],
                    bins=np.arange(
                        limits[axis][0], limits[axis][1] + binwidth, binwidth
                    ),
                    orientation="vertical" if axis == "x" else "horizontal",
                    zorder=zorder,
                    edgecolor=edgecolor,
                    linewidth=edgewidth,
                )
                other = "y" if axis == "x" else "x"
                ax_hist.tick_params(
                    axis=axis, **{f"label{'bottom' if axis == 'x' else 'left'}": False}
                )
                ax_hist.tick_params(axis=other, labelsize="large")
                getattr(ax_hist, f"set_{other}label")("Frequency", fontsize="large")

                tick_major_int = kwargs.get(f"hist{axis}_{other}tick_major")
                if tick_major_int is not None:
                    getattr(ax_hist, f"{other}axis").set_major_locator(
                        mpl.ticker.MultipleLocator(tick_major_int)
                    )
                    getattr(ax_hist, f"{other}axis").set_minor_locator(
                        mpl.ticker.MultipleLocator(tick_major_int / 2)
                    )
                getattr(ax_hist, f"set_{other}lim")((0, max(counts) * 1.1))
                if kwargs.get("grid", False):
                    ax_hist.grid(True, **dict(which="both", alpha=0.75))

                if vertical_lines and (axis == "x" and ax_hist is not None):
                    for lineval, (lineprops, _) in vertical_lines.items():
                        if lineprops:
                            ax_hist.vlines(
                                x=lineval, ymin=0.0, ymax=max(counts) * 1.1, **lineprops
                            )

        return ax, ax_histx, ax_histy

    def plot_flux_abundance_correlations(
        self, histx=True, histy=True, colorbar=True, save_filename=None, **kwargs
    ):
        """
        Plot the correlations in self.df_flux_abundance_correlation by calling
        plot_correlations() with appropriate defaults. If called with no
        parameters, there will be reasonsable defaults.

        Parameters
        ----------
        histx : bool, optional
            Defaults to True, which draws a histogram on the x-axis above
            the plot.

        histy : bool, optional
            Defaults to True, which draws a histogram on the y-axis
            to the right of the plot.

        colorbar : bool, optional
            Defaults to true which plots a colorbar legend on the plot.

        save_filename : str, optional
            If specified, saves an svg to this path.

        **kwargs : optional
            If additional arguments are provided, they can further customize
            the plot, but these areguments are optional.
        """
        scatter_inch = kwargs.get("scatter_inch", 5.0)
        hist_inch = kwargs.get("hist_inch", 1.0)
        hist_pad = kwargs.get("hist_pad", 0.5)
        nrows, ncols = (1, 1)
        expression_dep_rho_lb = 0.8
        expression_cor_rho_lb = 0.5
        ypos = 4
        ww = 11
        rotation = 90
        fontsize = "large"
        linewidth = 2
        vertical_lines = {
            expression_dep_rho_lb: (
                dict(color="black", linestyle="-", linewidth=linewidth),
                dict(
                    y=ypos,
                    s="\n".join(textwrap.wrap("Expression dependent", width=ww)),
                    rotation=rotation,
                    fontsize=fontsize,
                ),
            ),
            expression_cor_rho_lb: (
                dict(color="xkcd:dark grey", linestyle="--", linewidth=linewidth),
                dict(
                    y=ypos,
                    s="\n".join(textwrap.wrap("Expression correlated", width=ww)),
                    rotation=rotation,
                    fontsize=fontsize,
                ),
            ),
            0.0: (
                dict(),
                dict(
                    y=ypos + 50.0,
                    s="\n".join(textwrap.wrap("Expression independent", width=ww)),
                    rotation=rotation,
                    fontsize=fontsize,
                ),
            ),
        }
        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(
                (scatter_inch + (hist_inch + hist_pad if histx else 0)) * ncols,
                (scatter_inch + (hist_inch + hist_pad if histy else 0)) * nrows,
            ),
        )
        ax_scatter, ax_histx, ax_histy = self.plot_correlations(
            self.df_flux_abundance_correlation,
            ax=ax,
            histx=histx,
            histy=histy,
            colorbar=True,
            vertical_lines=vertical_lines,
            xbinwidth=0.1,
            ybinwidth=10,
            **kwargs,
        )
        # ax_scatter.set_title(
        #     f"Correlates between Flux and Abundance",
        #     fontsize="x-large",
        # )
        fig.suptitle("Correlates Between Flux and Abundance", fontsize=18)
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
