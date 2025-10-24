'''
    PM4Py â€“ A Process Mining Library for Python
Copyright (C) 2024 Process Intelligence Solutions UG (haftungsbeschrÃ¤nkt)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see this software project's root or
visit <https://www.gnu.org/licenses/>.

Website: https://processintelligence.solutions
Contact: info@processintelligence.solutions
'''
from pm4py.util import constants
from pm4py.util import vis_utils, exec_utils
import shutil
from pm4py.visualization.variants_duration.variants import classic
from enum import Enum
from typing import Optional, Dict, Any, Union
from pm4py.objects.log.obj import EventLog
import pandas as pd


class Variants(Enum):
    CLASSIC = classic


def apply(log: Union[EventLog, pd.DataFrame], variant=Variants.CLASSIC, parameters: Optional[Dict[Any, Any]] = None):
    """
    Comparative visualization of process variants based on the average time between the different steps

    Parameters
    -----------------
    log
        Event log
    variant
        Variant of the algorithm to be used (Variants.CLASSIC)
    parameters
        Variant-specific parameters, including:
        - Parameters.ALIGNMENT_CRITERIA => specifies if variants should be aligned on the "start", "end", or a specific activity.
        - Parameters.MAX_VARIANTS => max variants to visualize in the graph
        - Parameters.FORMAT => the format of the visualization
        - Parameters.NODE_HEIGHT, Parameters.NODE_WIDTH, Parameters.EDGE_PENWIDTH, Parameters.MIN_HORIZONTAL_DISTANCE,
            Parameters.MAX_HORIZONTAL_DISTANCE, Parameters.LAYOUT_EXT_MULTIPLIER, Parameters.SHOW_LEGEND,
            Parameters.ENABLE_GRAPH_TITLE, Parameters.GRAPH_TITLE

    """
    if parameters is None:
        parameters = {}

    import pm4py
    var_paths_durs = pm4py.get_variants_paths_duration(log)

    return exec_utils.get_variant(variant).apply(var_paths_durs, parameters)


def view(figure: str):
    """
    Views the dotted chart on the screen

    Parameters
    ---------------
    figure
        Path to the dotted chart
    """
    if constants.DEFAULT_ENABLE_VISUALIZATIONS_VIEW:
        if constants.DEFAULT_GVIZ_VIEW == "matplotlib_view":
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            img = mpimg.imread(figure)
            plt.axis("off")
            plt.tight_layout(pad=0, w_pad=0, h_pad=0)
            plt.imshow(img)
            plt.show()
            return

        if vis_utils.check_visualization_inside_jupyter():
            vis_utils.view_image_in_jupyter(figure)
        else:
            vis_utils.open_opsystem_image_viewer(figure)


def save(figure: str, output_file_path: str):
    """
    Saves the dotted chart to a specified path

    Parameters
    ----------------
    figure
        Current path to the dotted chart
    output_file_path
        Destination path
    """
    shutil.copyfile(figure, output_file_path)
    return ""


def serialize(figure: str):
    """
    Performs the serialization of the dotted chart visualization

    Parameters
    -----------------
    figure
        Current path to the dotted chart
    """
    with open(figure, "rb") as f:
        return f.read()


def matplotlib_view(figure: str):
    """
    Views the dotted chart on the screen using Matplotlib

    Parameters
    ---------------
    figure
        Path to the dotted chart
    """
    if constants.DEFAULT_ENABLE_VISUALIZATIONS_VIEW:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        img = mpimg.imread(figure)
        plt.imshow(img)
        plt.show()
