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
import os
import tempfile
import uuid
from enum import Enum
from math import log10
from typing import Any, Dict, Optional, Union
import pandas as pd
from pm4py.util import exec_utils, constants


class Parameters(Enum):
    FORMAT = "format"
    NODE_HEIGHT = "node_height"
    NODE_WIDTH = "node_width"
    EDGE_PENWIDTH = "edge_penwidth"
    MAX_VARIANTS = "max_variants"
    ALIGNMENT_CRITERIA = "alignment_criteria"
    MIN_HORIZONTAL_DISTANCE = "min_horizontal_distance"
    MAX_HORIZONTAL_DISTANCE = "max_horizontal_distance"
    LAYOUT_EXT_MULTIPLIER = "layout_ext_multiplier"
    SHOW_LEGEND = "show_legend"
    ENABLE_GRAPH_TITLE = "enable_graph_title"
    GRAPH_TITLE = "graph_title"


def apply(
        variants_df: pd.DataFrame,
        parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
):
    if parameters is None:
        parameters = {}

    # Extract parameters
    format = exec_utils.get_param_value(Parameters.FORMAT, parameters, "png")
    node_height = exec_utils.get_param_value(Parameters.NODE_HEIGHT, parameters, 0.85)
    node_width = exec_utils.get_param_value(Parameters.NODE_WIDTH, parameters, 0.85)
    edge_penwidth = exec_utils.get_param_value(Parameters.EDGE_PENWIDTH, parameters, 1.0)
    max_variants = exec_utils.get_param_value(Parameters.MAX_VARIANTS, parameters, 5)
    alignment_criteria = exec_utils.get_param_value(Parameters.ALIGNMENT_CRITERIA, parameters, "start")
    min_horizontal_distance = exec_utils.get_param_value(Parameters.MIN_HORIZONTAL_DISTANCE, parameters, 1.5)
    max_horizontal_distance = exec_utils.get_param_value(Parameters.MAX_HORIZONTAL_DISTANCE, parameters, 4.5)
    layout_ext_multiplier = exec_utils.get_param_value(Parameters.LAYOUT_EXT_MULTIPLIER, parameters, 75)
    enable_graph_title = exec_utils.get_param_value(Parameters.ENABLE_GRAPH_TITLE, parameters,
                                                    constants.DEFAULT_ENABLE_GRAPH_TITLES)
    graph_title = exec_utils.get_param_value(Parameters.GRAPH_TITLE, parameters, "Process Variants Paths and Durations")

    # Required column names
    variant_column = "@@variant_column"
    variant_count = "@@variant_count"
    index_column = "@@index_in_trace"
    flow_time_column = "@@flow_time"
    activity_key = "concept:name"
    activity_key_2 = "concept:name_2"

    # Sort variants from most frequent to least frequent and pick top N
    unique_variants = variants_df[[variant_column, variant_count]].drop_duplicates()
    unique_variants = unique_variants.sort_values(variant_count, ascending=False)
    top_variants = unique_variants.head(max_variants)[variant_column].tolist()
    filtered_df = variants_df[variants_df[variant_column].isin(top_variants)]

    # Temporary .gv and output file
    output_file_gv = tempfile.NamedTemporaryFile(suffix=".gv")
    output_file_gv.close()
    output_file_img = tempfile.NamedTemporaryFile(suffix="." + format)
    output_file_img.close()

    # For distance normalization
    max_flow_time = filtered_df[flow_time_column].max()

    # Assign each variant a color
    variant_colors = {
        variant: f"#{hash(str(variant)) % 0xffffff:06x}"
        for variant in top_variants
    }

    # Build the GraphViz lines
    lines = ["graph G {"]

    if enable_graph_title:
        lines.append(
            f'  label=<<FONT POINT-SIZE="20">{graph_title}</FONT>>;'
            '  labelloc="top";'
        )

    lines.append('  layout=neato;')
    lines.append('  splines=true;')

    # Store the computed positions of each activity per variant
    variant_y_pos = {}
    variant_node_positions = {}

    # Assign y-coordinates to variants
    total_variants = len(top_variants)
    for i, variant in enumerate(top_variants):
        y_coord = (total_variants - i) * layout_ext_multiplier
        variant_y_pos[variant] = y_coord
        variant_node_positions[variant] = {}

    # Calculate positions starting from 0 for all variants
    for variant in top_variants:
        vdf = filtered_df[filtered_df[variant_column] == variant].sort_values(index_column)
        for _, row in vdf.iterrows():
            src_activity = row[activity_key]
            tgt_activity = row[activity_key_2]
            flow_time = row[flow_time_column]
            src_idx = row[index_column]

            if src_activity not in variant_node_positions[variant]:
                if src_idx == 0:
                    variant_node_positions[variant][src_activity] = 0
                else:
                    continue  # Skip if source activity is missing (should not happen with sorted data)

            # Convert flow_time to a distance
            if flow_time == 0:
                distance = min_horizontal_distance
            else:
                norm_time = min(1.0, log10(1 + flow_time) / log10(1 + max_flow_time))
                distance = min_horizontal_distance + norm_time * (max_horizontal_distance - min_horizontal_distance)

            x_old = variant_node_positions[variant][src_activity]
            x_new = x_old + distance * layout_ext_multiplier
            variant_node_positions[variant][tgt_activity] = x_new

    # Apply shifts based on alignment criteria
    if alignment_criteria == "start":
        # No shift needed
        pass
    elif alignment_criteria == "end":
        for variant in top_variants:
            # Find the position of the last activity (maximum position)
            last_pos = max(variant_node_positions[variant].values())
            shift = -last_pos
            for activity in variant_node_positions[variant]:
                variant_node_positions[variant][activity] += shift
    else:
        # alignment_criteria is the name of some activity
        variants_with_activity = [
            variant for variant in top_variants
            if alignment_criteria in variant_node_positions[variant]
        ]
        variants_without_activity = [
            variant for variant in top_variants
            if alignment_criteria not in variant_node_positions[variant]
        ]

        if variants_without_activity:
            missing_variants = ", ".join(str(v) for v in variants_without_activity)
            raise ValueError(
                f"Alignment activity '{alignment_criteria}' not found in variants: {missing_variants}. "
                "All variants must contain the alignment activity."
            )

        for variant in variants_with_activity:
            align_pos = variant_node_positions[variant][alignment_criteria]
            shift = -align_pos
            for activity in variant_node_positions[variant]:
                variant_node_positions[variant][activity] += shift

    # Create node labels for each variant
    for i, variant in enumerate(top_variants):
        count = unique_variants[unique_variants[variant_column] == variant][variant_count].iloc[0]
        label_text = f"Variant\n{i + 1}\n({count} cases)"
        label_node_id = f"label_{uuid.uuid4().hex[:12]}"
        lines.append(
            f'  {label_node_id} [label="{label_text}", shape=none, '
            f'fontsize="10pt", pos="-60,{variant_y_pos[variant]}!", fixedsize=true];'
        )

    # Create actual activity nodes and edges
    for variant in top_variants:
        y_pos = variant_y_pos[variant]
        vdf = filtered_df[filtered_df[variant_column] == variant].sort_values(index_column)
        color = variant_colors[variant]
        activity_node_ids = {}

        # Nodes
        for activity, x_pos in variant_node_positions[variant].items():
            node_id = f"n{uuid.uuid4().hex[:12]}"
            activity_node_ids[activity] = node_id
            label = activity.replace(" ", "\\n")  # literal backslash-n for Graphviz

            lines.append(
                f'  {node_id} [label="{label}", shape=box, style="filled,rounded", '
                f'fillcolor="{color}", width={node_width}, height={node_height}, '
                f'pos="{x_pos},{y_pos}!", fontsize="8pt", fixedsize=true];'
            )

        # Edges
        for _, row in vdf.iterrows():
            src = row[activity_key]
            tgt = row[activity_key_2]
            ftime = row[flow_time_column]

            if ftime < 60:
                label_time = f"{ftime:.1f}s"
            elif ftime < 3600:
                label_time = f"{ftime / 60:.1f}m"
            elif ftime < 86400:
                label_time = f"{ftime / 3600:.1f}h"
            else:
                label_time = f"{ftime / 86400:.1f}d"

            src_id = activity_node_ids[src]
            tgt_id = activity_node_ids[tgt]
            lines.append(
                f'  {src_id} -- {tgt_id} [label="{label_time}", fontsize="7pt", '
                f'color="{color}", penwidth={edge_penwidth}];'
            )

    lines.append("}")

    # Write .gv file
    with open(output_file_gv.name, "w") as f:
        f.write("\n".join(lines))

    # Use neato -n2 to respect exact coordinates
    os.system(f'neato -n2 -T{format} "{output_file_gv.name}" > "{output_file_img.name}"')

    return output_file_img.name
