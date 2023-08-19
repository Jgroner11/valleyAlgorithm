from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath  # To prevent collision with pathlib.Path
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import math

def __rotate_vector(vector, theta):
    # theta is angle in radians
    # Create the rotation matrix
    norm = np.linalg.norm(vector)
    rotation_matrix = np.array([[math.cos(theta), -math.sin(theta)],
                                [math.sin(theta), math.cos(theta)]])

    # Multiply the vector with the rotation matrix
    vector_rotated = norm * rotation_matrix.dot(vector / norm)

    return vector_rotated

def __normalize_vector(vector: np.array, normalize_to: float) -> np.array:
    """Make `vector` norm equal to `normalize_to`
    vector: np.array
        Vector with 2 coordinates
    normalize_to: float
        A norm of the new vector
    Returns
    -------
    Vector with the same direction, but length normalized to `normalize_to`
    """

    vector_norm = np.linalg.norm(vector)

    return vector * normalize_to / vector_norm


def __orthogonal_vector(point: np.array, width: float,
                      normalize_to: Optional[float] = None) -> np.array:
    """Get orthogonal vector to a `point`
    point: np.array
        Vector with x and y coordinates of a point
    width: float
        Distance of the x-coordinate of the new vector from the `point` (in orthogonal direction)
    normalize_to: Optional[float] = None
        If a number is provided, normalize a new vector length to this number
    Returns
    -------
    Array with x and y coordinates of the vector, which is orthogonal to the vector
    from (0, 0) to the `point`
    """
    EPSILON = 0.000001

    x = width
    y = -x * point[0] / (point[1] + EPSILON)

    ort_vector = np.array([x, y])

    if normalize_to is not None:
        ort_vector = __normalize_vector(ort_vector, normalize_to)

    return ort_vector


def __draw_self_loop(
        point: np.array,
        ax: Optional[plt.Axes] = None,
        padding: float = 1.5,
        width: float = 0.3,
        plot_size: int = 10,
        linewidth=1.0,
        color: str = "k"
) -> plt.Axes:
    """Draw a loop from `point` to itself
    !Important! By "center" we assume a (0, 0) point. If your data is centered around a different
    point, it is strongly recommended to center it around zero. Otherwise, you will probably
    get ugly plots
    Parameters
    ----------
    point: np.array
        1D array with 2 coordinates of the point. Loop will be drawn from and to these coordinates.
    padding: float = 1.5
        Controls how the distance of the loop from the center. If `padding` > 1, the loop will be
        from the outside of the `point`. If `padding` < 1, the loop will be closer to the center
    width: float = 0.3
        Controls the width of the loop
    linewidth: float = 0.2
        Width of the line of the loop
    ax: Optional[matplotlib.pyplot.Axes]:
        Axis on which to draw a plot. If None, a new Axis is generated
    plot_size: int = 7
        Size of the plot sides in inches. Ignored if `ax` is provided
    color: str = "pink"
        Color of the arrow
    Returns
    -------
    Matplotlib axes with the self-loop drawn
    """

    if ax is None:
        ax = plt.gca() # Modded

    point_with_padding = padding * point

    ort_vector = __orthogonal_vector(point, width, normalize_to=width)

    first_anchor = ort_vector + point_with_padding
    second_anchor = -ort_vector + point_with_padding

    verts = [point, first_anchor, second_anchor, point]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]

    path = MplPath(verts, codes)


    patch = patches.FancyArrowPatch(
        path=path,
        lw=linewidth,
        linestyle = "solid",
        arrowstyle="-|>",
        color=color,
        alpha=None,
        mutation_scale=10  # arrowsize in draw_networkx_edges()
    )
    ax.add_patch(patch)

    return ax

def __get_node_radius_components(node_size, ax=None):
    # This function uses constants to account for changing node radius size in different environments
    if ax == None:
        ax = plt.gca()

    CONST = .0095
    fig_sz = ax.get_figure().get_size_inches()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax_sz = np.array([xlim[1] - xlim[0], ylim[1] - ylim[0]])

    return (CONST * np.sqrt(node_size) * ax_sz) / fig_sz

def draw_networkx_self_edges(graph: nx.graph, pos: dict, 
    padding: float = 1.5,
    ax: plt.Axes = None,
    edgelist=None,
    width=1.0,
    edge_color="k",
    style="solid",
    alpha=None,
    arrowstyle=None,
    arrowsize=10,
    node_size=300,
    ) -> plt.Axes:
    """Draw self edges
    graph: nx.Graph
        Graph, edges of which you want to draw
    pos: dict
        Dictionary, where keys are nodes and values are their positions. Can be obtained
        through networkx layout algorithms (e. g. nx.circular_layout())
    ax: plt.Axes
        Axis on which draw the edges
    padding: float
        length of self loop
    node_size: numpy.array() or float
        size of nodes
    Returns
    -------
    Axis with the edges drawn
    """

    if ax == None:
        ax = plt.gca()

    if edgelist == None:
        edgelist = [edge for edge in graph.edges() if edge[0] == edge[1]]

    for i, edge in enumerate(edgelist):
        point = np.array(pos[edge[0]])
        norm = np.linalg.norm(point)

        # point2 = point + (CONST * np.sqrt(node_size) * ax_sz * point) / (norm * fig_sz)
        point2 = point + __get_node_radius_components(node_size[edge[0]] if np.iterable(node_size) else node_size, ax=ax) * point / norm
        # plt.plot([point[0], point2[0]], [point[1], point2[1]])
        __draw_self_loop(point2, ax=ax, padding=padding, color=edge_color[i] if isinstance(edge_color, (list, np.ndarray)) else edge_color)

    return ax

# https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx
def draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph
    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items

def draw_networkx_self_edge_labels(
    G,
    pos,
    padding=1.5,
    node_size=300,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.
    padding
    node_size
    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    pos=dict(pos)
    CONST = .160

    for edge in edge_labels:
        point = np.array(pos[edge[0]])
        norm = np.linalg.norm(point)
        pos[edge[0]] = point + (__get_node_radius_components(node_size[edge[0]] if np.iterable(node_size) else node_size, ax=ax) * point / norm) + (padding * CONST * point / norm)

    return  nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    label_pos=label_pos,
    font_size=font_size,
    font_color=font_color,
    font_family=font_family,
    font_weight=font_weight,
    alpha=alpha,
    bbox=bbox,
    horizontalalignment=horizontalalignment,
    verticalalignment=verticalalignment,
    ax=ax,
    rotate=rotate,
    clip_on=clip_on)

def draw_networkx_labels(
    G,
    pos,
    node_size=300,
    labels=None,
    font_size=12,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    clip_on=True,
):
    """Draw node labels on the graph G.
    Parameters
    ----------
    G : graph
        A networkx graph
    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.
    labels : dictionary (default={n: n for n in G})
        Node labels in a dictionary of text labels keyed by node.
        Node-keys in labels should appear as keys in `pos`.
        If needed use: `{n:lab for n,lab in labels.items() if n in pos}`
    font_size : int (default=12)
        Font size for text labels
    font_color : string (default='k' black)
        Font color string
    font_weight : string (default='normal')
        Font weight
    font_family : string (default='sans-serif')
        Font family
    alpha : float or None (default=None)
        The text transparency
    bbox : Matplotlib bbox, (default is Matplotlib's ax.text default)
        Specify text box properties (e.g. shape, color etc.) for node labels.
    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}
    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.
    clip_on : bool (default=True)
        Turn on clipping of node labels at axis boundaries
    Returns
    -------
    dict
        `dict` of labels keyed on the nodes
    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> labels = nx.draw_networkx_labels(G, pos=nx.spring_layout(G))
    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html
    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_edge_labels
    """
    pos = dict(pos)
    for i in range(len(pos)):
        point = pos[i]
        norm = np.linalg.norm(point)
        rotated = __rotate_vector(point / norm, math.radians(-70))

        pos[i] = point + rotated * __get_node_radius_components(node_size[i] if np.iterable(node_size) else node_size, ax=ax) * 1.75

        # plt.plot([point[0], pos[i][0]], [point[1], pos[i][1]])

    return nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=font_size,
        font_color=font_color,
        font_family=font_family,
        font_weight=font_weight,
        alpha=alpha,
        bbox=bbox,
        horizontalalignment=horizontalalignment,
        verticalalignment=verticalalignment,
        ax=ax,
        clip_on=clip_on,
    )

    