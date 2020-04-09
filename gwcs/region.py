# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
from collections import OrderedDict
import numpy as np


class Region(metaclass=abc.ABCMeta):

    """
    Base class for regions.

    Parameters
    ----------
    rid : int or str
        region ID
    coordinate_frame : `~gwcs.coordinate_frames.CoordinateFrame`
        Coordinate frame in which the region is defined.
    """

    def __init__(self, rid, coordinate_frame):
        self._coordinate_system = coordinate_frame
        self._rid = rid

    @abc.abstractmethod
    def __contains__(self, x, y):
        """
        Determines if a pixel is within a region.

        Parameters
        ----------
        x, y : float
            x , y values of a pixel

        Returns
        -------
        True or False

        Subclasses must define this method.
        """

    def scan(self, mask):
        """
        Sets mask values to region id for all pixels within the region.
        Subclasses must define this method.

        Parameters
        ----------
        mask : ndarray
            An array with the shape of the mask to be uised in `~gwcs.selector.RegionsSelector`.

        Returns
        -------
        mask : ndarray
            An array where the value of the elements is the region ID.
            Pixels which are not included in any region are marked with 0 or "".
        """


class Polygon(Region):

    """
    Represents a 2D polygon region with multiple vertices.

    Parameters
    ----------
    rid : str
         polygon id
    vertices : list of (x,y) tuples or lists
         The list is ordered in such a way that when traversed in a
         counterclockwise direction, the enclosed area is the polygon.
         The last vertex must coincide with the first vertex, minimum
         4 vertices are needed to define a triangle
    coord_frame : str or `~gwcs.coordinate_frames.CoordinateFrame`
        Coordinate frame in which the polygon is defined.

    """

    def __init__(self, rid, vertices, coord_frame="detector"):
        if len(vertices) < 4:
            raise ValueError("Expected vertices to be "
                             "a list of minimum 4 tuples (x,y)")
        super(Polygon, self).__init__(rid, coord_frame)

        self._vertices = np.asarray(vertices)
        self._bbox = self._get_bounding_box()
        self._scan_line_range = list(range(self._bbox[1], self._bbox[3] + self._bbox[1] + 1))
        # constructs a Global Edge Table (GET) in bbox coordinates
        self._GET = self._construct_ordered_GET()

    def _get_bounding_box(self):
        x = self._vertices[:, 0].min()
        y = self._vertices[:, 1].min()
        w = self._vertices[:, 0].max() - x
        h = self._vertices[:, 1].max() - y
        return (x, y, w, h)

    def _construct_ordered_GET(self):
        """
        Construct a Global Edge Table (GET)

        The GET is an OrderedDict. Keys are scan  line numbers,
        ordered from bbox.ymin to bbox.ymax, where bbox is the
        bounding box of the polygon.
        Values are lists of edges for which edge.ymin==scan_line_number.

        Returns
        -------
        GET: OrderedDict
            {scan_line: [edge1, edge2]}
        """
        # edges is a list of Edge objects which define a polygon
        # with these vertices
        edges = self.get_edges()
        GET = OrderedDict.fromkeys(self._scan_line_range)
        ymin = np.asarray([e._ymin for e in edges])
        for i in self._scan_line_range:
            ymin_ind = (ymin == i).nonzero()[0]
            if ymin_ind.any():
                GET[i] = [edges[ymin_ind[0]]]
                for j in ymin_ind[1:]:
                    GET[i].append(edges[j])
        return GET

    def get_edges(self):
        """
        Create a list of Edge objects from vertices
        """
        return [Edge(name='E{}'.format(i - 1), start=self._vertices[i - 1], stop=self._vertices[i])
                for i in range(1, len(self._vertices))
                ]

    def scan(self, data):
        """
        This is the main function which scans the polygon and creates the mask

        Parameters
        ----------
        data : array
            the mask array
            it has all zeros initially, elements within a region are set to
            the region's ID

        Algorithm:
        - Set the Global Edge Table (GET)
        - Set y to be the smallest y coordinate that has an entry in GET
        - Initialize the Active Edge Table (AET) to be empty
        - For each scan line:
          1. Add edges from GET to AET for which ymin==y
          2. Remove edges from AET fro which ymax==y
          3. Compute the intersection of the current scan line with all edges in the AET
          4. Sort on X of intersection point
          5. Set elements between pairs of X in the AET to the Edge's ID

        """
        # TODO: 1.This algorithm does not mark pixels in the top row and left most column.
        # Pad the initial pixel description on top and left with 1 px to prevent this.
        # 2. Currently it uses intersection of the scan line with edges. If this is
        # too slow it should use the 1/m increment (replace 3 above) (or the increment
        # should be removed from the GET entry).
        if self._bbox[2] <= 0:
            return data

        y = np.min(list(self._GET.keys()))
        AET = []
        scline = self._scan_line_range[-1]
        while y <= scline:
            AET = self.update_AET(y, AET)
            scan_line = Edge('scan_line', start=[self._bbox[0], y],
                             stop=[self._bbox[0] + self._bbox[2], y])
            x = [np.ceil(e.compute_AET_entry(scan_line)[1]) for e in AET if e is not None]
            xnew = np.asarray(np.sort(x), dtype=np.int)
            for i, j in zip(xnew[::2], xnew[1::2]):
                data[y][i:j + 1] = self._rid
            y = y + 1
        return data

    def update_AET(self, y, AET):
        """
        Update the Active Edge Table (AET)

        Add edges from GET to AET for which ymin of the edge is
        equal to the y of the scan line.
        Remove edges from AET for which ymax of the edge is
        equal to y of the scan line.

        """
        edge_cont = self._GET[y]
        if edge_cont is not None:
            for edge in edge_cont:
                if edge._start[1] != edge._stop[1] and edge._ymin == y:
                    AET.append(edge)
        for edge in AET[::-1]:
            if edge is not None:
                if edge._ymax == y:
                    AET.remove(edge)
        return AET

    def __contains__(self, px):
        """even-odd algorithm or smth else better sould be used"""
        return px[0] >= self._bbox[0] and px[0] <= self._bbox[0] + self._bbox[2] and \
            px[1] >= self._bbox[1] and px[1] <= self._bbox[1] + self._bbox[3]


class Edge:

    """
    Edge representation.

    An edge has a "start" and "stop" (x,y) vertices and an entry in the
    GET table of a polygon. The GET entry is a list of these values:

    [ymax, x_at_ymin, delta_x/delta_y]

    """

    def __init__(self, name=None, start=None, stop=None, next=None):
        self._start = None
        if start is not None:
            self._start = np.asarray(start)
        self._name = name
        self._stop = stop
        if stop is not None:
            self._stop = np.asarray(stop)
        self._next = next

        if self._stop is not None and self._start is not None:
            if self._start[1] < self._stop[1]:
                self._ymin = self._start[1]
                self._yminx = self._start[0]
            else:
                self._ymin = self._stop[1]
                self._yminx = self._stop[0]
            self._ymax = max(self._start[1], self._stop[1])
            self._xmin = min(self._start[0], self._stop[0])
            self._xmax = max(self._start[0], self._stop[1])
        else:
            self._ymin = None
            self._yminx = None
            self._ymax = None
            self._xmin = None
            self._xmax = None
        self.GET_entry = self.compute_GET_entry()

    @property
    def ymin(self):
        return self._ymin

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def ymax(self):
        return self._ymax

    def compute_GET_entry(self):
        """
        Compute the entry in the Global Edge Table

        [ymax, x@ymin, 1/m]

        """
        if self._start is None:
            entry = None
        else:
            earr = np.asarray([self._start, self._stop])
            if np.diff(earr[:, 1]).item() == 0:
                return None
            else:
                entry = [self._ymax, self._yminx,
                         (np.diff(earr[:, 0]) / np.diff(earr[:, 1])).item(), None]
        return entry

    def compute_AET_entry(self, edge):
        """
        Compute the entry for an edge in the current Active Edge Table

        [ymax, x_intersect, 1/m]
        note: currently 1/m is not used
        """
        x = self.intersection(edge)[0]
        return [self._ymax, x, self.GET_entry[2]]

    def __repr__(self):
        fmt = ""
        if self._name is not None:
            fmt += self._name
            next = self.next
            while next is not None:
                fmt += "-->"
                fmt += next._name
                next = next.next
        return fmt

    @property
    def next(self):
        return self._next

    @next.setter
    def next(self, edge):
        if self._name is None:
            self._name = edge._name
            self._stop = edge._stop
            self._start = edge._start
            self._next = edge.next
        else:
            self._next = edge

    def intersection(self, edge):
        u = self._stop - self._start
        v = edge._stop - edge._start
        w = self._start - edge._start
        eps = 1e2 * np.finfo(np.float).eps
        if np.allclose(np.cross(u, v), 0, rtol=0, atol=eps):
            return np.array(self._start)
        D = np.cross(u, v)
        return np.cross(v, w) / D * u + self._start

    def is_parallel(self, edge):
        u = self._stop - self._start
        v = edge._stop - edge._start
        if np.cross(u, v):
            return False
        else:
            return True
