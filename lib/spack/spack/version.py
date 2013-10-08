"""
This file implements Version and version-ish objects.  These are:

  Version
      A single version of a package.
  VersionRange
      A range of versions of a package.
  VersionList
      A list of Versions and VersionRanges.

All of these types support the following operations, which can
be called on any of the types:

  __eq__, __ne__, __lt__, __gt__, __ge__, __le__, __hash__
  __contains__
  overlaps
  merge
  concrete
      True if the Version, VersionRange or VersionList represents
      a single version.
"""
import os
import sys
import re
from bisect import bisect_left
from functools import total_ordering

import utils
from none_compare import *
import spack.error

# Valid version characters
VALID_VERSION = r'[A-Za-z0-9_.-]'

def int_if_int(string):
    """Convert a string to int if possible.  Otherwise, return a string."""
    try:
        return int(string)
    except ValueError:
        return string


def coerce_versions(a, b):
    """Convert both a and b to the 'greatest' type between them, in this order:
           Version < VersionRange < VersionList
       This is used to simplify comparison operations below so that we're always
       comparing things that are of the same type.
    """
    order = (Version, VersionRange, VersionList)
    ta, tb = type(a), type(b)

    def check_type(t):
        if t not in order:
            raise TypeError("coerce_versions cannot be called on %s" % t)
    check_type(ta)
    check_type(tb)

    if ta == tb:
        return (a, b)
    elif order.index(ta) > order.index(tb):
        if ta == VersionRange:
            return (a, VersionRange(b, b))
        else:
            return (a, VersionList([b]))
    else:
        if tb == VersionRange:
            return (VersionRange(a, a), b)
        else:
            return (VersionList([a]), b)


def coerced(method):
    """Decorator that ensures that argument types of a method are coerced."""
    def coercing_method(a, b):
        if type(a) == type(b) or a is None or b is None:
            return method(a, b)
        else:
            ca, cb = coerce_versions(a, b)
            return getattr(ca, method.__name__)(cb)
    return coercing_method


@total_ordering
class Version(object):
    """Class to represent versions"""
    def __init__(self, string):
        if not re.match(VALID_VERSION, string):
            raise ValueError("Bad characters in version string: %s" % string)

        # preserve the original string, but trimmed.
        string = string.strip()
        self.string = string

        # Split version into alphabetical and numeric segments
        segment_regex = r'[a-zA-Z]+|[0-9]+'
        segments = re.findall(segment_regex, string)
        self.version = tuple(int_if_int(seg) for seg in segments)

        # Store the separators from the original version string as well.
        # last element of separators is ''
        self.separators = tuple(re.split(segment_regex, string)[1:-1])


    def up_to(self, index):
        """Return a version string up to the specified component, exclusive.
           e.g., if this is 10.8.2, self.up_to(2) will return '10.8'.
        """
        return '.'.join(str(x) for x in self[:index])


    def lowest(self):
        return self


    def highest(self):
        return self


    def wildcard(self):
        """Create a regex that will match variants of this version string."""
        def a_or_n(seg):
            if type(seg) == int:
                return r'[0-9]+'
            else:
                return r'[a-zA-Z]+'

        version = self.version
        separators = ('',) + self.separators

        version += (version[-1],) * 2
        separators += (separators[-1],) * 2

        sep_res = [re.escape(sep) for sep in separators]
        seg_res = [a_or_n(seg) for seg in version]

        wc = seg_res[0]
        for i in xrange(1, len(sep_res)):
            wc += '(?:' + sep_res[i] + seg_res[i]
        wc += ')?' * (len(seg_res) - 1)
        return wc


    def __iter__(self):
        for v in self.version:
            yield v


    def __getitem__(self, idx):
        return tuple(self.version[idx])


    def __repr__(self):
        return self.string


    def __str__(self):
        return self.string


    @property
    def concrete(self):
        return self

    @coerced
    def __lt__(self, other):
        """Version comparison is designed for consistency with the way RPM
           does things.  If you need more complicated versions in installed
           packages, you should override your package's version string to
           express it more sensibly.
        """
        if other is None:
            return False

        # Coerce if other is not a Version
        # simple equality test first.
        if self.version == other.version:
            return False

        for a, b in zip(self.version, other.version):
            if a == b:
                continue
            else:
                # Numbers are always "newer" than letters.  This is for
                # consistency with RPM.  See patch #60884 (and details)
                # from bugzilla #50977 in the RPM project at rpm.org.
                # Or look at rpmvercmp.c if you want to see how this is
                # implemented there.
                if type(a) != type(b):
                    return type(b) == int
                else:
                    return a < b

        # If the common prefix is equal, the one with more segments is bigger.
        return len(self.version) < len(other.version)


    @coerced
    def __eq__(self, other):
        return (other is not None and
                type(other) == Version and self.version == other.version)


    def __ne__(self, other):
        return not (self == other)


    def __hash__(self):
        return hash(self.version)


    @coerced
    def __contains__(self, other):
        return self == other


    @coerced
    def overlaps(self, other):
        return self == other


    @coerced
    def merge(self, other):
        if self == other:
            return self
        else:
            return VersionList([self, other])


@total_ordering
class VersionRange(object):
    def __init__(self, start, end):
        if type(start) == str:
            start = Version(start)
        if type(end) == str:
            end = Version(end)

        self.start = start
        self.end = end
        if start and end and  end < start:
            raise ValueError("Invalid Version range: %s" % self)


    def lowest(self):
        return self.start


    def highest(self):
        return self.end


    @coerced
    def __lt__(self, other):
        """Sort VersionRanges lexicographically so that they are ordered first
           by start and then by end.  None denotes an open range, so None in
           the start position is less than everything except None, and None in
           the end position is greater than everything but None.
        """
        if other is None:
            return False

        return (none_low_lt(self.start, other.start) or
                (self.start == other.start and
                 none_high_lt(self.end, other.end)))


    @coerced
    def __eq__(self, other):
        return (other is not None and
                type(other) == VersionRange and
                self.start == other.start and self.end == other.end)


    def __ne__(self, other):
        return not (self == other)


    @property
    def concrete(self):
        return self.start if self.start == self.end else None


    @coerced
    def __contains__(self, other):
        return (none_low_ge(other.start, self.start) and
                none_high_le(other.end, self.end))


    @coerced
    def overlaps(self, other):
        return (other in self or self in other or
                ((self.start == None or other.end == None or
                  self.start <= other.end) and
                 (other.start == None or self.end == None or
                  other.start <= self.end)))


    @coerced
    def merge(self, other):
        return VersionRange(none_low_min(self.start, other.start),
                            none_high_max(self.end, other.end))


    def __hash__(self):
        return hash((self.start, self.end))


    def __repr__(self):
        return self.__str__()


    def __str__(self):
        out = ""
        if self.start:
            out += str(self.start)
        out += ":"
        if self.end:
            out += str(self.end)
        return out


@total_ordering
class VersionList(object):
    """Sorted, non-redundant list of Versions and VersionRanges."""
    def __init__(self, vlist=None):
        self.versions = []
        if vlist != None:
            vlist = list(vlist)
            for v in vlist:
                self.add(ver(v))


    def add(self, version):
        if type(version) in (Version, VersionRange):
            # This normalizes single-value version ranges.
            if version.concrete:
                version = version.concrete

            i = bisect_left(self, version)

            while i-1 >= 0 and version.overlaps(self[i-1]):
                version = version.merge(self[i-1])
                del self.versions[i-1]
                i -= 1

            while i < len(self) and version.overlaps(self[i]):
                version = version.merge(self[i])
                del self.versions[i]

            self.versions.insert(i, version)

        elif type(version) == VersionList:
            for v in version:
                self.add(v)

        else:
            raise TypeError("Can't add %s to VersionList" % type(version))


    @property
    def concrete(self):
        if len(self) == 1:
            return self[0].concrete
        else:
            return None


    def copy(self):
        return VersionList(self)


    def lowest(self):
        """Get the lowest version in the list."""
        if not self:
            return None
        else:
            return self[0].lowest()


    def highest(self):
        """Get the highest version in the list."""
        if not self:
            return None
        else:
            return self[-1].highest()


    @coerced
    def overlaps(self, other):
        if not other or not self:
            return False

        i = o = 0
        while i < len(self) and o < len(other):
            if self[i].overlaps(other[o]):
                return True
            elif self[i] < other[o]:
                i += 1
            else:
                o += 1
        return False


    @coerced
    def merge(self, other):
        return VersionList(self.versions + other.versions)


    @coerced
    def __contains__(self, other):
        if len(self) == 0:
            return False

        for version in other:
            i = bisect_left(self, other)
            if i == 0:
                if version not in self[0]:
                    return False
            elif all(version not in v for v in self[i-1:]):
                return False

        return True


    def __getitem__(self, index):
        return self.versions[index]


    def __iter__(self):
        for v in self.versions:
            yield v


    def __len__(self):
        return len(self.versions)


    @coerced
    def __eq__(self, other):
        return other is not None and self.versions == other.versions


    def __ne__(self, other):
        return not (self == other)


    @coerced
    def __lt__(self, other):
        return other is not None and self.versions < other.versions


    def __hash__(self):
        return hash(tuple(self.versions))


    def __str__(self):
        return ",".join(str(v) for v in self.versions)


    def __repr__(self):
        return str(self.versions)


def _string_to_version(string):
    """Converts a string to a Version, VersionList, or VersionRange.
       This is private.  Client code should use ver().
    """
    string = string.replace(' ','')

    if ',' in string:
        return VersionList(string.split(','))

    elif ':' in string:
        s, e = string.split(':')
        start = Version(s) if s else None
        end   = Version(e) if e else None
        return VersionRange(start, end)

    else:
        return Version(string)


def ver(obj):
    """Parses a Version, VersionRange, or VersionList from a string
       or list of strings.
    """
    t = type(obj)
    if t == list:
        return VersionList(obj)
    elif t == str:
        return _string_to_version(obj)
    elif t in (Version, VersionRange, VersionList):
        return obj
    else:
        raise TypeError("ver() can't convert %s to version!" % t)